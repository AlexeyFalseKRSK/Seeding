"""Сохранение и загрузка сессии анализа в/из SQLite."""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np

from seeding import database
from seeding.models import AllClassImage, AppState, ObjectImage, OriginalImage


def save_session(state: AppState) -> int:
    """Сохраняет или обновляет текущую сессию. Возвращает session_id."""
    orig = state.image_storage
    if state.current_user_id is None:
        raise ValueError("Нельзя сохранить сессию: пользователь не авторизован.")
    if not orig.file_path:
        raise ValueError("Нельзя сохранить сессию: не выбран исходный файл.")

    if state.session_id is None:
        session_id = database.insert_analysis_session(
            user_id=state.current_user_id,
            source_path=orig.file_path,
            page_count=len(orig.images),
            calibration_ppm=state.pixels_per_mm or None,
            report_path=state.last_report_path or None,
        )
        state.session_id = session_id
    else:
        database.update_session(
            session_id=state.session_id,
            calibration_ppm=state.pixels_per_mm or None,
            report_path=state.last_report_path or None,
        )
        database.delete_detections_by_session(state.session_id)

    source_files = list(orig.source_files or [])
    if len(source_files) < len(orig.images):
        source_files.extend([orig.file_path] * (len(orig.images) - len(source_files)))
    page_rotation_k = list(orig.page_rotation_k or [])
    if len(page_rotation_k) < len(orig.images):
        page_rotation_k.extend([0] * (len(orig.images) - len(page_rotation_k)))
    database.replace_session_sources(
        state.session_id,
        source_files[: len(orig.images)],
        page_rotation_k[: len(orig.images)],
    )
    _insert_all_detections(state.session_id, orig)

    return state.session_id


def _insert_all_detections(session_id: int, orig: OriginalImage) -> None:
    pages = orig.class_object_image or []
    for page_idx, objects in enumerate(pages):
        for obj_idx, obj in enumerate(objects):
            bbox = obj.bbox or (0, 0, 0, 0)
            det_id = database.insert_detection(
                session_id=session_id,
                page_index=page_idx,
                object_index=obj_idx,
                bbox=bbox,
                confidence=obj.confidence,
                rotation_deg=float(obj.rotation_k) * 90,
                orientation_uncertain=obj.orientation_uncertain,
            )
            for part in obj.image_all_class or []:
                polygon_json = None
                if part.mask_polygon is not None:
                    polygon_json = json.dumps(part.mask_polygon.tolist())
                database.insert_plant_part(
                    detection_id=det_id,
                    class_name=part.class_name,
                    confidence=part.confidence,
                    bbox=part.bbox,
                    polygon_json=polygon_json,
                    mask_bitmap=_encode_mask_bitmap(part.mask_bitmap),
                )


def load_session(session_id: int) -> AppState | None:
    """Загружает сессию из БД. Возвращает None если сессия не найдена."""
    row = database.fetch_session_by_id(session_id)
    if row is None:
        return None

    source_path = row["source_path"]
    source_rows = database.fetch_session_sources(session_id)
    if source_rows:
        source_files = [source["source_path"] for source in source_rows]
        page_rotation_k = [
            int(round(float(source["rotation_deg"] or 0.0) / 90.0)) % 4
            for source in source_rows
        ]
    else:
        page_count = int(row["page_count"] or 0)
        source_files = [source_path for _ in range(max(1, page_count))]
        page_rotation_k = [0 for _ in source_files]
    missing_sources = [path for path in source_files if not Path(path).exists()]

    detections = database.fetch_detections_by_session(session_id)

    pages: dict[int, list[ObjectImage]] = {}
    for det in detections:
        parts = database.fetch_parts_by_detection(det["id"])
        all_class = [_row_to_part(p) for p in parts]
        obj = ObjectImage(
            class_name="seeding",
            confidence=det["confidence"] or 0.0,
            image=[],
            image_all_class=all_class,
            bbox=_bbox_from_row(det),
            rotation_k=int((det["rotation_deg"] or 0) / 90) % 4,
            orientation_uncertain=bool(det["orientation_uncertain"]),
        )
        pages.setdefault(det["page_index"], []).append(obj)

    page_list = [pages.get(i, []) for i in range(max(pages.keys(), default=-1) + 1)]

    orig = OriginalImage(
        file_path=source_path,
        source_files=source_files,
        images=[],
        class_object_image=page_list,
        page_rotation_k=page_rotation_k,
    )

    state = AppState(
        image_storage=orig,
        pixels_per_mm=row["calibration_ppm"] or 0.0,
        last_report_path=row["report_path"] or "",
        session_id=session_id,
        current_user_id=row["user_id"],
    )

    if missing_sources:
        state._missing_source = "\n".join(dict.fromkeys(missing_sources))

    return state


def record_edit(
    state: AppState,
    target_type: str,
    target_id: int,
    field: str,
    value_before,
    value_after,
) -> None:
    """Записывает ручную правку в журнал edit_history."""
    database.insert_edit_history(
        user_id=state.current_user_id,
        target_type=target_type,
        target_id=target_id,
        field=field,
        value_before=json.dumps(value_before),
        value_after=json.dumps(value_after),
    )


def _row_to_part(row) -> AllClassImage:
    polygon = None
    if row["polygon_json"]:
        polygon = np.array(json.loads(row["polygon_json"]))
    return AllClassImage(
        class_name=row["class_name"],
        confidence=row["confidence"] or 0.0,
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        bbox=_bbox_from_row(row),
        mask_polygon=polygon,
        mask_bitmap=_decode_mask_bitmap(row["mask_bitmap"]),
    )


def _bbox_from_row(row) -> tuple[int, int, int, int] | None:
    if row["bbox_x"] is None:
        return None
    return (int(row["bbox_x"]), int(row["bbox_y"]),
            int(row["bbox_w"]), int(row["bbox_h"]))


def _encode_mask_bitmap(mask_bitmap: np.ndarray | None) -> bytes | None:
    if mask_bitmap is None:
        return None
    buffer = BytesIO()
    np.save(buffer, np.asarray(mask_bitmap, dtype=np.uint8), allow_pickle=False)
    return buffer.getvalue()


def _decode_mask_bitmap(payload: bytes | None) -> np.ndarray | None:
    if payload is None:
        return None
    buffer = BytesIO(payload)
    return np.load(buffer, allow_pickle=False)
