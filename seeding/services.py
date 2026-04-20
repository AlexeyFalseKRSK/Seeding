"""Функции обработки изображений, детекции, классификации и отчетов."""

from __future__ import annotations

import numpy as np

from seeding.mask_refiner import (
    bitmap_to_polygon,
    polygon_to_bitmap,
    refine_mask_bitmap,
    rotate_bitmap,
)
from seeding.models import (
    AllClassImage,
    AppState,
    ObjectImage,
    OriginalImage,
    RotateSelectionResult,
    SelectionPayload,
)
from seeding.report import create_pdf_report
from seeding.utils import (
    clip_bbox_to_image,
    rotate_image_and_boxes,
    rotate_polygon_points,
    simple_nms,
)


def _iter_page_objects(
    image_storage: OriginalImage,
    page_index: int,
) -> list[ObjectImage]:
    """Возвращает объекты указанной страницы или пустой список при их отсутствии."""
    if (
        page_index < 0
        or not image_storage.class_object_image
        or page_index >= len(image_storage.class_object_image)
    ):
        return []
    return image_storage.class_object_image[page_index]


def refresh_page_crops(
    image_storage: OriginalImage,
    page_index: int,
    *,
    rotate_k: int,
    clear_classification: bool,
) -> None:
    """Пересобирает кропы объектов страницы по текущим bbox и настройкам поворота."""
    page_objects = _iter_page_objects(image_storage, page_index)
    if not page_objects:
        return

    base_img = image_storage.images[page_index]
    if base_img is None:
        return
    height, width = base_img.shape[:2]

    for obj in page_objects:
        if not obj.bbox:
            continue

        clipped = clip_bbox_to_image(obj.bbox, width, height)
        if clipped is None:
            obj.bbox = None
            obj.image = []
            if clear_classification:
                obj.image_all_class = None
            continue

        x1, y1, x2, y2 = clipped
        obj.bbox = clipped
        crop = base_img[y1:y2, x1:x2].copy()

        rotation_k = 0
        if crop.shape[1] > crop.shape[0]:
            crop = np.rot90(crop, k=rotate_k)
            rotation_k = rotate_k

        obj.rotation_k = rotation_k
        obj.image = [crop]
        if clear_classification:
            obj.image_all_class = None


def rotate_page(
    image_storage: OriginalImage,
    page_index: int,
    *,
    angle: float,
    rotate_k: int,
) -> np.ndarray:
    """Поворачивает страницу, обновляет bbox объектов и пересобирает их кропы."""
    page_image = image_storage.images[page_index]
    if page_image is None:
        raise ValueError("Изображение страницы пустое")

    page_objects = _iter_page_objects(image_storage, page_index)
    object_map: list[int] = []
    boxes_to_rotate: list[tuple[int, int, int, int]] = []
    for obj_idx, obj in enumerate(page_objects):
        if obj.bbox:
            boxes_to_rotate.append(obj.bbox)
            object_map.append(obj_idx)

    rotated_image, rotated_boxes = rotate_image_and_boxes(
        page_image,
        boxes_to_rotate,
        angle,
    )
    image_storage.images[page_index] = rotated_image

    for mapped_idx, obj_idx in enumerate(object_map):
        page_objects[obj_idx].bbox = rotated_boxes[mapped_idx]

    refresh_page_crops(
        image_storage,
        page_index,
        rotate_k=rotate_k,
        clear_classification=True,
    )
    return rotated_image


def rotate_crop(
    image_storage: OriginalImage,
    page_index: int,
    crop_index: int,
    *,
    angle: float,
    rotate_k: int,
) -> np.ndarray:
    """Поворачивает кроп объекта, его bbox частей и связанные маски сегментации."""
    page_objects = _iter_page_objects(image_storage, page_index)
    obj = page_objects[crop_index]
    if not obj.image or obj.image[0] is None:
        raise ValueError("Изображение кропа пустое")

    crop = obj.image[0]
    crop_height, crop_width = crop.shape[:2]
    class_map: list[int] = []
    class_boxes: list[tuple[int, int, int, int]] = []
    if obj.image_all_class:
        for class_idx, cls_obj in enumerate(obj.image_all_class):
            if cls_obj.bbox:
                class_boxes.append(cls_obj.bbox)
                class_map.append(class_idx)

    rotated_crop, rotated_class_boxes = rotate_image_and_boxes(
        crop,
        class_boxes,
        angle,
    )
    obj.image[0] = rotated_crop
    obj.rotation_k = (obj.rotation_k + rotate_k) % 4

    for mapped_idx, class_idx in enumerate(class_map):
        obj.image_all_class[class_idx].bbox = rotated_class_boxes[mapped_idx]
    if obj.image_all_class:
        for cls_obj in obj.image_all_class:
            if cls_obj.mask_polygon is not None:
                cls_obj.mask_polygon = rotate_polygon_points(
                    cls_obj.mask_polygon,
                    crop_width,
                    crop_height,
                    angle,
                )
            if cls_obj.mask_bitmap is not None:
                cls_obj.mask_bitmap = rotate_bitmap(cls_obj.mask_bitmap, angle)
    return rotated_crop


def rotate_selection(
    state: AppState,
    item_data: SelectionPayload,
    *,
    angle: float,
    rotate_k: int,
) -> RotateSelectionResult | None:
    """Поворачивает выбранный элемент и возвращает данные для перерисовки."""
    item_type = item_data.get("type")
    if item_type in ("original", "pdf"):
        page_index = int(item_data["index"])
        if page_index < 0 or page_index >= len(state.image_storage.images):
            return None
        rotated = rotate_page(
            state.image_storage,
            page_index,
            angle=angle,
            rotate_k=rotate_k,
        )
        state.active_image_index = page_index
        return RotateSelectionResult(
            target="page",
            page_index=page_index,
            image=rotated,
        )

    if item_type == "seeding":
        page_index = int(item_data["parent_index"])
        crop_index = int(item_data["index"])
        if page_index < 0 or page_index >= len(state.image_storage.images):
            return None
        page_objects = _iter_page_objects(state.image_storage, page_index)
        if crop_index < 0 or crop_index >= len(page_objects):
            return None
        rotated = rotate_crop(
            state.image_storage,
            page_index,
            crop_index,
            angle=angle,
            rotate_k=rotate_k,
        )
        state.active_image_index = page_index
        return RotateSelectionResult(
            target="crop",
            page_index=page_index,
            crop_index=crop_index,
            image=rotated,
        )

    return None


def build_detected_objects(
    image: np.ndarray,
    results,
    *,
    detection_class_names: tuple[str, ...] | list[str] | set[str] | None,
    iou_threshold: float,
    rotate_k: int,
) -> list[ObjectImage]:
    """Фильтрует детекции, применяет NMS и формирует список объектов сеянцев."""
    if image is None or results is None or not results:
        return []

    parsed: list[dict] = []
    allowed_class_names = (
        {str(value).lower() for value in detection_class_names}
        if detection_class_names
        else None
    )
    height, width = image.shape[:2]

    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = str(results[0].names[class_id]).lower()
        if allowed_class_names is not None and class_name not in allowed_class_names:
            continue

        score = float(box.conf)
        x_center, y_center, box_width, box_height = box.xywh[0].cpu().numpy()
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        clipped = clip_bbox_to_image((x1, y1, x2, y2), width, height)
        if clipped is None:
            continue

        cx1, cy1, cx2, cy2 = clipped
        parsed.append(
            {
                "class_name": class_name,
                "score": score,
                "bbox": (cx1, cy1, cx2, cy2),
            }
        )

    if not parsed:
        return []

    kept_indices: list[int] = []
    indices_by_class: dict[str, list[int]] = {}
    for idx, item in enumerate(parsed):
        indices_by_class.setdefault(str(item["class_name"]), []).append(idx)
    for class_indices in indices_by_class.values():
        boxes = [list(parsed[idx]["bbox"]) for idx in class_indices]
        scores = [float(parsed[idx]["score"]) for idx in class_indices]
        kept_local = simple_nms(boxes, scores, iou_threshold=iou_threshold)
        kept_indices.extend(class_indices[idx] for idx in kept_local)
    kept_indices.sort(key=lambda idx: float(parsed[idx]["score"]), reverse=True)

    objects: list[ObjectImage] = []
    for idx in kept_indices:
        item = parsed[idx]
        x1, y1, x2, y2 = item["bbox"]
        crop = image[y1:y2, x1:x2].copy()

        rotation_k = 0
        if crop.shape[1] > crop.shape[0]:
            crop = np.rot90(crop, k=rotate_k)
            rotation_k = rotate_k

        objects.append(
            ObjectImage(
                class_name=item["class_name"],
                confidence=float(item["score"]),
                image=[crop],
                bbox=(x1, y1, x2, y2),
                rotation_k=rotation_k,
            )
        )
    return objects


def run_detection(
    state: AppState,
    page_index: int,
    results,
    *,
    detection_class_names: tuple[str, ...] | list[str] | set[str] | None,
    iou_threshold: float,
    rotate_k: int,
) -> list[ObjectImage]:
    """Парсит предсказания детекции и записывает объекты в состояние приложения."""
    image = state.image_storage.images[page_index]
    objects = build_detected_objects(
        image,
        results,
        detection_class_names=detection_class_names,
        iou_threshold=iou_threshold,
        rotate_k=rotate_k,
    )

    if state.image_storage.class_object_image is None:
        state.image_storage.class_object_image = []
    while len(state.image_storage.class_object_image) < len(state.image_storage.images):
        state.image_storage.class_object_image.append([])
    state.image_storage.class_object_image[page_index] = objects
    return objects


def _clip_mask_polygon(
    mask_polygon: np.ndarray | None,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Ограничивает полигон маски размерами изображения части."""
    if mask_polygon is None or width <= 0 or height <= 0:
        return None
    polygon = np.asarray(mask_polygon, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        return None
    clipped = polygon.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(width - 1))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(height - 1))
    return np.ascontiguousarray(clipped)


def build_classified_parts(
    crop_image: np.ndarray,
    results,
) -> list[AllClassImage]:
    """Собирает части растения из результатов модели для одного кропа объекта."""
    if crop_image is None or results is None or not results:
        return []

    crop_height, crop_width = crop_image.shape[:2]
    parts: list[AllClassImage] = []
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf)
            class_id = int(box.cls)
            class_name = result.names[class_id]
            coords = box.xyxy[0].cpu().numpy().astype(int)
            local_bbox = clip_bbox_to_image(
                (
                    int(coords[0]),
                    int(coords[1]),
                    int(coords[2]),
                    int(coords[3]),
                ),
                crop_width,
                crop_height,
            )
            if local_bbox is None:
                continue

            lx1, ly1, lx2, ly2 = local_bbox
            part_image = crop_image[ly1:ly2, lx1:lx2].copy()
            mask_polygon = _clip_mask_polygon(
                getattr(box, "mask_polygon", None),
                crop_width,
                crop_height,
            )

            # Строим попиксельный bitmap: растеризация полигона → Otsu-уточнение
            mask_bitmap: np.ndarray | None = None
            coarse_bitmap = polygon_to_bitmap(
                mask_polygon, crop_height, crop_width
            )
            if coarse_bitmap is not None:
                refined_bitmap = refine_mask_bitmap(crop_image, coarse_bitmap)
                if refined_bitmap is not None:
                    mask_bitmap = refined_bitmap
                    # Обновляем полигон как наибольший контур уточнённой маски
                    refined_polygon = bitmap_to_polygon(refined_bitmap)
                    if refined_polygon is not None:
                        mask_polygon = _clip_mask_polygon(
                            refined_polygon, crop_width, crop_height
                        )

            parts.append(
                AllClassImage(
                    class_name=class_name,
                    confidence=confidence,
                    image=part_image,
                    bbox=local_bbox,
                    mask_polygon=mask_polygon,
                    mask_bitmap=mask_bitmap,
                )
            )
    return parts


def run_classification_for_selection(
    state: AppState,
    page_index: int,
    crop_index: int,
    results,
) -> list[AllClassImage]:
    """Парсит предсказания классификации для выбранного кропа."""
    page_objects = state.image_storage.class_object_image or []
    if page_index >= len(page_objects):
        return []
    objects = page_objects[page_index]
    if crop_index >= len(objects):
        return []
    selected_object = objects[crop_index]
    if not selected_object.image:
        return []

    parts = build_classified_parts(
        selected_object.image[0],
        results,
    )
    selected_object.image_all_class = parts
    return parts


def generate_report(state: AppState, output_path: str) -> str:
    """Создает PDF-отчет и сохраняет путь в состоянии приложения."""
    create_pdf_report(state.image_storage, output_path)
    state.last_report_path = output_path
    return output_path


__all__ = [
    "build_classified_parts",
    "build_detected_objects",
    "generate_report",
    "refresh_page_crops",
    "rotate_crop",
    "rotate_page",
    "rotate_selection",
    "run_classification_for_selection",
    "run_detection",
]
