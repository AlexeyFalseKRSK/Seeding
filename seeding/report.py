"""Утилиты генерации PDF-отчётов."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    CondPageBreak,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from seeding.config import (
    PDF_BBOX_COLOR_CLASS,
    PDF_BBOX_COLOR_MAIN,
    PDF_BBOX_THICKNESS,
    PDF_CROPS_PER_PAGE,
    PDF_CROP_MAX_HEIGHT_MM,
    PDF_CROP_MAX_WIDTH_MM,
    PDF_FONT_SCALE,
    PDF_IMAGE_MAX_HEIGHT_MM,
    PDF_IMAGE_MAX_WIDTH_MM,
    PDF_JPEG_QUALITY,
    PDF_LABEL_OFFSET_Y,
    PDF_SPACER_MM,
)
from seeding.models import ObjectImage, OriginalImage
from seeding.utils import rotate_bbox


def _np_to_pil(img: np.ndarray) -> Image.Image:
    """Преобразует изображение NumPy (BGR/gray) в ``PIL.Image``."""
    if img is None:
        raise ValueError("Изображение отсутствует")
    if img.ndim == 3 and img.shape[2] == 3:
        return Image.fromarray(img[:, :, ::-1])
    return Image.fromarray(img)


def _annotate_image(img: np.ndarray, objects: list[ObjectImage]) -> np.ndarray:
    """Рисует bbox объектов и частей на изображении для отчёта."""
    annotated = img.copy()
    for i, obj in enumerate(objects, start=1):
        if obj.bbox:
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                PDF_BBOX_COLOR_MAIN,
                PDF_BBOX_THICKNESS,
            )
            cv2.putText(
                annotated,
                str(i),
                (x1, max(y1 - PDF_LABEL_OFFSET_Y, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                PDF_FONT_SCALE,
                PDF_BBOX_COLOR_MAIN,
                PDF_BBOX_THICKNESS,
            )

        if obj.image_all_class and obj.bbox:
            k = getattr(obj, "rotation_k", 0) % 4
            h_rot, w_rot = obj.image[0].shape[:2] if obj.image else (0, 0)
            for cls in obj.image_all_class:
                if not cls.bbox:
                    continue
                lx1, ly1, lx2, ly2 = cls.bbox
                if k and h_rot and w_rot:
                    ux1, uy1, ux2, uy2 = rotate_bbox(
                        lx1,
                        ly1,
                        lx2,
                        ly2,
                        w_rot,
                        h_rot,
                        (-k) % 4,
                    )
                else:
                    ux1, uy1, ux2, uy2 = lx1, ly1, lx2, ly2

                x1 = obj.bbox[0] + ux1
                y1 = obj.bbox[1] + uy1
                x2 = obj.bbox[0] + ux2
                y2 = obj.bbox[1] + uy2
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    PDF_BBOX_COLOR_CLASS,
                    PDF_BBOX_THICKNESS,
                )
    return annotated


def _pil_to_buf(
    image: Image.Image,
    *,
    quality: int | None = None,
) -> io.BytesIO:
    """Кодирует PIL-изображение в JPEG-буфер в памяти."""
    if quality is None:
        quality = PDF_JPEG_QUALITY
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return buffer


def _object_to_pil(obj: ObjectImage) -> Image.Image | None:
    """Преобразует кроп объекта в ``PIL.Image`` при возможности."""
    if not obj.image:
        return None
    img = obj.image[0]
    if isinstance(img, np.ndarray):
        return _np_to_pil(img)
    if isinstance(img, Image.Image):
        return img
    return None


def _rl_image_from_pil(
    image: Image.Image,
    max_w: float,
    max_h: float,
) -> RLImage:
    """Создаёт изображение ReportLab с ограничением максимальных размеров."""
    aspect = image.height / float(image.width)
    width_pt = max_w
    height_pt = width_pt * aspect
    if height_pt > max_h:
        height_pt = max_h
        width_pt = height_pt / aspect
    return RLImage(_pil_to_buf(image), width=width_pt, height=height_pt)


def _estimate_table_height(row_count: int) -> float:
    """Оценивает высоту таблицы в points для условного разрыва страницы."""
    approx_row_height_mm = 6
    return max(row_count, 1) * approx_row_height_mm * mm


def create_pdf_report(data: OriginalImage, output_path: str) -> None:
    """Формирует PDF-отчёт с аннотированными страницами и кропами.

    Args:
        data: Структура с исходными изображениями и результатами детекции.
        output_path: Путь к итоговому PDF-файлу.
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    for idx, img in enumerate(data.images):
        objs: list[ObjectImage] = []
        if data.class_object_image and len(data.class_object_image) > idx:
            objs = data.class_object_image[idx]

        annotated_img = _annotate_image(img, objs)
        pil_img = _np_to_pil(annotated_img)

        story.append(Paragraph(f"Страница {idx + 1}", styles["Heading1"]))
        story.append(
            _rl_image_from_pil(
                pil_img,
                PDF_IMAGE_MAX_WIDTH_MM * mm,
                PDF_IMAGE_MAX_HEIGHT_MM * mm,
            )
        )
        story.append(Spacer(1, PDF_SPACER_MM * mm))

        table_data = [["№", "Класс", "Уверенность", "BBox"]]
        for i, obj in enumerate(objs, start=1):
            bbox = obj.bbox if obj.bbox else ("", "", "", "")
            table_data.append(
                [
                    str(i),
                    obj.class_name,
                    f"{obj.confidence:.2f}",
                    str(bbox),
                ]
            )

        table = Table(table_data, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(table)

        if objs:
            # Добавляем разрыв страницы только если таблица и первый блок
            # кропа не помещаются.
            table_height = _estimate_table_height(len(table_data))
            projected_block_height = table_height + (
                PDF_CROP_MAX_HEIGHT_MM + 15
            ) * mm
            story.append(CondPageBreak(projected_block_height))

        crops_on_page = 0
        for i, obj in enumerate(objs, start=1):
            pil_obj = _object_to_pil(obj)
            if pil_obj is None:
                continue

            crop_block = [
                Paragraph(f"Объект {i}", styles["Heading3"]),
                _rl_image_from_pil(
                    pil_obj,
                    PDF_CROP_MAX_WIDTH_MM * mm,
                    PDF_CROP_MAX_HEIGHT_MM * mm,
                ),
                Spacer(1, PDF_SPACER_MM * mm),
            ]

            story.append(CondPageBreak((PDF_CROP_MAX_HEIGHT_MM + 12) * mm))
            story.append(KeepTogether(crop_block))

            crops_on_page += 1
            if crops_on_page >= PDF_CROPS_PER_PAGE and i != len(objs):
                story.append(PageBreak())
                crops_on_page = 0

        if idx < len(data.images) - 1:
            story.append(PageBreak())

    doc.build(story)
