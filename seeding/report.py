"""Utilities for generating polished PDF reports."""

from __future__ import annotations

import io
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus import (
    Image as RLImage,
)

from seeding.config import (
    CLASS_DISPLAY_NAMES,
    PDF_BBOX_COLOR_CLASS,
    PDF_BBOX_COLOR_MAIN,
    PDF_BBOX_THICKNESS,
    PDF_FONT_SCALE,
    PDF_IMAGE_MAX_HEIGHT_MM,
    PDF_IMAGE_MAX_WIDTH_MM,
    PDF_JPEG_QUALITY,
    PDF_LABEL_OFFSET_Y,
    PDF_SPACER_MM,
)
from seeding.models import ObjectImage, OriginalImage
from seeding.utils import rotate_bbox

_FONT_REGISTRY_DONE = False
_FONT_REGULAR = "Helvetica"
_FONT_BOLD = "Helvetica-Bold"
_MASK_PALETTE = [
    (46, 125, 50),
    (21, 101, 192),
    (216, 67, 21),
    (123, 31, 162),
    (0, 137, 123),
    (198, 40, 40),
]


@dataclass(frozen=True)
class ReportSummary:
    """High-level report counts."""

    page_count: int
    object_count: int
    classified_object_count: int
    part_count: int
    uncertain_orientation_count: int
    part_counts: Counter[str]


def _register_fonts() -> tuple[str, str]:
    """Register a Unicode-capable font when one is available on the system."""
    global _FONT_REGISTRY_DONE, _FONT_REGULAR, _FONT_BOLD
    if _FONT_REGISTRY_DONE:
        return _FONT_REGULAR, _FONT_BOLD
    _FONT_REGISTRY_DONE = True

    candidates = [
        (
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
            "ArialSeeding",
            "ArialSeeding-Bold",
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            "DejaVuSeeding",
            "DejaVuSeeding-Bold",
        ),
    ]
    for regular_path, bold_path, regular_name, bold_name in candidates:
        if regular_path.exists() and bold_path.exists():
            pdfmetrics.registerFont(TTFont(regular_name, str(regular_path)))
            pdfmetrics.registerFont(TTFont(bold_name, str(bold_path)))
            _FONT_REGULAR = regular_name
            _FONT_BOLD = bold_name
            break
    return _FONT_REGULAR, _FONT_BOLD


def _styles():
    regular, bold = _register_fonts()
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName=bold,
            fontSize=24,
            leading=30,
            textColor=colors.HexColor("#172033"),
            spaceAfter=8 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["BodyText"],
            fontName=regular,
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=8 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontName=bold,
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#172033"),
            spaceBefore=3 * mm,
            spaceAfter=3 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallText",
            parent=styles["BodyText"],
            fontName=regular,
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#4b5563"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="CardTitle",
            parent=styles["BodyText"],
            fontName=bold,
            fontSize=10,
            leading=12,
            textColor=colors.HexColor("#172033"),
            spaceAfter=1.5 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CardMeta",
            parent=styles["BodyText"],
            fontName=regular,
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#4b5563"),
            spaceBefore=1 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Footer",
            parent=styles["BodyText"],
            fontName=regular,
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#6b7280"),
            alignment=TA_RIGHT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetricValue",
            parent=styles["BodyText"],
            fontName=bold,
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0f766e"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetricLabel",
            parent=styles["BodyText"],
            fontName=regular,
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4b5563"),
        )
    )
    for style in styles.byName.values():
        if hasattr(style, "fontName") and style.fontName in {"Helvetica", "Times-Roman"}:
            style.fontName = regular
    return styles


def _display_class_name(class_name: str | None) -> str:
    value = (class_name or "").strip().lower()
    return CLASS_DISPLAY_NAMES.get(value, class_name or "")


def _source_name(data: OriginalImage, page_index: int) -> str:
    if 0 <= page_index < len(data.source_files):
        return Path(data.source_files[page_index]).name
    return Path(data.file_path).name if data.file_path else f"page_{page_index + 1}"


def _page_objects(data: OriginalImage, page_index: int) -> list[ObjectImage]:
    if not data.class_object_image or page_index >= len(data.class_object_image):
        return []
    return data.class_object_image[page_index]


def _build_summary(data: OriginalImage) -> ReportSummary:
    objects = [
        obj
        for page_index in range(len(data.images))
        for obj in _page_objects(data, page_index)
    ]
    part_counts: Counter[str] = Counter()
    part_count = 0
    classified_object_count = 0
    uncertain_orientation_count = 0
    for obj in objects:
        if obj.image_all_class:
            classified_object_count += 1
            for part in obj.image_all_class:
                part_count += 1
                part_counts[_display_class_name(part.class_name)] += 1
        if obj.orientation_uncertain:
            uncertain_orientation_count += 1
    return ReportSummary(
        page_count=len(data.images),
        object_count=len(objects),
        classified_object_count=classified_object_count,
        part_count=part_count,
        uncertain_orientation_count=uncertain_orientation_count,
        part_counts=part_counts,
    )


def _np_to_pil(img: np.ndarray) -> Image.Image:
    if img is None:
        raise ValueError("Image is missing")
    if img.ndim == 3 and img.shape[2] == 3:
        return Image.fromarray(img[:, :, ::-1])
    return Image.fromarray(img)


def _annotate_image(img: np.ndarray, objects: list[ObjectImage]) -> np.ndarray:
    """Draw object and part boxes on a page image."""
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
            for part in obj.image_all_class:
                if not part.bbox:
                    continue
                lx1, ly1, lx2, ly2 = part.bbox
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

                cv2.rectangle(
                    annotated,
                    (obj.bbox[0] + ux1, obj.bbox[1] + uy1),
                    (obj.bbox[0] + ux2, obj.bbox[1] + uy2),
                    PDF_BBOX_COLOR_CLASS,
                    PDF_BBOX_THICKNESS,
                )
    return annotated


def _pil_to_buf(image: Image.Image, *, quality: int | None = None) -> io.BytesIO:
    if quality is None:
        quality = PDF_JPEG_QUALITY
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return buffer


def _object_to_pil(obj: ObjectImage) -> Image.Image | None:
    if not obj.image:
        return None
    img = obj.image[0]
    if isinstance(img, np.ndarray):
        return _np_to_pil(img)
    if isinstance(img, Image.Image):
        return img
    return None


def _part_color(part_index: int) -> tuple[int, int, int]:
    return _MASK_PALETTE[part_index % len(_MASK_PALETTE)]


def _mask_from_part(part, crop_shape: tuple[int, int]) -> np.ndarray | None:
    crop_height, crop_width = crop_shape
    if part.mask_bitmap is not None:
        mask = np.asarray(part.mask_bitmap)
        if mask.shape[:2] != (crop_height, crop_width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (crop_width, crop_height),
                interpolation=cv2.INTER_NEAREST,
            )
        return (mask > 0).astype(np.uint8) * 255

    if part.mask_polygon is not None:
        polygon = np.asarray(part.mask_polygon, dtype=np.int32)
        if polygon.ndim == 2 and polygon.shape[0] >= 3:
            mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            return mask
    return None


def _object_overlay_to_pil(obj: ObjectImage) -> Image.Image | None:
    """Convert object crop to PIL with colored part masks and contours overlaid."""
    if not obj.image:
        return None
    crop = obj.image[0]
    if isinstance(crop, Image.Image):
        crop_array = np.array(crop.convert("RGB"))[:, :, ::-1].copy()
    elif isinstance(crop, np.ndarray):
        crop_array = crop.copy()
    else:
        return None
    if crop_array.ndim != 3 or crop_array.shape[2] != 3:
        crop_array = cv2.cvtColor(crop_array, cv2.COLOR_GRAY2BGR)

    overlay = crop_array.copy()
    crop_height, crop_width = crop_array.shape[:2]
    for part_index, part in enumerate(obj.image_all_class or []):
        color = _part_color(part_index)
        mask = _mask_from_part(part, (crop_height, crop_width))
        if mask is not None and np.any(mask):
            color_layer = np.zeros_like(crop_array)
            color_layer[:, :] = color
            overlay = np.where(mask[:, :, None] > 0, color_layer, overlay)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(crop_array, contours, -1, color, 2)
        if part.bbox:
            x1, y1, x2, y2 = part.bbox
            cv2.rectangle(crop_array, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                crop_array,
                str(part_index + 1),
                (x1, max(y1 - 4, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    blended = cv2.addWeighted(overlay, 0.32, crop_array, 0.68, 0)
    return _np_to_pil(blended)


def _legend_rows(obj: ObjectImage) -> list[list[str]]:
    rows = [["№", "Часть"]]
    for part_index, part in enumerate(obj.image_all_class or []):
        rows.append([str(part_index + 1), _display_class_name(part.class_name)])
    return rows


def _parts_inline(obj: ObjectImage) -> str:
    parts = obj.image_all_class or []
    if not parts:
        return "части не сегментированы"
    counts = Counter(_display_class_name(part.class_name) for part in parts)
    return ", ".join(f"{name}: {count}" for name, count in counts.most_common())


def _rl_image_from_pil(image: Image.Image, max_w: float, max_h: float) -> RLImage:
    aspect = image.height / float(max(image.width, 1))
    width_pt = max_w
    height_pt = width_pt * aspect
    if height_pt > max_h:
        height_pt = max_h
        width_pt = height_pt / aspect
    return RLImage(_pil_to_buf(image), width=width_pt, height=height_pt)


def _table(data, *, col_widths=None) -> Table:
    table = Table(data, hAlign="LEFT", colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7f5f1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#172033")),
                ("FONTNAME", (0, 0), (-1, 0), _FONT_BOLD),
                ("FONTNAME", (0, 1), (-1, -1), _FONT_REGULAR),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _metric_table(summary: ReportSummary, styles) -> Table:
    cells = [
        [
            Paragraph(str(summary.page_count), styles["MetricValue"]),
            Paragraph("страниц", styles["MetricLabel"]),
        ],
        [
            Paragraph(str(summary.object_count), styles["MetricValue"]),
            Paragraph("растений", styles["MetricLabel"]),
        ],
        [
            Paragraph(str(summary.classified_object_count), styles["MetricValue"]),
            Paragraph("с сегментацией", styles["MetricLabel"]),
        ],
        [
            Paragraph(str(summary.part_count), styles["MetricValue"]),
            Paragraph("частей", styles["MetricLabel"]),
        ],
    ]
    table = Table(
        [cells],
        colWidths=[38 * mm, 38 * mm, 38 * mm, 38 * mm],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#e5e7eb")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def _summary_section(data: OriginalImage, styles) -> list:
    summary = _build_summary(data)
    story: list = [
        Paragraph("Отчет Seeding", styles["ReportTitle"]),
        Paragraph(
            f"Сформировано: {datetime.now().strftime('%d.%m.%Y %H:%M')}<br/>"
            f"Исходный файл: {Path(data.file_path).name if data.file_path else '-'}",
            styles["ReportSubtitle"],
        ),
        _metric_table(summary, styles),
        Spacer(1, 5 * mm),
    ]

    if summary.part_counts:
        story.append(Paragraph("Сводка по классам частей", styles["SectionTitle"]))
        rows = [["Класс", "Количество"]]
        rows.extend([name, str(count)] for name, count in summary.part_counts.most_common())
        story.append(_table(rows, col_widths=[90 * mm, 35 * mm]))
        story.append(Spacer(1, 4 * mm))

    warnings = []
    if summary.object_count and summary.classified_object_count < summary.object_count:
        warnings.append(
            f"Не все растения сегментированы: {summary.classified_object_count} из {summary.object_count}."
        )
    if summary.uncertain_orientation_count:
        warnings.append(
            f"Есть объекты с неопределенной ориентацией: {summary.uncertain_orientation_count}."
        )
    if warnings:
        story.append(Paragraph("Контроль качества", styles["SectionTitle"]))
        for warning in warnings:
            story.append(Paragraph(f"• {warning}", styles["SmallText"]))
        story.append(Spacer(1, 4 * mm))
    return story


def _object_rows(objects: list[ObjectImage]) -> list[list[str]]:
    rows = [["№", "Класс", "Уверенность", "Части", "Ориентация"]]
    for index, obj in enumerate(objects, start=1):
        rows.append(
            [
                str(index),
                _display_class_name(obj.class_name),
                f"{obj.confidence:.2f}",
                _parts_inline(obj),
                "проверить" if obj.orientation_uncertain else "ok",
            ]
        )
    return rows


def _part_rows(obj: ObjectImage) -> list[list[str]]:
    rows = [["Часть", "Уверенность", "BBox"]]
    for part in obj.image_all_class or []:
        rows.append(
            [
                _display_class_name(part.class_name),
                f"{part.confidence:.2f}",
                str(part.bbox or ("", "", "", "")),
            ]
        )
    return rows


def _object_card(obj: ObjectImage, index: int, styles) -> Table | None:
    pil_obj = _object_overlay_to_pil(obj) or _object_to_pil(obj)
    if pil_obj is None:
        return None

    title = f"Растение {index}"
    if obj.orientation_uncertain:
        title += " · проверить ориентацию"
    image = _rl_image_from_pil(pil_obj, 73 * mm, 48 * mm)
    meta = Paragraph(
        f"Уверенность: {obj.confidence:.2f}<br/>"
        f"Части: {_parts_inline(obj)}",
        styles["CardMeta"],
    )
    card = Table(
        [[Paragraph(title, styles["CardTitle"])], [image], [meta]],
        colWidths=[78 * mm],
        hAlign="LEFT",
    )
    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
                ("BOX", (0, 0), (-1, -1), 0.55, colors.HexColor("#d7dee8")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.35, colors.HexColor("#e5e7eb")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return card


def _object_cards_grid(objects: list[ObjectImage], styles) -> Table | None:
    cards = [
        card
        for index, obj in enumerate(objects, start=1)
        if (card := _object_card(obj, index, styles)) is not None
    ]
    if not cards:
        return None

    rows = []
    for offset in range(0, len(cards), 2):
        row = cards[offset : offset + 2]
        if len(row) == 1:
            row.append("")
        rows.append(row)

    grid = Table(
        rows,
        colWidths=[82 * mm, 82 * mm],
        hAlign="LEFT",
    )
    grid.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return grid


def _page_section(data: OriginalImage, page_index: int, styles) -> list:
    img = data.images[page_index]
    objects = _page_objects(data, page_index)
    annotated_img = _annotate_image(img, objects)
    pil_img = _np_to_pil(annotated_img)

    story: list = [
        Paragraph(
            f"Страница {page_index + 1}: {_source_name(data, page_index)}",
            styles["SectionTitle"],
        ),
        _rl_image_from_pil(
            pil_img,
            PDF_IMAGE_MAX_WIDTH_MM * mm,
            PDF_IMAGE_MAX_HEIGHT_MM * mm,
        ),
        Spacer(1, PDF_SPACER_MM * mm),
    ]

    if objects:
        story.append(_table(_object_rows(objects), col_widths=[10 * mm, 28 * mm, 24 * mm, 70 * mm, 24 * mm]))
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("Растения с масками", styles["SectionTitle"]))
        cards_grid = _object_cards_grid(objects, styles)
        if cards_grid is not None:
            story.append(CondPageBreak(70 * mm))
            story.append(cards_grid)
    else:
        story.append(Paragraph("На странице нет найденных растений.", styles["SmallText"]))
    story.append(Spacer(1, 4 * mm))
    return story


def _footer(canvas, doc) -> None:
    canvas.saveState()
    regular, _ = _register_fonts()
    canvas.setFont(regular, 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(A4[0] - 14 * mm, 10 * mm, f"Seeding • стр. {doc.page}")
    canvas.restoreState()


def create_pdf_report(data: OriginalImage, output_path: str) -> None:
    """Create a PDF report with summary, annotated pages, tables, and crops."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title="Seeding report",
        author="Seeding",
    )
    story: list = _summary_section(data, styles)

    if data.images:
        story.append(PageBreak())
    for page_index in range(len(data.images)):
        story.extend(_page_section(data, page_index, styles))
        if page_index < len(data.images) - 1:
            story.append(PageBreak())

    if not data.images:
        story.append(Paragraph("Нет изображений для отчета.", styles["SmallText"]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
