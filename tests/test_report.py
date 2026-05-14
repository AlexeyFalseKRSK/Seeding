import fitz
import numpy as np

from seeding.models import AllClassImage, ObjectImage, OriginalImage
from seeding.report import (
    _annotate_image,
    _build_summary,
    _object_overlay_to_pil,
    create_pdf_report,
)


def test_create_pdf_report_contains_summary_and_page_text(tmp_path):
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    part = AllClassImage(
        class_name="root",
        confidence=0.83,
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        bbox=(2, 3, 20, 25),
        mask_polygon=np.array(
            [[2, 3], [20, 3], [20, 25], [2, 25]],
            dtype=np.float32,
        ),
    )
    obj = ObjectImage(
        class_name="seeding",
        confidence=0.91,
        image=[np.zeros((30, 30, 3), dtype=np.uint8)],
        image_all_class=[part],
        bbox=(10, 12, 50, 60),
    )
    data = OriginalImage(
        file_path="input.png",
        source_files=["input.png"],
        images=[image],
        class_object_image=[[obj]],
    )
    output = tmp_path / "report.pdf"

    create_pdf_report(data, str(output))

    assert output.is_file()
    assert output.stat().st_size > 0
    doc = fitz.open(str(output))
    try:
        text = "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()
    assert "Отчет Seeding" in text
    assert "Страница 1" in text
    assert "Растение 1" in text


def test_create_pdf_report_without_objects_has_no_extra_blank_pages(tmp_path):
    data = OriginalImage(
        file_path="input.png",
        source_files=["input.png", "input.png"],
        images=[
            np.zeros((30, 30, 3), dtype=np.uint8),
            np.zeros((30, 30, 3), dtype=np.uint8),
        ],
        class_object_image=[[], []],
    )
    output = tmp_path / "report_no_objects.pdf"

    create_pdf_report(data, str(output))

    doc = fitz.open(str(output))
    try:
        assert doc.page_count == 3
        text = "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()
    assert "На странице нет найденных растений" in text


def test_create_pdf_report_uses_compact_mask_cards(tmp_path):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    objects = []
    for index in range(6):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[6:24, 8:26] = 255
        part = AllClassImage(
            class_name="root",
            confidence=0.8,
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            bbox=(8, 6, 26, 24),
            mask_bitmap=mask,
        )
        objects.append(
            ObjectImage(
                class_name="seeding",
                confidence=0.9,
                image=[np.zeros((32, 32, 3), dtype=np.uint8)],
                image_all_class=[part],
                bbox=(10 + index * 8, 12, 42 + index * 8, 44),
            )
        )
    data = OriginalImage(
        file_path="input.png",
        source_files=["input.png"],
        images=[image],
        class_object_image=[objects],
    )
    output = tmp_path / "compact_report.pdf"

    create_pdf_report(data, str(output))

    doc = fitz.open(str(output))
    try:
        text = "\n".join(page.get_text() for page in doc)
        page_count = doc.page_count
    finally:
        doc.close()

    assert "Растения с масками" in text
    assert "BBox" not in text
    assert page_count <= 3


def test_build_summary_counts_parts_and_uncertain_orientation():
    obj = ObjectImage(
        class_name="seeding",
        confidence=0.7,
        image=[],
        image_all_class=[
            AllClassImage("root", 0.8, np.zeros((1, 1, 3), dtype=np.uint8)),
            AllClassImage("stem", 0.6, np.zeros((1, 1, 3), dtype=np.uint8)),
        ],
        bbox=(0, 0, 10, 10),
        orientation_uncertain=True,
    )
    data = OriginalImage(
        images=[np.zeros((20, 20, 3), dtype=np.uint8)],
        class_object_image=[[obj]],
    )

    summary = _build_summary(data)

    assert summary.page_count == 1
    assert summary.object_count == 1
    assert summary.classified_object_count == 1
    assert summary.part_count == 2
    assert summary.uncertain_orientation_count == 1


def test_annotate_image_draws_object_bbox():
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    obj = ObjectImage(
        class_name="seeding",
        confidence=0.9,
        image=[],
        bbox=(5, 6, 20, 22),
    )

    annotated = _annotate_image(image, [obj])

    assert not np.array_equal(annotated, image)


def test_object_overlay_to_pil_draws_part_mask():
    crop = np.zeros((40, 40, 3), dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:25, 12:28] = 255
    part = AllClassImage(
        class_name="root",
        confidence=0.9,
        image=np.zeros((1, 1, 3), dtype=np.uint8),
        bbox=(12, 10, 28, 25),
        mask_bitmap=mask,
    )
    obj = ObjectImage(
        class_name="seeding",
        confidence=0.9,
        image=[crop],
        image_all_class=[part],
        bbox=(0, 0, 40, 40),
    )

    overlaid = _object_overlay_to_pil(obj)

    assert overlaid is not None
    assert not np.array_equal(np.array(overlaid), crop[:, :, ::-1])
