import os

import numpy as np
from PyQt5.QtWidgets import QApplication

from seeding.mask_refiner import refine_manual_mask_bitmap
from seeding.models import ObjectImage
from seeding.ui.main_window import ImageEditor


def _ensure_offscreen_qt() -> tuple[QApplication, bool]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    created = app is None
    if app is None:
        app = QApplication([])
    return app, created


def test_manual_part_mask_keeps_dominant_component_inside_bbox():
    app, created = _ensure_offscreen_qt()
    page = np.full((64, 64, 3), 245, dtype=np.uint8)

    for x in range(0, 64, 6):
        page[:, x : x + 1] = 170
    for y in range(0, 64, 6):
        page[y : y + 1, :] = 170

    page[10:54, 28:33] = 70
    seed_crop = page[8:56, 16:48].copy()

    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    window.image_storage.images = [page]
    window.image_storage.source_files = ["page1.png"]
    window.image_storage.file_path = "page1.png"
    window.image_storage.class_object_image = [
        [
            ObjectImage(
                class_name="seeding",
                confidence=0.9,
                image=[seed_crop],
                bbox=(16, 8, 48, 56),
            )
        ]
    ]
    window.app_state.selected_item = {
        "type": "seeding",
        "parent_index": 0,
        "index": 0,
    }

    created_ok = window._create_manual_object(0, "root", (24, 12, 38, 52))

    assert created_ok is True
    parts = window.image_storage.class_object_image[0][0].image_all_class
    assert parts is not None and len(parts) == 1

    part = parts[0]
    assert part.mask_bitmap is not None

    lx1, ly1, lx2, ly2 = part.bbox
    roi = part.mask_bitmap[ly1:ly2, lx1:lx2]
    cols = np.where(np.count_nonzero(roi, axis=0) > 0)[0]
    assert cols.size > 0
    assert cols.min() >= 3
    assert cols.max() <= 10

    window.close()
    if created:
        app.quit()


def test_refine_manual_mask_bitmap_suppresses_grid_lines():
    image = np.full((48, 32, 3), 245, dtype=np.uint8)

    for x in range(0, 32, 6):
        image[:, x : x + 1] = 170
    for y in range(0, 48, 6):
        image[y : y + 1, :] = 170

    image[4:44, 13:18] = 60

    coarse = np.zeros((48, 32), dtype=np.uint8)
    coarse[0:48, 8:22] = 255

    mask = refine_manual_mask_bitmap(image, coarse, (8, 0, 22, 48))

    assert mask is not None
    roi = mask[:, 8:22]
    cols = np.where(np.count_nonzero(roi, axis=0) > 0)[0]
    assert cols.size > 0
    assert cols.min() >= 4
    assert cols.max() <= 10
