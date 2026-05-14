import cv2
import numpy as np
import pytest

from seeding.image_loader import (
    ImageLoadError,
    load_pages_from_path,
    load_source_file_images,
    restore_session_source_images,
)
from seeding.models import AllClassImage, AppState, ObjectImage, OriginalImage


def test_load_source_file_images_reads_image(tmp_path):
    image = np.full((8, 9, 3), 77, dtype=np.uint8)
    path = str(tmp_path / "scan.png")
    assert cv2.imwrite(path, image)

    loaded = load_source_file_images(path)

    assert len(loaded) == 1
    assert loaded[0].shape == (8, 9, 3)


def test_load_source_file_images_raises_for_bad_image(tmp_path):
    path = str(tmp_path / "bad.png")

    with pytest.raises(ImageLoadError):
        load_source_file_images(path)


def test_load_pages_from_path_wraps_image_as_page(tmp_path):
    image = np.full((5, 6, 3), 33, dtype=np.uint8)
    path = str(tmp_path / "page.jpg")
    assert cv2.imwrite(path, image)

    pages = load_pages_from_path(path)

    assert len(pages) == 1
    assert pages[0].source_path == path
    assert pages[0].image.shape == (5, 6, 3)


def test_restore_session_source_images_rebuilds_crops(tmp_path):
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    image[10:30, 12:32] = (10, 120, 220)
    source_path = str(tmp_path / "scan.png")
    assert cv2.imwrite(source_path, image)
    part = AllClassImage(
        class_name="root",
        confidence=0.8,
        image=np.zeros((1, 1, 3), dtype=np.uint8),
        bbox=(2, 3, 10, 12),
    )
    obj = ObjectImage(
        class_name="seeding",
        confidence=0.9,
        image=[],
        image_all_class=[part],
        bbox=(12, 10, 32, 30),
    )
    state = AppState(
        image_storage=OriginalImage(
            file_path=source_path,
            source_files=[source_path],
            images=[],
            class_object_image=[[obj]],
        )
    )

    missing = restore_session_source_images(state)

    assert missing is None
    assert len(state.image_storage.images) == 1
    assert obj.image[0].shape == (20, 20, 3)
    assert part.image.shape == (9, 8, 3)
