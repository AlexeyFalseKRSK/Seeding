import numpy as np
import pytest
from seeding.mask_refiner import bitmap_to_contours


def _make_bitmap_with_hole():
    """Кольцо 20x20: внешний квадрат 20x20, внутренний белый квадрат 8x8 в центре."""
    bm = np.zeros((20, 20), dtype=np.uint8)
    bm[2:18, 2:18] = 255   # внешний контур
    bm[6:14, 6:14] = 0     # дыра
    return bm


def test_bitmap_to_contours_returns_list():
    bm = np.zeros((10, 10), dtype=np.uint8)
    bm[1:9, 1:9] = 255
    result = bitmap_to_contours(bm)
    assert isinstance(result, list)
    assert len(result) >= 1
    assert all(isinstance(c, np.ndarray) for c in result)
    assert all(c.ndim == 2 and c.shape[1] == 2 for c in result)


def test_bitmap_to_contours_detects_hole():
    bm = _make_bitmap_with_hole()
    result = bitmap_to_contours(bm)
    # Должно быть >= 2 контуров: внешний + дыра
    assert len(result) >= 2


def test_bitmap_to_contours_empty_returns_empty():
    assert bitmap_to_contours(np.zeros((10, 10), dtype=np.uint8)) == []
    assert bitmap_to_contours(None) == []
