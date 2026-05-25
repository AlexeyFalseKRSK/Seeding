import numpy as np

from seeding.mask_refiner import bitmap_to_contours, build_refine_context, refine_mask_bitmap


def test_refine_mask_bitmap_suppresses_grid_lines():
    image = np.full((48, 32, 3), 245, dtype=np.uint8)

    for x in range(0, 32, 6):
        image[:, x : x + 1] = 170
    for y in range(0, 48, 6):
        image[y : y + 1, :] = 170

    image[4:44, 13:18] = 60

    coarse = np.zeros((48, 32), dtype=np.uint8)
    coarse[0:48, 8:22] = 255

    mask = refine_mask_bitmap(image, coarse)

    assert mask is not None
    roi = mask[:, 8:22]
    cols = np.where(np.count_nonzero(roi, axis=0) > 0)[0]
    assert cols.size > 0
    assert cols.min() >= 4
    assert cols.max() <= 10


def test_refine_mask_bitmap_suppresses_orange_graph_paper():
    image = np.full((64, 64, 3), 245, dtype=np.uint8)
    image[:, 0::8] = (80, 175, 225)
    image[0::8, :] = (80, 175, 225)
    image[28:32, 10:54] = (35, 45, 65)

    coarse = np.full((64, 64), 255, dtype=np.uint8)

    mask = refine_mask_bitmap(image, coarse)

    assert mask is not None
    assert np.count_nonzero(mask[28:32, 10:54]) > 120
    assert np.count_nonzero(mask[:, 0::8]) < 80
    assert np.count_nonzero(mask[0::8, :]) < 80


def test_refine_mask_bitmap_preserves_green_needles_and_dark_roots():
    image = np.full((80, 80, 3), 242, dtype=np.uint8)
    image[10:28, 18:22] = (48, 95, 52)
    image[14:30, 24:28] = (58, 115, 62)
    image[36:40, 16:68] = (28, 32, 45)
    image[42:46, 50:72] = (24, 26, 38)

    coarse = np.zeros((80, 80), dtype=np.uint8)
    coarse[5:55, 10:75] = 255

    mask = refine_mask_bitmap(image, coarse)

    assert mask is not None
    assert np.count_nonzero(mask[10:30, 18:28]) > 100
    assert np.count_nonzero(mask[36:46, 16:72]) > 260
    assert np.count_nonzero(mask[:5, :]) == 0


def test_refine_mask_bitmap_context_matches_uncached_result():
    image = np.full((80, 80, 3), 242, dtype=np.uint8)
    image[:, 0::8] = (80, 175, 225)
    image[0::8, :] = (80, 175, 225)
    image[20:24, 18:62] = (28, 32, 45)
    image[30:48, 35:39] = (48, 95, 52)

    coarse = np.zeros((80, 80), dtype=np.uint8)
    coarse[12:56, 12:68] = 255

    uncached = refine_mask_bitmap(image, coarse)
    cached = refine_mask_bitmap(image, coarse, build_refine_context(image))

    assert uncached is not None
    assert cached is not None
    assert np.array_equal(cached, uncached)


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
