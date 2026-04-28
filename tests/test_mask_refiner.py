import numpy as np

from seeding.mask_refiner import refine_mask_bitmap


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
