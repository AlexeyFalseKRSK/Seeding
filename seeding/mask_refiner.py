"""Pixel-level refinement for segmentation masks."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Pixels brighter than this are treated as paper by the brightness fallback.
_PAPER_THRESHOLD = 210
# Pixels below this are confidently plant-like even on skewed histograms.
_PLANT_THRESHOLD = 180
_GREEN_HUE_RANGE = (35, 95)
_ORANGE_GRID_HUE_RANGE = (5, 32)
_LAB_FOREGROUND_DISTANCE = 22.0


@dataclass(frozen=True)
class RefineMaskContext:
    """Precomputed color spaces for repeated mask refinement on one crop."""

    image: np.ndarray
    gray: np.ndarray
    hsv: np.ndarray | None
    lab: np.ndarray | None


def build_refine_context(image: np.ndarray) -> RefineMaskContext | None:
    """Build reusable color-space data for mask refinement."""
    if image is None or image.ndim < 2:
        return None
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    else:
        gray = image.astype(np.uint8) if image.dtype != np.uint8 else image
        hsv = None
        lab = None
    return RefineMaskContext(image=image, gray=gray, hsv=hsv, lab=lab)


def polygon_to_bitmap(
    polygon: np.ndarray | None,
    h: int,
    w: int,
) -> np.ndarray | None:
    """Rasterize an Nx2 polygon into a 0/255 bitmap with shape (h, w)."""
    if polygon is None or polygon.ndim != 2 or polygon.shape[1] != 2:
        return None
    if polygon.shape[0] < 3:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = polygon.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def bitmap_to_polygon(binary_mask: np.ndarray) -> np.ndarray | None:
    """Extract the largest external contour from a binary mask as Nx2 float32."""
    if binary_mask is None or binary_mask.size == 0:
        return None
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return None
    return np.ascontiguousarray(largest.reshape(-1, 2).astype(np.float32))


def bitmap_to_contours(
    binary_mask: np.ndarray | None,
) -> list[np.ndarray]:
    """Возвращает все контуры маски — внешние + дыры — через RETR_CCOMP.

    Каждый контур — np.ndarray Nx2 float32 в координатах маски.
    Используется для QPainterPath с OddEvenFill: дыры прозрачны.
    """
    if binary_mask is None or binary_mask.size == 0:
        return []
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    result = []
    for contour in (contours or []):
        if len(contour) >= 3:
            result.append(
                np.ascontiguousarray(contour.reshape(-1, 2).astype(np.float32))
            )
    return result


def _compute_threshold(
    gray: np.ndarray,
    coarse_mask: np.ndarray,
) -> int:
    """Compute a brightness threshold where paper is above and plant is below."""
    inside = gray[coarse_mask > 0]
    if inside.size < 20:
        return _PAPER_THRESHOLD

    dark_ratio = float(np.mean(inside < _PLANT_THRESHOLD))
    if 0.15 <= dark_ratio <= 0.85:
        column = inside.reshape(-1, 1).astype(np.uint8)
        otsu_val, _ = cv2.threshold(
            column, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return min(int(otsu_val), _PAPER_THRESHOLD)

    return _PAPER_THRESHOLD


def _estimate_paper_lab(
    context: RefineMaskContext,
    coarse_mask: np.ndarray,
) -> np.ndarray | None:
    """Estimate the graph-paper color inside the coarse mask in LAB space."""
    if context.hsv is None or context.lab is None:
        return None

    inside = coarse_mask > 0
    paper_like = inside & (context.hsv[:, :, 2] >= 170) & (context.hsv[:, :, 1] <= 80)
    if np.count_nonzero(paper_like) < 20:
        paper_like = inside
    if np.count_nonzero(paper_like) < 20:
        return None
    return np.median(context.lab[paper_like], axis=0)


def _color_foreground_candidates(
    context: RefineMaskContext,
    coarse_mask: np.ndarray,
    threshold: int,
) -> np.ndarray:
    """Build color-aware plant candidates inside the coarse YOLO mask.

    The source photos are scanned on orange graph paper. A grayscale threshold
    alone tends to include grid lines and miss lighter green needles, so the
    final mask combines darkness, green hue, and distance from the paper color.
    """
    gray = context.gray
    inside = coarse_mask > 0
    if context.hsv is None or context.lab is None:
        return (gray <= threshold) & inside

    hue = context.hsv[:, :, 0]
    saturation = context.hsv[:, :, 1]
    value = context.hsv[:, :, 2]

    green = (
        (hue >= _GREEN_HUE_RANGE[0])
        & (hue <= _GREEN_HUE_RANGE[1])
        & (saturation >= 24)
        & (value <= 240)
    )

    paper_lab = _estimate_paper_lab(context, coarse_mask)
    if paper_lab is None:
        color_distance = np.zeros(gray.shape, dtype=np.float32)
    else:
        color_distance = np.linalg.norm(
            context.lab - paper_lab.reshape(1, 1, 3), axis=2
        )

    color_foreground = (
        (color_distance >= _LAB_FOREGROUND_DISTANCE)
        & (value <= 245)
        & (gray <= 235)
    )

    orange_grid = (
        (hue >= _ORANGE_GRID_HUE_RANGE[0])
        & (hue <= _ORANGE_GRID_HUE_RANGE[1])
        & (saturation >= 25)
        & (value >= 125)
        & (gray >= 135)
    )
    neutral_grid_or_pencil = (saturation <= 20) & (gray >= 120)
    bright_paper = (value >= 238) & (saturation <= 35)

    dark_plant = gray <= threshold
    candidates = (dark_plant | green | color_foreground) & inside
    candidates &= ~bright_paper
    candidates &= ~(orange_grid & ~green)
    candidates &= ~(neutral_grid_or_pencil & ~green)
    return candidates


def refine_mask_bitmap(
    image: np.ndarray,
    coarse_mask: np.ndarray,
    context: RefineMaskContext | None = None,
) -> np.ndarray | None:
    """Refine a coarse mask by keeping plant pixels and suppressing paper."""
    if image is None or coarse_mask is None:
        return None
    if image.ndim < 2 or coarse_mask.ndim != 2:
        return None
    h, w = image.shape[:2]
    if coarse_mask.shape[:2] != (h, w):
        return None
    if not np.any(coarse_mask > 0):
        return None

    if context is None:
        context = build_refine_context(image)
    if context is None:
        return None
    if context.gray.shape[:2] != (h, w):
        return None

    ys, xs = np.nonzero(coarse_mask > 0)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    local_context = RefineMaskContext(
        image=context.image[y1:y2, x1:x2],
        gray=context.gray[y1:y2, x1:x2],
        hsv=context.hsv[y1:y2, x1:x2] if context.hsv is not None else None,
        lab=context.lab[y1:y2, x1:x2] if context.lab is not None else None,
    )
    local_coarse = coarse_mask[y1:y2, x1:x2]

    threshold = _compute_threshold(local_context.gray, local_coarse)
    plant_local = (
        _color_foreground_candidates(local_context, local_coarse, threshold).astype(
            np.uint8
        )
        * 255
    )

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    plant_local = cv2.morphologyEx(
        plant_local, cv2.MORPH_CLOSE, close_kernel, iterations=1
    )
    plant_local = _filter_small_components(plant_local, min_area=8)

    plant = np.zeros_like(coarse_mask)
    plant[y1:y2, x1:x2] = plant_local

    if np.count_nonzero(plant) < 10:
        logger.debug("refine_mask_bitmap: refined mask is empty, using coarse mask")
        return coarse_mask.copy()

    return plant


def _filter_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return binary
    keep = np.zeros(num_labels, dtype=bool)
    keep[0] = False
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[i] = True
    mask = keep[labels]
    return (mask.astype(np.uint8) * 255)


def rotate_bitmap(
    bitmap: np.ndarray | None,
    angle: float,
) -> np.ndarray | None:
    """Rotate a bitmap mask by angle degrees."""
    if bitmap is None:
        return None

    quarter_turns = int(round(angle / 90.0))
    if np.isclose(angle, quarter_turns * 90.0):
        return np.ascontiguousarray(np.rot90(bitmap, k=quarter_turns))

    h, w = bitmap.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    abs_cos = abs(matrix[0, 0])
    abs_sin = abs(matrix[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    rotated = cv2.warpAffine(
        bitmap, matrix, (new_w, new_h), flags=cv2.INTER_NEAREST
    )
    return np.ascontiguousarray(rotated)


__all__ = [
    "bitmap_to_polygon",
    "build_refine_context",
    "bitmap_to_contours",
    "polygon_to_bitmap",
    "RefineMaskContext",
    "refine_mask_bitmap",
    "rotate_bitmap",
]
