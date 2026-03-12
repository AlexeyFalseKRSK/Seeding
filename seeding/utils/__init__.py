"""Пакет утилит Seeding.

Содержит:
- геометрию и преобразования изображений/боксов;
- кроссплатформенные утилиты путей.
"""

from .geometry import (
    clip_bbox_to_image,
    rotate_bbox,
    rotate_image_and_boxes,
    rotate_polygon_points,
    simple_nms,
)
from .paths import ensure_dir, resolve_weights_path

__all__ = [
    "simple_nms",
    "rotate_bbox",
    "clip_bbox_to_image",
    "rotate_image_and_boxes",
    "rotate_polygon_points",
    "resolve_weights_path",
    "ensure_dir",
]
