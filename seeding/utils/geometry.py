"""Утилиты обработки изображений и bounding-box.

Секции:
1. Detection helpers (NMS)
2. Geometry helpers (bbox rotate/clip)
3. Image transforms (rotation image + boxes)
"""

from __future__ import annotations

import logging
from typing import Iterable

import cv2
import numpy as np

from seeding.config import NMS_IOU_THRESHOLD

logger = logging.getLogger(__name__)


def simple_nms(boxes, scores, iou_threshold=None):
    """Выполняет Non-Maximum Suppression.

    Args:
        boxes: Список bbox в формате ``[x1, y1, x2, y2]``.
        scores: Список confidence-значений той же длины.
        iou_threshold: Порог IoU; если ``None``, берётся из конфигурации.

    Returns:
        list[int]: Индексы боксов, оставшихся после NMS.
    """
    if iou_threshold is None:
        iou_threshold = NMS_IOU_THRESHOLD
    if not boxes:
        logger.debug("simple_nms: empty boxes input")
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    logger.debug("simple_nms: %s boxes left after NMS", len(keep))
    return keep


def rotate_bbox(x1, y1, x2, y2, w, h, k):
    """Поворачивает bbox на ``k`` шагов по 90 градусов против часовой.

    Args:
        x1, y1, x2, y2: Координаты исходного bbox.
        w, h: Размеры исходного изображения.
        k: Количество четверть-оборотов (может быть отрицательным).

    Returns:
        tuple[int, int, int, int]: Координаты повернутого bbox.
    """
    k = k % 4
    coords = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    if k == 1:
        pts = [(y, w - 1 - x) for x, y in coords]
    elif k == 2:
        pts = [(w - 1 - x, h - 1 - y) for x, y in coords]
    elif k == 3:
        pts = [(h - 1 - y, x) for x, y in coords]
    else:
        pts = coords

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def rotate_polygon_points(
    points: np.ndarray,
    width: int,
    height: int,
    angle: float,
) -> np.ndarray | None:
    """Rotates polygon points with the same transform as image rotation."""
    if points is None or width <= 0 or height <= 0:
        return None

    polygon = np.asarray(points, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        return None

    quarter_turns = int(round(angle / 90.0))
    if np.isclose(angle, quarter_turns * 90.0):
        k_mod = quarter_turns % 4
        if k_mod == 1:
            transformed = np.column_stack((polygon[:, 1], width - 1 - polygon[:, 0]))
            target_width, target_height = height, width
        elif k_mod == 2:
            transformed = np.column_stack(
                (width - 1 - polygon[:, 0], height - 1 - polygon[:, 1])
            )
            target_width, target_height = width, height
        elif k_mod == 3:
            transformed = np.column_stack((height - 1 - polygon[:, 1], polygon[:, 0]))
            target_width, target_height = height, width
        else:
            transformed = polygon.copy()
            target_width, target_height = width, height
    else:
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        abs_cos = abs(matrix[0, 0])
        abs_sin = abs(matrix[0, 1])
        target_width = int(height * abs_sin + width * abs_cos)
        target_height = int(height * abs_cos + width * abs_sin)

        matrix[0, 2] += (target_width / 2.0) - center[0]
        matrix[1, 2] += (target_height / 2.0) - center[1]

        homogeneous = np.concatenate(
            (polygon, np.ones((polygon.shape[0], 1), dtype=np.float64)),
            axis=1,
        )
        transformed = homogeneous @ matrix.T

    transformed[:, 0] = np.clip(
        transformed[:, 0],
        0.0,
        max(float(target_width - 1), 0.0),
    )
    transformed[:, 1] = np.clip(
        transformed[:, 1],
        0.0,
        max(float(target_height - 1), 0.0),
    )
    return np.ascontiguousarray(transformed.astype(np.float32))


def clip_bbox_to_image(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Ограничивает bbox размерами изображения и нормализует углы.

    Args:
        bbox: Координаты ``(x1, y1, x2, y2)``.
        width: Ширина изображения.
        height: Высота изображения.

    Returns:
        tuple[int, int, int, int] | None:
            Валидный bbox внутри изображения либо ``None``,
            если после clipping область пустая.
    """
    if width <= 0 or height <= 0:
        return None

    x1, y1, x2, y2 = (int(v) for v in bbox)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    clipped = np.clip(
        np.array([x1, y1, x2, y2], dtype=np.int64),
        np.array([0, 0, 0, 0], dtype=np.int64),
        np.array([width, height, width, height], dtype=np.int64),
    )
    cx1, cy1, cx2, cy2 = (int(v) for v in clipped.tolist())
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2


def _transform_bbox_with_matrix(
    bbox: tuple[int, int, int, int],
    matrix: np.ndarray,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Трансформирует вершины bbox аффинной матрицей и делает clipping."""
    x1, y1, x2, y2 = bbox
    corners = np.array(
        [
            [x1, y1, 1.0],
            [x2, y1, 1.0],
            [x2, y2, 1.0],
            [x1, y2, 1.0],
        ],
        dtype=np.float64,
    )
    transformed = corners @ matrix.T
    tx1 = int(np.floor(np.min(transformed[:, 0])))
    ty1 = int(np.floor(np.min(transformed[:, 1])))
    tx2 = int(np.ceil(np.max(transformed[:, 0])))
    ty2 = int(np.ceil(np.max(transformed[:, 1])))
    return clip_bbox_to_image((tx1, ty1, tx2, ty2), width, height)


def rotate_image_and_boxes(
    image: np.ndarray,
    boxes: Iterable[tuple[int, int, int, int]],
    angle: float,
) -> tuple[np.ndarray, list[tuple[int, int, int, int] | None]]:
    """Поворачивает изображение и набор bbox единым преобразованием.

    Args:
        image: Исходный массив изображения.
        boxes: Итерируемый набор bbox в координатах исходного изображения.
        angle: Угол поворота в градусах (положительный — против часовой).

    Returns:
        tuple[np.ndarray, list[tuple[int, int, int, int] | None]]:
            Повернутое изображение и список bbox после трансформации.

    Raises:
        ValueError: Если ``image`` не является валидным ``numpy.ndarray``.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Требуется корректный numpy-массив изображения")

    height, width = image.shape[:2]
    quarter_turns = int(round(angle / 90.0))
    if np.isclose(angle, quarter_turns * 90.0):
        rotated_image = np.rot90(image, k=quarter_turns)
        k_mod = quarter_turns % 4
        if k_mod % 2:
            rotated_width, rotated_height = height, width
        else:
            rotated_width, rotated_height = width, height

        rotated_boxes = []
        for box in boxes:
            rx1, ry1, rx2, ry2 = rotate_bbox(
                box[0],
                box[1],
                box[2],
                box[3],
                width,
                height,
                k_mod,
            )
            rotated_boxes.append(
                clip_bbox_to_image(
                    (rx1, ry1, rx2, ry2),
                    rotated_width,
                    rotated_height,
                )
            )
        return rotated_image, rotated_boxes

    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    abs_cos = abs(matrix[0, 0])
    abs_sin = abs(matrix[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    matrix[0, 2] += (bound_w / 2.0) - center[0]
    matrix[1, 2] += (bound_h / 2.0) - center[1]

    rotated_image = cv2.warpAffine(image, matrix, (bound_w, bound_h))
    rotated_boxes = []
    for box in boxes:
        rotated_boxes.append(
            _transform_bbox_with_matrix(box, matrix, bound_w, bound_h)
        )
    return rotated_image, rotated_boxes
