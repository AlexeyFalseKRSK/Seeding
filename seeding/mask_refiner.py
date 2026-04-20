"""Попиксельное уточнение сегментационных масок.

Стратегия: YOLO даёт грубый полигон/bitmap с фоновыми пикселями внутри
(бумага в клетку просвечивает между хвоинками/корнями). Мы применяем
порог Otsu по яркости внутри YOLO-маски — тёмные пиксели = растение,
светлые = бумага. Результат — точная попиксельная маска.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Пиксели ярче этого значения точно бумага (белый/чуть серый)
_PAPER_THRESHOLD = 210
# Ниже этого считаем точно растением даже при скошенной гистограмме
_PLANT_THRESHOLD = 180


def polygon_to_bitmap(
    polygon: np.ndarray | None,
    h: int,
    w: int,
) -> np.ndarray | None:
    """Растеризует полигон в бинарную маску 0/255 размером (h, w)."""
    if polygon is None or polygon.ndim != 2 or polygon.shape[1] != 2:
        return None
    if polygon.shape[0] < 3:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = polygon.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def bitmap_to_polygon(binary_mask: np.ndarray) -> np.ndarray | None:
    """Извлекает наибольший контур бинарной маски как полигон Nx2 float32."""
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
    """Вычисляет порог яркости: paper > thresh, plant <= thresh.

    Стратегия: Otsu работает только при сбалансированной гистограмме.
    При разреженном растении на бумаге Otsu даёт слишком низкий порог.
    Поэтому берём Otsu только если доля тёмных пикселей достаточная,
    иначе используем фиксированный _PAPER_THRESHOLD.
    """
    inside = gray[coarse_mask > 0]
    if inside.size < 20:
        return _PAPER_THRESHOLD

    # Доля явно тёмных пикселей (точно не бумага)
    dark_ratio = float(np.mean(inside < _PLANT_THRESHOLD))

    # Otsu имеет смысл только при относительно сбалансированной гистограмме:
    # от ~15% до ~85% тёмных. Иначе он выбирает случайный хвост.
    if 0.15 <= dark_ratio <= 0.85:
        column = inside.reshape(-1, 1).astype(np.uint8)
        otsu_val, _ = cv2.threshold(
            column, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # Всё равно не даём порогу уйти выше paper-уровня
        return min(int(otsu_val), _PAPER_THRESHOLD)

    return _PAPER_THRESHOLD


def refine_mask_bitmap(
    image: np.ndarray,
    coarse_mask: np.ndarray,
) -> np.ndarray | None:
    """Уточняет грубую маску по яркости — оставляет только тёмные пиксели.

    Args:
        image: BGR uint8 изображение (crop сеянца).
        coarse_mask: Грубая бинарная маска 0/255 того же HxW — обычно
                     растеризованный YOLO-полигон или bitmap от YOLO.

    Returns:
        Уточнённая бинарная маска uint8 (0/255) того же HxW, либо None.
    """
    if image is None or coarse_mask is None:
        return None
    if image.ndim < 2 or coarse_mask.ndim != 2:
        return None
    h, w = image.shape[:2]
    if coarse_mask.shape[:2] != (h, w):
        return None
    if not np.any(coarse_mask > 0):
        return None

    # Готовим grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.uint8) if image.dtype != np.uint8 else image

    # Подбираем порог: Otsu при сбалансированной гистограмме, иначе paper-уровень
    threshold = _compute_threshold(gray, coarse_mask)

    # Плант = тёмные пиксели И внутри грубой маски (<= чтобы граничные пиксели
    # попали в объект)
    plant = ((gray <= threshold) & (coarse_mask > 0)).astype(np.uint8) * 255

    # Closing соединяет близкие фрагменты хвоинок/корней (НЕ erode — иначе
    # тонкие 1-2px структуры съедаются)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    plant = cv2.morphologyEx(plant, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # Удаляем мелкие изолированные компоненты (пыль/артефакты сканирования),
    # но сохраняем тонкие протяжённые структуры
    plant = _filter_small_components(plant, min_area=8)

    # Если после чистки почти ничего не осталось — возвращаем исходную маску
    if np.count_nonzero(plant) < 10:
        logger.debug("refine_mask_bitmap: после Otsu+morph маска пустая, fallback")
        return coarse_mask.copy()

    return plant


def _filter_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Обнуляет компоненты связности с площадью меньше min_area."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return binary
    keep = np.zeros(num_labels, dtype=bool)
    keep[0] = False  # фон
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[i] = True
    mask = keep[labels]
    return (mask.astype(np.uint8) * 255)


def rotate_bitmap(
    bitmap: np.ndarray | None,
    angle: float,
) -> np.ndarray | None:
    """Поворачивает bitmap-маску на угол (для ортогональных углов — np.rot90)."""
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
    "bitmap_to_contours",
    "polygon_to_bitmap",
    "refine_mask_bitmap",
    "rotate_bitmap",
]
