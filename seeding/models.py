"""Доменные модели данных, используемые приложением Seeding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

import numpy as np
from PIL import Image

BBox = tuple[int, int, int, int]


@dataclass
class AllClassImage:
    """Описывает найденную часть растения, её bbox, изображение и маску."""

    class_name: str
    confidence: float
    image: np.ndarray | Image.Image
    bbox: BBox | None = None
    mask_polygon: np.ndarray | None = None


@dataclass
class ObjectImage:
    """Хранит данные о найденном сеянце и классифицированных частях внутри него."""

    class_name: str
    confidence: float
    image: list[np.ndarray | Image.Image] = field(default_factory=list)
    image_all_class: list[AllClassImage] | None = None
    bbox: BBox | None = None
    rotation_k: int = 0


@dataclass
class OriginalImage:
    """Содержит исходные страницы проекта и результаты их обработки."""

    file_path: str = ""
    source_files: list[str] = field(default_factory=list)
    images: list[np.ndarray | Image.Image] = field(default_factory=list)
    class_object_image: list[list[ObjectImage]] | None = None


class SelectionPayload(TypedDict, total=False):
    """Описывает выбранный в интерфейсе элемент и его индексы в структуре проекта."""

    type: Literal["original", "pdf", "seeding", "class"]
    index: int
    parent_index: int
    seeding_index: int
    class_index: int


@dataclass
class RotateSelectionResult:
    """Возвращает результат поворота страницы или кропа для последующей перерисовки."""

    target: Literal["page", "crop"]
    page_index: int
    image: np.ndarray
    crop_index: int | None = None


@dataclass
class AppState:
    """Хранит текущее состояние проекта, выбора пользователя и параметров просмотра."""

    image_storage: OriginalImage = field(default_factory=OriginalImage)
    active_image_index: int = 0
    selected_item: SelectionPayload | None = None
    zoom_factor: float = 1.0
    pixels_per_mm: float = 0.0
    last_report_path: str = ""


__all__ = [
    "AllClassImage",
    "AppState",
    "BBox",
    "ObjectImage",
    "OriginalImage",
    "RotateSelectionResult",
    "SelectionPayload",
]
