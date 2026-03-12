"""Инференс и адаптеры результатов для моделей Ultralytics на PyTorch."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def infer_backend_kind(model_path: str | Path) -> str:
    """Определяет поддерживаемый тип backend по пути к файлу модели."""
    suffix = Path(model_path).suffix.strip().lower()
    if suffix == ".onnx":
        raise ValueError("ONNX models are no longer supported in this project.")
    return "torch"


class _TensorAdapter:
    """Небольшой адаптер, имитирующий API тензора для сервисной логики."""

    def __init__(self, values: tuple[float, ...]) -> None:
        """Сохраняет набор числовых значений в виде массива NumPy."""
        self._values = np.asarray(values, dtype=float)

    def cpu(self) -> "_TensorAdapter":
        """Возвращает сам адаптер для совместимости с цепочкой вызовов Ultralytics."""
        return self

    def numpy(self) -> np.ndarray:
        """Возвращает сохранённые значения как массив NumPy."""
        return self._values


@dataclass(frozen=True)
class InferenceBox:
    """Представляет один результат детекции или сегментации с bbox и маской."""

    cls: int
    conf: float
    bbox_xyxy: tuple[float, float, float, float]
    mask_polygon: np.ndarray | None = None

    @property
    def xyxy(self) -> list[_TensorAdapter]:
        """Возвращает bbox в формате угловых координат через адаптер тензора."""
        return [_TensorAdapter(self.bbox_xyxy)]

    @property
    def xywh(self) -> list[_TensorAdapter]:
        """Возвращает bbox в формате центр-ширина-высота через адаптер тензора."""
        x1, y1, x2, y2 = self.bbox_xyxy
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        return [
            _TensorAdapter(
                (
                    x1 + width / 2.0,
                    y1 + height / 2.0,
                    width,
                    height,
                )
            )
        ]


@dataclass(frozen=True)
class InferenceResult:
    """Хранит карту имён классов и список найденных объектов для одного результата."""

    names: dict[int, str]
    boxes: list[InferenceBox]


class InferenceBackend(ABC):
    """Базовый интерфейс backend для запуска модели на изображениях."""

    def __init__(self, model_path: str | Path) -> None:
        """Сохраняет путь к модели и определяет тип используемого backend."""
        self.model_path = str(model_path)
        self.backend_kind = infer_backend_kind(model_path)

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        *,
        conf_threshold: float | None = None,
    ) -> list[InferenceResult]:
        """Выполняет инференс на изображении и возвращает нормализованные результаты."""


def _load_yolo_class():
    """Лениво импортирует класс `YOLO`, чтобы не загружать зависимость заранее."""
    from ultralytics import YOLO

    return YOLO


def _normalize_names_map(names: Any) -> dict[int, str]:
    """Приводит описание имён классов к словарю вида `int -> str`."""
    if isinstance(names, dict):
        normalized: dict[int, str] = {}
        for key, value in names.items():
            try:
                normalized[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return normalized
    if isinstance(names, list):
        return {idx: str(value) for idx, value in enumerate(names)}
    return {}


def _normalize_mask_polygon(polygon: Any) -> np.ndarray | None:
    """Проверяет и нормализует полигон маски к массиву `float32`."""
    if polygon is None:
        return None
    normalized = np.asarray(polygon, dtype=np.float32)
    if normalized.ndim != 2 or normalized.shape[1] != 2 or normalized.shape[0] < 3:
        return None
    return np.ascontiguousarray(normalized)


def _extract_mask_polygons(result: Any) -> list[np.ndarray | None]:
    """Извлекает полигоны масок из результата модели, если они присутствуют."""
    masks = getattr(result, "masks", None)
    if masks is None:
        return []
    polygons = getattr(masks, "xy", None)
    if polygons is None:
        return []
    return [_normalize_mask_polygon(polygon) for polygon in polygons]


def normalize_yolo_results(raw_results) -> list[InferenceResult]:
    """Преобразует сырые результаты Ultralytics в единый внутренний формат."""
    if raw_results is None:
        return []

    normalized_results: list[InferenceResult] = []
    for result in raw_results:
        names = _normalize_names_map(getattr(result, "names", {}))
        mask_polygons = _extract_mask_polygons(result)
        boxes: list[InferenceBox] = []
        for index, box in enumerate(getattr(result, "boxes", []) or []):
            coords = box.xyxy[0].cpu().numpy()
            boxes.append(
                InferenceBox(
                    cls=int(box.cls),
                    conf=float(box.conf),
                    bbox_xyxy=tuple(float(value) for value in coords[:4]),
                    mask_polygon=(
                        mask_polygons[index] if index < len(mask_polygons) else None
                    ),
                )
            )
        normalized_results.append(InferenceResult(names=names, boxes=boxes))
    return normalized_results


class TorchYoloBackend(InferenceBackend):
    """Backend для запуска моделей Ultralytics YOLO на PyTorch."""

    def __init__(self, model_path: str | Path) -> None:
        """Загружает модель YOLO по указанному пути к весам."""
        super().__init__(model_path)
        yolo_class = _load_yolo_class()
        self._model = yolo_class(self.model_path)

    def predict(
        self,
        image: np.ndarray,
        *,
        conf_threshold: float | None = None,
    ) -> list[InferenceResult]:
        """Запускает модель на изображении и нормализует полученные результаты."""
        kwargs = {}
        if conf_threshold is not None:
            kwargs["conf"] = float(conf_threshold)
        return normalize_yolo_results(self._model(image, **kwargs))


def load_inference_backend(model_path: str | Path) -> InferenceBackend:
    """Создаёт backend для работы с моделью Ultralytics на PyTorch."""
    return TorchYoloBackend(model_path)


__all__ = [
    "InferenceBackend",
    "InferenceBox",
    "InferenceResult",
    "TorchYoloBackend",
    "infer_backend_kind",
    "load_inference_backend",
    "normalize_yolo_results",
]
