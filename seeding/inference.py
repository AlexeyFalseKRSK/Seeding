"""Inference backend for Ultralytics PyTorch models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def infer_backend_kind(model_path: str | Path) -> str:
    """Returns the only supported backend kind."""
    suffix = Path(model_path).suffix.strip().lower()
    if suffix == ".onnx":
        raise ValueError("ONNX models are no longer supported in this project.")
    return "torch"


class _TensorAdapter:
    """Small adapter that mimics the tensor API expected by services."""

    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = np.asarray(values, dtype=float)

    def cpu(self) -> "_TensorAdapter":
        return self

    def numpy(self) -> np.ndarray:
        return self._values


@dataclass(frozen=True)
class InferenceBox:
    cls: int
    conf: float
    bbox_xyxy: tuple[float, float, float, float]

    @property
    def xyxy(self) -> list[_TensorAdapter]:
        return [_TensorAdapter(self.bbox_xyxy)]

    @property
    def xywh(self) -> list[_TensorAdapter]:
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
    names: dict[int, str]
    boxes: list[InferenceBox]


class InferenceBackend(ABC):
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.backend_kind = infer_backend_kind(model_path)

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        *,
        conf_threshold: float | None = None,
    ) -> list[InferenceResult]:
        """Runs inference on an image."""


def _load_yolo_class():
    from ultralytics import YOLO

    return YOLO


def _normalize_names_map(names: Any) -> dict[int, str]:
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


def normalize_yolo_results(raw_results) -> list[InferenceResult]:
    if raw_results is None:
        return []

    normalized_results: list[InferenceResult] = []
    for result in raw_results:
        names = _normalize_names_map(getattr(result, "names", {}))
        boxes: list[InferenceBox] = []
        for box in getattr(result, "boxes", []) or []:
            coords = box.xyxy[0].cpu().numpy()
            boxes.append(
                InferenceBox(
                    cls=int(box.cls),
                    conf=float(box.conf),
                    bbox_xyxy=tuple(float(value) for value in coords[:4]),
                )
            )
        normalized_results.append(InferenceResult(names=names, boxes=boxes))
    return normalized_results


class TorchYoloBackend(InferenceBackend):
    def __init__(self, model_path: str | Path) -> None:
        super().__init__(model_path)
        yolo_class = _load_yolo_class()
        self._model = yolo_class(self.model_path)

    def predict(
        self,
        image: np.ndarray,
        *,
        conf_threshold: float | None = None,
    ) -> list[InferenceResult]:
        kwargs = {}
        if conf_threshold is not None:
            kwargs["conf"] = float(conf_threshold)
        return normalize_yolo_results(self._model(image, **kwargs))


def load_inference_backend(model_path: str | Path) -> InferenceBackend:
    """Loads a PyTorch-backed Ultralytics model."""
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
