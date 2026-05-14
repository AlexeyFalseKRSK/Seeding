import sys
from types import SimpleNamespace

import numpy as np
import pytest

from seeding.inference import (
    InferenceBackend,
    InferenceBox,
    InferenceResult,
    infer_backend_kind,
    normalize_yolo_results,
    resolve_yolo_device,
)


class _TensorStub:
    """Минимальная заглушка тензора для проверки адаптеров инференса."""

    def __init__(self, values):
        """Сохраняет переданные числовые значения."""
        self._values = values

    def cpu(self):
        """Возвращает сам объект для совместимости с ожидаемым API."""
        return self

    def numpy(self):
        """Возвращает сохранённые значения без дополнительной обработки."""
        return self._values


class _BoxStub:
    """Тестовая заглушка одного найденного объекта модели."""

    def __init__(self, cls_id: int, conf: float, xyxy):
        """Сохраняет класс, уверенность и координаты bbox."""
        self.cls = cls_id
        self.conf = conf
        self.xyxy = [_TensorStub(xyxy)]


class _MasksStub:
    """Тестовая заглушка контейнера с полигонами масок."""

    def __init__(self, polygons):
        """Сохраняет набор полигонов масок."""
        self.xy = polygons


class _ResultStub:
    """Минимальный результат модели с именами классов, bbox и опциональной маской."""

    def __init__(self, with_masks: bool = False):
        """Создаёт результат с одной рамкой и по желанию добавляет полигон маски."""
        self.names = {0: "seedling", 1: "root"}
        self.boxes = [_BoxStub(1, 0.82, [10.0, 20.0, 40.0, 60.0])]
        self.masks = (
            _MasksStub([np.array([[10.0, 20.0], [40.0, 20.0], [25.0, 60.0]])])
            if with_masks
            else None
        )


class _BackendStub(InferenceBackend):
    """Minimal backend that relies on the default predict_batch implementation."""

    def __init__(self):
        super().__init__("models/detect.pt")
        self.calls = 0

    def predict(self, image, *, conf_threshold=None):
        _ = image
        self.calls += 1
        return [
            InferenceResult(
                names={0: "root"},
                boxes=[
                    InferenceBox(
                        cls=0,
                        conf=float(conf_threshold or 0.5),
                        bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
                    )
                ],
            )
        ]


def test_infer_backend_kind_uses_pt_only():
    """Проверяет, что поддерживаются только `.pt`-модели, а ONNX запрещён."""
    with pytest.raises(ValueError, match="ONNX"):
        infer_backend_kind("models/detect.onnx")
    assert infer_backend_kind("models/detect.pt") == "torch"


def test_normalize_yolo_results_converts_boxes_to_ultralytics_like_adapter():
    """Проверяет нормализацию bbox и имён классов в совместимый внутренний формат."""
    results = normalize_yolo_results([_ResultStub()])

    assert len(results) == 1
    assert results[0].names == {0: "seedling", 1: "root"}
    assert len(results[0].boxes) == 1
    box = results[0].boxes[0]
    assert box.cls == 1
    assert abs(box.conf - 0.82) < 1e-9
    assert np.allclose(box.xyxy[0].cpu().numpy(), [10.0, 20.0, 40.0, 60.0])
    assert box.xywh[0].cpu().numpy().tolist() == [25.0, 40.0, 30.0, 40.0]


def test_inference_box_exposes_xyxy_and_xywh_adapters():
    """Проверяет корректную выдачу координат bbox в форматах `xyxy` и `xywh`."""
    box = InferenceBox(
        cls=2,
        conf=0.55,
        bbox_xyxy=(10.0, 20.0, 34.0, 44.0),
    )

    assert box.xyxy[0].cpu().numpy().tolist() == [10.0, 20.0, 34.0, 44.0]
    assert box.xywh[0].cpu().numpy().tolist() == [22.0, 32.0, 24.0, 24.0]


def test_normalize_yolo_results_preserves_mask_polygon():
    """Проверяет сохранение и нормализацию полигона маски из результата модели."""
    results = normalize_yolo_results([_ResultStub(with_masks=True)])

    box = results[0].boxes[0]
    assert box.mask_polygon is not None
    assert box.mask_polygon.shape == (3, 2)
    assert np.allclose(
        box.mask_polygon,
        np.array([[10.0, 20.0], [40.0, 20.0], [25.0, 60.0]], dtype=np.float32),
    )


def test_inference_backend_predict_batch_preserves_single_predict_shape():
    """Проверяет базовый batch-интерфейс без зависимости от Ultralytics."""
    backend = _BackendStub()
    images = [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros((5, 5, 3), dtype=np.uint8),
    ]

    batch = backend.predict_batch(images, conf_threshold=0.7)

    assert len(batch) == 2
    assert backend.calls == 2
    assert batch[0][0].names == {0: "root"}
    assert batch[1][0].boxes[0].conf == 0.7


def test_resolve_yolo_device_uses_cuda_when_available(monkeypatch):
    """Проверяет авто-выбор видеокарты для YOLO при доступной CUDA."""
    monkeypatch.delenv("YOLO_DEVICE", raising=False)
    monkeypatch.setattr("seeding.config.YOLO_DEVICE", "auto")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
            )
        ),
    )

    assert resolve_yolo_device() == 0


def test_resolve_yolo_device_falls_back_to_cpu_without_cuda(monkeypatch):
    """Проверяет CPU fallback, если CUDA-видеокарты нет."""
    monkeypatch.delenv("YOLO_DEVICE", raising=False)
    monkeypatch.setattr("seeding.config.YOLO_DEVICE", "auto")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: False,
                device_count=lambda: 0,
            )
        ),
    )

    assert resolve_yolo_device() == "cpu"


def test_resolve_yolo_device_respects_manual_config(monkeypatch):
    """Проверяет ручное переопределение устройства YOLO."""
    monkeypatch.delenv("YOLO_DEVICE", raising=False)
    monkeypatch.setattr("seeding.config.YOLO_DEVICE", "cpu")

    assert resolve_yolo_device() == "cpu"


def test_resolve_yolo_device_falls_back_when_manual_gpu_is_unavailable(monkeypatch):
    """Проверяет, что сохранённый GPU не роняет инференс без CUDA."""
    monkeypatch.setenv("YOLO_DEVICE", "0")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: False,
                device_count=lambda: 0,
            )
        ),
    )

    assert resolve_yolo_device() == "cpu"


def test_resolve_yolo_device_accepts_available_manual_gpu(monkeypatch):
    """Проверяет, что ручной GPU используется только когда он реально доступен."""
    monkeypatch.setenv("YOLO_DEVICE", "0")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
            )
        ),
    )

    assert resolve_yolo_device() == "0"
