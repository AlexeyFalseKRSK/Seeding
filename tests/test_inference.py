import numpy as np
import pytest

from seeding.inference import InferenceBox, infer_backend_kind, normalize_yolo_results


class _TensorStub:
    def __init__(self, values):
        self._values = values

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _BoxStub:
    def __init__(self, cls_id: int, conf: float, xyxy):
        self.cls = cls_id
        self.conf = conf
        self.xyxy = [_TensorStub(xyxy)]


class _MasksStub:
    def __init__(self, polygons):
        self.xy = polygons


class _ResultStub:
    def __init__(self, with_masks: bool = False):
        self.names = {0: "seedling", 1: "root"}
        self.boxes = [_BoxStub(1, 0.82, [10.0, 20.0, 40.0, 60.0])]
        self.masks = (
            _MasksStub([np.array([[10.0, 20.0], [40.0, 20.0], [25.0, 60.0]])])
            if with_masks
            else None
        )


def test_infer_backend_kind_uses_pt_only():
    with pytest.raises(ValueError, match="ONNX"):
        infer_backend_kind("models/detect.onnx")
    assert infer_backend_kind("models/detect.pt") == "torch"


def test_normalize_yolo_results_converts_boxes_to_ultralytics_like_adapter():
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
    box = InferenceBox(
        cls=2,
        conf=0.55,
        bbox_xyxy=(10.0, 20.0, 34.0, 44.0),
    )

    assert box.xyxy[0].cpu().numpy().tolist() == [10.0, 20.0, 34.0, 44.0]
    assert box.xywh[0].cpu().numpy().tolist() == [22.0, 32.0, 24.0, 24.0]


def test_normalize_yolo_results_preserves_mask_polygon():
    results = normalize_yolo_results([_ResultStub(with_masks=True)])

    box = results[0].boxes[0]
    assert box.mask_polygon is not None
    assert box.mask_polygon.shape == (3, 2)
    assert np.allclose(
        box.mask_polygon,
        np.array([[10.0, 20.0], [40.0, 20.0], [25.0, 60.0]], dtype=np.float32),
    )
