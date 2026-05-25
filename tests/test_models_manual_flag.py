# tests/test_models_manual_flag.py
from seeding.models import AllClassImage, ObjectImage
import numpy as np


def test_object_image_manual_defaults_to_false():
    obj = ObjectImage(class_name="seeding", confidence=0.9)
    assert obj.manual is False


def test_all_class_image_manual_defaults_to_false():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    part = AllClassImage(class_name="root", confidence=0.8, image=img)
    assert part.manual is False


def test_object_image_manual_can_be_set():
    obj = ObjectImage(class_name="seeding", confidence=1.0, manual=True)
    assert obj.manual is True


def test_all_class_image_manual_can_be_set():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    part = AllClassImage(class_name="stem", confidence=1.0, image=img, manual=True)
    assert part.manual is True
