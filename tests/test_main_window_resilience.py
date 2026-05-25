import os
import time

import cv2
import numpy as np
import pytest
from PyQt5.QtCore import QPointF, QRectF, QSettings, Qt
from PyQt5.QtWidgets import QApplication, QGraphicsItem, QSplitter

from seeding.auth import AuthUser
from seeding.config import QSETTINGS_APP, QSETTINGS_ORG
from seeding.inference import InferenceBox, InferenceResult
from seeding.models import AllClassImage, AppState, ObjectImage, OriginalImage
from seeding.ui.main_window import ImageEditor


def _ensure_offscreen_qt() -> tuple[QApplication, bool]:
    """Создаёт `QApplication` в offscreen-режиме для GUI-тестов главного окна."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    created = app is None
    if app is None:
        app = QApplication([])
    return app, created


def _isolate_qsettings(tmp_path) -> None:
    """Изолирует `QSettings` во временном каталоге теста."""
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path))
    settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
    settings.clear()
    settings.sync()


def _wait_for_analysis(window: ImageEditor, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while window._analysis_is_running() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()
    assert not window._analysis_is_running()


class _StaticBackend:
    """Заглушка backend, возвращающая заранее подготовленные результаты."""

    def __init__(self, results):
        """Сохраняет результаты, которые должен вернуть тестовый backend."""
        self._results = results

    def predict(self, image, *, conf_threshold=None):
        """Возвращает сохранённые результаты независимо от входного изображения."""
        _ = (image, conf_threshold)
        return self._results

    def predict_batch(self, images, *, conf_threshold=None):
        """Возвращает сохранённые результаты для каждого входного изображения."""
        _ = conf_threshold
        return [self.predict(image) for image in images]


class _SequenceBackend:
    """Backend stub that returns prepared results in batch call order."""

    def __init__(self, batches):
        self._batches = list(batches)
        self.batch_sizes = []

    def predict(self, image, *, conf_threshold=None):
        _ = (image, conf_threshold)
        return self.predict_batch([image])[0]

    def predict_batch(self, images, *, conf_threshold=None):
        _ = conf_threshold
        self.batch_sizes.append(len(images))
        result = self._batches[: len(images)]
        self._batches = self._batches[len(images) :]
        return result


def test_append_page_keeps_storage_lists_in_sync():
    """Проверяет синхронное обновление списков проекта при добавлении страниц."""
    app, created = _ensure_offscreen_qt()
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    assert window.canvas_stack.currentWidget() is window.empty_state

    window._append_page(np.zeros((32, 32, 3), dtype=np.uint8), "page1.png")
    window._append_page(np.zeros((24, 24, 3), dtype=np.uint8), "page2.png")
    window._refresh_tree()
    window.display_image_with_boxes(0)

    assert len(window.image_storage.images) == 2
    assert len(window.image_storage.source_files) == 2
    assert len(window.image_storage.class_object_image or []) == 2
    assert window.tree_widget.topLevelItemCount() == 2
    assert window.canvas_stack.currentWidget() is window.graphics_view

    window.close()
    if created:
        app.quit()


def test_rotate_image_uses_current_display_after_stale_crop_selection():
    """Проверяет, что поворот применяется к текущей странице, а не к устаревшему кропу."""
    app, created = _ensure_offscreen_qt()
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    assert window.windowFlags() & Qt.FramelessWindowHint
    assert window.window_drag_handle.text() == "Seeding"
    assert window.menuBar().cornerWidget(Qt.TopRightCorner) is window.window_controls_widget
    window._append_page(np.zeros((30, 10, 3), dtype=np.uint8), "first.pdf")
    window._append_page(np.zeros((12, 24, 3), dtype=np.uint8), "second.pdf")
    window.image_storage.class_object_image[0] = [
        ObjectImage(
            class_name="seeding",
            confidence=0.9,
            image=[np.zeros((8, 8, 3), dtype=np.uint8)],
            bbox=(1, 1, 9, 9),
        )
    ]
    window.image_storage.class_object_image[1] = []
    window.app_state.selected_item = {
        "type": "seeding",
        "parent_index": 0,
        "index": 0,
    }

    window.display_image_with_boxes(1)
    window.rotate_image()

    assert window.app_state.selected_item == {"type": "pdf", "index": 1}
    assert window.image_storage.images[1].shape == (24, 12, 3)

    window.close()
    if created:
        app.quit()


def test_frameless_window_supports_drag_and_edge_detection():
    """Проверяет наличие drag-зоны и определение краёв для ресайза."""

    app, created = _ensure_offscreen_qt()
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    window.resize(900, 600)

    assert window._is_window_drag_source(window.window_drag_handle, None)
    edges = window._resize_edges_for_pos(window.rect().topLeft() + QPointF(1, 1).toPoint())
    assert edges & Qt.LeftEdge
    assert edges & Qt.TopEdge

    window.close()
    if created:
        app.quit()


def test_logout_action_emits_logout_signal():
    """Проверяет, что действие выхода переводит приложение в режим повторного входа."""

    app, created = _ensure_offscreen_qt()
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    emitted = []
    window.logout_requested.connect(lambda: emitted.append(True))

    window._request_logout()

    assert emitted == [True]

    window.close()
    if created:
        app.quit()


def test_clear_project_preserves_authenticated_user():
    app, created = _ensure_offscreen_qt()
    user = AuthUser(id=7, login="tester")
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt", current_user=user)

    window.clear_project()

    assert window.app_state.current_user_id == user.id

    window.close()
    if created:
        app.quit()


def test_inference_settings_loaded_from_qsettings(tmp_path):
    app, created = _ensure_offscreen_qt()
    _isolate_qsettings(tmp_path)
    settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
    settings.setValue("analysis/yolo_device", "cpu")
    settings.setValue("analysis/yolo_batch_size", 3)
    settings.setValue("app/auto_save_enabled", False)
    settings.setValue("report/output_dir", str(tmp_path))
    settings.sync()

    previous_device = os.environ.get("YOLO_DEVICE")
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")

    assert window._analysis_device == "cpu"
    assert window._analysis_batch_size == 3
    assert window._auto_save_enabled is False
    assert window._report_output_dir == str(tmp_path)
    assert os.environ["YOLO_DEVICE"] == "cpu"

    window.close()
    if previous_device is None:
        os.environ.pop("YOLO_DEVICE", None)
    else:
        os.environ["YOLO_DEVICE"] = previous_device
    if created:
        app.quit()


def test_auto_save_respects_settings_toggle(monkeypatch):
    app, created = _ensure_offscreen_qt()
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    called = False

    def fake_auto_save(*args, **kwargs):
        nonlocal called
        called = True
        return True

    monkeypatch.setattr("seeding.ui.main_window.auto_save_current_session", fake_auto_save)

    window._auto_save_enabled = False
    window._auto_save_session()
    assert called is False

    window._auto_save_enabled = True
    window._auto_save_session()
    assert called is True

    window.close()
    if created:
        app.quit()


def test_default_report_dir_prefers_settings_directory(tmp_path):
    app, created = _ensure_offscreen_qt()
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    window._report_output_dir = str(report_dir)

    assert window._default_report_dir() == str(report_dir)

    window.close()
    if created:
        app.quit()


def test_load_session_source_images_rebuilds_crops(tmp_path):
    app, created = _ensure_offscreen_qt()
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    image[10:30, 12:32] = (10, 120, 220)
    source_path = str(tmp_path / "scan.png")
    assert cv2.imwrite(source_path, image)

    part = AllClassImage(
        class_name="root",
        confidence=0.8,
        image=np.zeros((1, 1, 3), dtype=np.uint8),
        bbox=(2, 3, 10, 12),
    )
    obj = ObjectImage(
        class_name="seeding",
        confidence=0.9,
        image=[],
        image_all_class=[part],
        bbox=(12, 10, 32, 30),
    )
    state = AppState(
        image_storage=OriginalImage(
            file_path=source_path,
            images=[],
            class_object_image=[[obj]],
        )
    )
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")

    missing = window._load_session_source_images(state)

    assert missing is None
    assert len(state.image_storage.images) == 1
    assert obj.image[0].shape == (20, 20, 3)
    assert part.image.shape == (9, 8, 3)

    window.close()
    if created:
        app.quit()


def test_load_session_source_images_uses_page_source_files(tmp_path):
    app, created = _ensure_offscreen_qt()
    first_image = np.full((8, 9, 3), 40, dtype=np.uint8)
    second_image = np.full((10, 11, 3), 90, dtype=np.uint8)
    first_path = str(tmp_path / "first.png")
    second_path = str(tmp_path / "second.png")
    assert cv2.imwrite(first_path, first_image)
    assert cv2.imwrite(second_path, second_image)
    state = AppState(
        image_storage=OriginalImage(
            file_path=first_path,
            source_files=[first_path, second_path],
            images=[],
            class_object_image=[[], []],
        )
    )
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")

    missing = window._load_session_source_images(state)

    assert missing is None
    assert [image.shape[:2] for image in state.image_storage.images] == [(8, 9), (10, 11)]
    assert int(state.image_storage.images[0][0, 0, 0]) == 40
    assert int(state.image_storage.images[1][0, 0, 0]) == 90

    window.close()
    if created:
        app.quit()


def test_find_seedlings_updates_tree_and_statistics():
    """Проверяет обновление дерева и статистики после детекции на странице."""
    app, created = _ensure_offscreen_qt()
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    window._append_page(np.zeros((40, 40, 3), dtype=np.uint8), "page1.png")
    window.detect_model = _StaticBackend(
        [
            InferenceResult(
                names={0: "seeding"},
                boxes=[
                    InferenceBox(
                        cls=0,
                        conf=0.88,
                        bbox_xyxy=(5.0, 6.0, 20.0, 24.0),
                    )
                ],
            )
        ]
    )

    window.find_seedlings()
    _wait_for_analysis(window)

    assert len(window.image_storage.class_object_image[0]) == 1
    assert window.tree_widget.topLevelItemCount() == 1
    assert window.tree_widget.topLevelItem(0).isExpanded()
    assert window.statistics_panel._summary.objects_count == 1

    window.close()
    if created:
        app.quit()


def test_classify_adds_parts_to_detected_object():
    """Проверяет добавление частей и маски после классификации объекта."""
    app, created = _ensure_offscreen_qt()
    crop = np.zeros((20, 12, 3), dtype=np.uint8)
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    window.image_storage.images = [np.zeros((40, 40, 3), dtype=np.uint8)]
    window.image_storage.source_files = ["page1.png"]
    window.image_storage.file_path = "page1.png"
    window.image_storage.class_object_image = [
        [
            ObjectImage(
                class_name="seeding",
                confidence=0.9,
                image=[crop],
                bbox=(10, 10, 22, 30),
            )
        ]
    ]
    window.classify_model = _StaticBackend(
        [
            InferenceResult(
                names={0: "root"},
                boxes=[
                    InferenceBox(
                        cls=0,
                        conf=0.7,
                        bbox_xyxy=(1.0, 2.0, 8.0, 12.0),
                        mask_polygon=np.array(
                            [[1.0, 2.0], [8.0, 2.0], [8.0, 12.0], [1.0, 12.0]],
                            dtype=np.float32,
                        ),
                    )
                ],
            )
        ]
    )

    window.classify()
    _wait_for_analysis(window)

    parts = window.image_storage.class_object_image[0][0].image_all_class
    assert parts is not None
    assert len(parts) == 1
    assert parts[0].class_name == "root"
    assert parts[0].mask_polygon is not None
    # Полигон теперь извлекается из уточнённого bitmap, точное число вершин
    # варьируется; проверяем корректность формы
    assert parts[0].mask_polygon.ndim == 2
    assert parts[0].mask_polygon.shape[1] == 2
    assert parts[0].mask_polygon.shape[0] >= 3

    window.close()
    if created:
        app.quit()


def test_classify_batches_multiple_objects_preserving_order():
    """Проверяет batch-сегментацию нескольких объектов без смешивания результатов."""
    app, created = _ensure_offscreen_qt()
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    window.image_storage.images = [np.zeros((40, 40, 3), dtype=np.uint8)]
    window.image_storage.source_files = ["page1.png"]
    window.image_storage.file_path = "page1.png"
    window.image_storage.class_object_image = [
        [
            ObjectImage(
                class_name="seeding",
                confidence=0.9,
                image=[np.zeros((20, 12, 3), dtype=np.uint8)],
                bbox=(1, 1, 13, 21),
            ),
            ObjectImage(
                class_name="seeding",
                confidence=0.8,
                image=[np.zeros((20, 12, 3), dtype=np.uint8)],
                bbox=(20, 1, 32, 21),
            ),
        ]
    ]
    window.classify_model = _SequenceBackend(
        [
            [
                InferenceResult(
                    names={0: "root"},
                    boxes=[
                        InferenceBox(
                            cls=0,
                            conf=0.7,
                            bbox_xyxy=(1.0, 12.0, 8.0, 18.0),
                            mask_polygon=np.array(
                                [[1.0, 12.0], [8.0, 12.0], [8.0, 18.0]],
                                dtype=np.float32,
                            ),
                        )
                    ],
                )
            ],
            [
                InferenceResult(
                    names={0: "flower"},
                    boxes=[
                        InferenceBox(
                            cls=0,
                            conf=0.6,
                            bbox_xyxy=(2.0, 2.0, 9.0, 8.0),
                            mask_polygon=np.array(
                                [[2.0, 2.0], [9.0, 2.0], [9.0, 8.0]],
                                dtype=np.float32,
                            ),
                        )
                    ],
                )
            ],
        ]
    )

    window.classify()
    _wait_for_analysis(window)

    objects = window.image_storage.class_object_image[0]
    assert objects[0].image_all_class[0].class_name == "root"
    assert objects[1].image_all_class[0].class_name == "flower"
    assert window.classify_model.batch_sizes == [2]

    window.close()
    if created:
        app.quit()


def test_classify_repredicts_rotated_objects_in_batch():
    """Проверяет, что повернутые кропы получают финальный batch-прогон."""
    app, created = _ensure_offscreen_qt()
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    window.image_storage.images = [np.zeros((40, 40, 3), dtype=np.uint8)]
    window.image_storage.source_files = ["page1.png"]
    window.image_storage.file_path = "page1.png"
    window.image_storage.class_object_image = [
        [
            ObjectImage(
                class_name="seeding",
                confidence=0.9,
                image=[np.zeros((20, 12, 3), dtype=np.uint8)],
                bbox=(1, 1, 13, 21),
            )
        ]
    ]
    window.classify_model = _SequenceBackend(
        [
            [
                InferenceResult(
                    names={0: "root"},
                    boxes=[
                        InferenceBox(
                            cls=0,
                            conf=0.7,
                            bbox_xyxy=(1.0, 1.0, 8.0, 5.0),
                        )
                    ],
                )
            ],
            [
                InferenceResult(
                    names={0: "flower"},
                    boxes=[
                        InferenceBox(
                            cls=0,
                            conf=0.6,
                            bbox_xyxy=(2.0, 12.0, 9.0, 18.0),
                            mask_polygon=np.array(
                                [[2.0, 12.0], [9.0, 12.0], [9.0, 18.0]],
                                dtype=np.float32,
                            ),
                        )
                    ],
                )
            ],
        ]
    )

    window.classify()
    _wait_for_analysis(window)

    obj = window.image_storage.class_object_image[0][0]
    assert obj.rotation_k == 2
    assert obj.image_all_class[0].class_name == "flower"
    assert window.classify_model.batch_sizes == [1, 1]

    window.close()
    if created:
        app.quit()


def test_crop_view_renders_part_mask_overlay():
    """Проверяет отображение масок и bbox частей в режиме просмотра кропа."""
    app, created = _ensure_offscreen_qt()
    crop = np.zeros((20, 12, 3), dtype=np.uint8)
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    window.image_storage.images = [np.zeros((40, 40, 3), dtype=np.uint8)]
    window.image_storage.source_files = ["page1.png"]
    window.image_storage.file_path = "page1.png"
    window.image_storage.class_object_image = [
        [
            ObjectImage(
                class_name="seeding",
                confidence=0.9,
                image=[crop],
                bbox=(10, 10, 22, 30),
                image_all_class=[
                    AllClassImage(
                        class_name="root",
                        confidence=0.75,
                        image=np.zeros((10, 7, 3), dtype=np.uint8),
                        bbox=(1, 2, 8, 12),
                        mask_polygon=np.array(
                            [[1.0, 2.0], [8.0, 2.0], [8.0, 12.0], [1.0, 12.0]],
                            dtype=np.float32,
                        ),
                    )
                ],
            )
        ]
    ]

    window.display_image_with_boxes(0, seeding_idx=0)

    assert window.show_boxes_button.isChecked()
    assert window.show_masks_button.isChecked()
    assert window.view_mode_button.isChecked()
    assert window.edit_boxes_mode_button.isEnabled()
    assert not window.edit_masks_mode_button.isEnabled()
    assert len(window.mask_items) == 1
    assert len(window.rect_items) == 1
    assert not (
        window.rect_items[0].flags() & QGraphicsItem.ItemIsMovable
    )
    window._set_interaction_mode("edit_boxes")
    assert window.rect_items[0].flags() & QGraphicsItem.ItemIsMovable

    window.show_masks_button.setChecked(False)
    assert len(window.mask_items) == 0
    assert len(window.rect_items) == 1

    window.show_boxes_button.setChecked(False)
    assert len(window.rect_items) == 0

    window.close()
    if created:
        app.quit()


def test_manual_seedling_bbox_commit_rebuilds_crop_and_clears_parts(monkeypatch):
    app, created = _ensure_offscreen_qt()
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    obj = ObjectImage(
        class_name="seeding",
        confidence=0.8,
        image=[np.zeros((10, 10, 3), dtype=np.uint8)],
        bbox=(5, 5, 15, 15),
        image_all_class=[
            AllClassImage(
                class_name="root",
                confidence=0.5,
                image=np.zeros((3, 3, 3), dtype=np.uint8),
                bbox=(1, 1, 4, 4),
            )
        ],
    )
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    monkeypatch.setattr(window, "_auto_save_session", lambda: None)
    window.image_storage.images = [image]
    window.image_storage.source_files = ["page1.png"]
    window.image_storage.file_path = "page1.png"
    window.image_storage.class_object_image = [[obj]]

    obj.bbox = (10, 12, 30, 32)
    window._commit_seedling_bbox(0, obj)

    assert obj.bbox == (10, 12, 30, 32)
    assert obj.image[0].shape == (20, 20, 3)
    assert obj.image_all_class is None

    window.close()
    if created:
        app.quit()


def test_manual_add_part_box_appends_part(monkeypatch):
    app, created = _ensure_offscreen_qt()
    seed_obj = ObjectImage(
        class_name="seeding",
        confidence=0.9,
        image=[np.zeros((30, 20, 3), dtype=np.uint8)],
        bbox=(0, 0, 20, 30),
        image_all_class=[],
    )
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    monkeypatch.setattr(window, "_auto_save_session", lambda: None)
    monkeypatch.setattr(window, "_choose_annotation_class", lambda classes, title: "root")
    window.image_storage.images = [np.zeros((40, 40, 3), dtype=np.uint8)]
    window.image_storage.source_files = ["page1.png"]
    window.image_storage.file_path = "page1.png"
    window.image_storage.class_object_image = [[seed_obj]]
    window.display_image_with_boxes(0, seeding_idx=0)

    window._add_part_box(seed_obj, (2, 3, 12, 18))

    assert len(seed_obj.image_all_class) == 1
    part = seed_obj.image_all_class[0]
    assert part.class_name == "root"
    assert part.bbox == (2, 3, 12, 18)
    assert part.image.shape == (15, 10, 3)
    assert part.mask_bitmap is not None
    assert part.mask_polygon is not None

    window.close()
    if created:
        app.quit()


def test_pending_seedling_box_requires_confirmation_and_starts_segmentation(monkeypatch):
    app, created = _ensure_offscreen_qt()
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    window.image_storage.images = [np.zeros((40, 40, 3), dtype=np.uint8)]
    window.image_storage.source_files = ["page1.png"]
    window.image_storage.file_path = "page1.png"
    window.image_storage.class_object_image = [[]]
    window.display_image_with_boxes(0)
    monkeypatch.setattr(window, "_auto_save_session", lambda: None)
    monkeypatch.setattr(window, "_choose_annotation_class", lambda classes, title: "cedr")
    started = []

    def fake_segment(page_index, object_index, obj):
        started.append((page_index, object_index, obj.bbox))

    monkeypatch.setattr(window, "_segment_single_seedling", fake_segment)

    window._set_interaction_mode("edit_boxes")
    window._create_pending_box(QRectF(5, 6, 12, 14))

    assert len(window.image_storage.class_object_image[0]) == 0
    assert window.confirm_box_button.isEnabled()

    window._pending_box_item.setRect(QRectF(8, 9, 10, 12))
    window._confirm_pending_box()

    assert len(window.image_storage.class_object_image[0]) == 1
    obj = window.image_storage.class_object_image[0][0]
    assert obj.class_name == "cedr"
    assert obj.bbox == (8, 9, 18, 21)
    assert started == [(0, 0, (8, 9, 18, 21))]

    window.close()
    if created:
        app.quit()


def test_annotation_toolbar_follows_edit_mode_and_selection():
    app, created = _ensure_offscreen_qt()
    obj = ObjectImage(
        class_name="seeding",
        confidence=0.9,
        image=[np.zeros((10, 10, 3), dtype=np.uint8)],
        bbox=(4, 5, 14, 15),
    )
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    window.image_storage.images = [np.zeros((24, 24, 3), dtype=np.uint8)]
    window.image_storage.source_files = ["page1.png"]
    window.image_storage.file_path = "page1.png"
    window.image_storage.class_object_image = [[obj]]

    window.display_image_with_boxes(0)

    assert not window.add_box_button.isEnabled()
    assert not window.delete_annotation_button.isEnabled()

    window._set_interaction_mode("edit_boxes")
    assert window.add_box_button.isEnabled()
    assert not window.delete_annotation_button.isEnabled()

    window.rect_items[0].setSelected(True)
    QApplication.processEvents()
    window._update_annotation_controls()
    assert window.delete_annotation_button.isEnabled()

    window.close()
    if created:
        app.quit()


def test_create_report_updates_last_report_path(tmp_path, monkeypatch):
    """Проверяет сохранение пути к последнему отчёту после его создания."""
    app, created = _ensure_offscreen_qt()
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    window.image_storage.images = [np.zeros((32, 32, 3), dtype=np.uint8)]
    window.image_storage.source_files = [str(tmp_path / "page1.png")]
    window.image_storage.file_path = str(tmp_path / "page1.png")
    window.image_storage.class_object_image = [[]]

    output_path = tmp_path / "report.pdf"
    monkeypatch.setattr(
        "seeding.ui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(output_path), "PDF Files (*.pdf)"),
    )
    monkeypatch.setattr(window, "_show_info", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "seeding.ui.main_window.generate_report",
        lambda state, path: path,
    )

    window.create_report()

    assert window.app_state.last_report_path == str(output_path)

    window.close()
    if created:
        app.quit()


def test_pdf_page_titles_include_local_page_numbers():
    """Проверяет локальную нумерацию страниц для разных PDF-источников."""
    app, created = _ensure_offscreen_qt()
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")

    window._append_page(np.zeros((16, 16, 3), dtype=np.uint8), "pak1.pdf")
    window._append_page(np.zeros((16, 16, 3), dtype=np.uint8), "pak1.pdf")
    window._append_page(np.zeros((16, 16, 3), dtype=np.uint8), "pak2.pdf")

    assert "стр. 1/2" in window._page_title(0)
    assert "стр. 2/2" in window._page_title(1)
    assert "стр. 1/1" in window._page_title(2)

    window.close()
    if created:
        app.quit()


def test_tree_and_statistics_follow_active_page():
    """Проверяет синхронизацию дерева и статистики с активной страницей."""
    app, created = _ensure_offscreen_qt()
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    window.image_storage.images = [
        np.zeros((40, 40, 3), dtype=np.uint8),
        np.zeros((40, 40, 3), dtype=np.uint8),
    ]
    window.image_storage.source_files = ["page1.png", "page2.png"]
    window.image_storage.file_path = "page1.png"
    window.image_storage.class_object_image = [
        [
            ObjectImage(
                class_name="seeding",
                confidence=0.9,
                image=[np.zeros((10, 10, 3), dtype=np.uint8)],
                bbox=(1, 1, 10, 10),
            )
        ],
        [
            ObjectImage(
                class_name="seeding",
                confidence=0.9,
                image=[np.zeros((10, 10, 3), dtype=np.uint8)],
                bbox=(1, 1, 10, 10),
            ),
            ObjectImage(
                class_name="seeding",
                confidence=0.8,
                image=[np.zeros((10, 10, 3), dtype=np.uint8)],
                bbox=(12, 12, 22, 22),
            ),
        ],
    ]
    window._refresh_tree()

    window._select_page(1)

    assert window.tree_widget.topLevelItemCount() == 2
    assert window.tree_widget.currentItem().data(0, Qt.UserRole)["index"] == 1
    assert window.statistics_panel._summary.objects_count == 2

    window.close()
    if created:
        app.quit()


def test_measure_label_ignores_view_transform():
    """Проверяет, что подпись измерения не масштабируется вместе с видом."""
    app, created = _ensure_offscreen_qt()
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    window._append_page(np.zeros((80, 80, 3), dtype=np.uint8), "page1.png")
    window.display_image_with_boxes(0)
    window._set_measure_mode(True)
    window._start_manual_measure(QPointF(10.0, 10.0))

    assert window._measure_text_item is not None
    assert (
        window._measure_text_item.flags()
        & QGraphicsItem.ItemIgnoresTransformations
    )

    window.close()
    if created:
        app.quit()


def test_calibration_restores_per_source_file(tmp_path):
    """Проверяет раздельное восстановление калибровки для разных файлов."""
    app, created = _ensure_offscreen_qt()
    _isolate_qsettings(tmp_path)
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    window._append_page(np.zeros((32, 32, 3), dtype=np.uint8), str(tmp_path / "a.png"))
    window._append_page(np.zeros((32, 32, 3), dtype=np.uint8), str(tmp_path / "b.png"))

    window._select_page(0)
    window._save_calibration_for_current_source(12.5)
    window._select_page(1)
    assert window.pixels_per_mm == 0.0

    window._select_page(0)
    assert abs(window.pixels_per_mm - 12.5) < 1e-9

    window.close()
    if created:
        app.quit()




@pytest.fixture
def fake_pixmap():
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QGraphicsPixmapItem
    pix = QPixmap(100, 100)
    pix.fill()
    return QGraphicsPixmapItem(pix)


@pytest.fixture
def main_window():
    app, created = _ensure_offscreen_qt()
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    yield window
    window.close()
    if created:
        app.quit()


def test_tools_stack_view_mode(main_window):
    """В режиме Просмотр стек показывает страницу 0."""
    main_window._set_interaction_mode("view")
    assert main_window.tools_stack.currentIndex() == 0


def test_main_window_uses_local_two_panel_layout(main_window):
    """The project tree must stay in the left panel without a duplicate right sidebar."""
    splitter = main_window.findChild(QSplitter)

    assert splitter is not None
    assert splitter.count() == 2
    assert not hasattr(main_window, "right_tabs")


def test_tools_stack_edit_boxes_mode(main_window):
    """В режиме Ред. боксов стек показывает страницу 1 (idle)."""
    main_window._set_interaction_mode("edit_boxes")
    assert main_window.tools_stack.currentIndex() == 1


def test_tools_stack_drawing_mode(main_window, fake_pixmap):
    """После _start_add_box_mode стек показывает страницу 2."""
    main_window._set_interaction_mode("edit_boxes")
    main_window._pixmap_item = fake_pixmap
    main_window._start_add_box_mode()
    assert main_window.tools_stack.currentIndex() == 2


def test_tools_stack_cancel_returns_to_edit(main_window, fake_pixmap):
    """После отмены рисования стек возвращается на страницу 1."""
    main_window._set_interaction_mode("edit_boxes")
    main_window._pixmap_item = fake_pixmap
    main_window._start_add_box_mode()
    main_window._cancel_add_box_mode()
    assert main_window.tools_stack.currentIndex() == 1
