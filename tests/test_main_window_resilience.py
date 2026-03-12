import os

import numpy as np
from PyQt5.QtCore import QPointF, QSettings, Qt
from PyQt5.QtWidgets import QGraphicsItem
from PyQt5.QtWidgets import QApplication

from seeding.config import QSETTINGS_APP, QSETTINGS_ORG
from seeding.inference import InferenceBox, InferenceResult
from seeding.models import AllClassImage, ObjectImage
from seeding.ui.main_window import ImageEditor


def _ensure_offscreen_qt() -> tuple[QApplication, bool]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    created = app is None
    if app is None:
        app = QApplication([])
    return app, created


def _isolate_qsettings(tmp_path) -> None:
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path))
    settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
    settings.clear()
    settings.sync()


class _StaticBackend:
    def __init__(self, results):
        self._results = results

    def predict(self, image, *, conf_threshold=None):
        _ = (image, conf_threshold)
        return self._results


def test_append_page_keeps_storage_lists_in_sync():
    app, created = _ensure_offscreen_qt()
    window = ImageEditor("dummy_weights.pt", "dummy_classify.pt")
    assert window.canvas_stack.currentWidget() is window.empty_state
    assert window.right_tabs.count() == 2

    window._append_page(np.zeros((32, 32, 3), dtype=np.uint8), "page1.png")
    window._append_page(np.zeros((24, 24, 3), dtype=np.uint8), "page2.png")
    window.display_image_with_boxes(0)

    assert len(window.image_storage.images) == 2
    assert len(window.image_storage.source_files) == 2
    assert len(window.image_storage.class_object_image or []) == 2
    assert window.project_files_list.count() == 2
    assert window.canvas_stack.currentWidget() is window.graphics_view

    window.close()
    if created:
        app.quit()


def test_find_seedlings_updates_tree_and_statistics():
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

    assert len(window.image_storage.class_object_image[0]) == 1
    assert window.tree_widget.topLevelItemCount() == 1
    assert not window.tree_widget.topLevelItem(0).isExpanded()
    assert window.statistics_panel._summary.objects_count == 1

    window.close()
    if created:
        app.quit()


def test_classify_adds_parts_to_detected_object():
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

    parts = window.image_storage.class_object_image[0][0].image_all_class
    assert parts is not None
    assert len(parts) == 1
    assert parts[0].class_name == "root"
    assert parts[0].mask_polygon is not None
    assert parts[0].mask_polygon.shape == (4, 2)

    window.close()
    if created:
        app.quit()


def test_crop_view_renders_part_mask_overlay():
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
    assert not window.edit_boxes_mode_button.isEnabled()
    assert not window.edit_masks_mode_button.isEnabled()
    assert len(window.mask_items) == 1
    assert len(window.rect_items) == 1
    assert not (
        window.rect_items[0].flags() & QGraphicsItem.ItemIsMovable
    )

    window.show_masks_button.setChecked(False)
    assert len(window.mask_items) == 0
    assert len(window.rect_items) == 1

    window.show_boxes_button.setChecked(False)
    assert len(window.rect_items) == 0

    window.close()
    if created:
        app.quit()


def test_create_report_updates_last_report_path(tmp_path, monkeypatch):
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
        window.app_controller.report_service,
        "generate_report",
        lambda data, path: path,
    )

    window.create_report()

    assert window.app_state.last_report_path == str(output_path)

    window.close()
    if created:
        app.quit()


def test_pdf_page_titles_include_local_page_numbers():
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
    window._refresh_thumbnails_panel()

    window._select_page(1)

    assert window.tree_widget.topLevelItemCount() == 1
    assert window.tree_widget.topLevelItem(0).data(0, Qt.UserRole)["index"] == 1
    assert window.statistics_panel._summary.objects_count == 2

    window.close()
    if created:
        app.quit()


def test_measure_label_ignores_view_transform():
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
