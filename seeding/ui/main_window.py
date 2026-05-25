"""Главное окно приложения Seeding с загрузкой, анализом и просмотром результатов."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt5.QtCore import (
    QEvent,
    QPoint,
    QPointF,
    QRectF,
    QSettings,
    QSize,
    Qt,
    QThread,
    pyqtSignal,
)
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
)
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSplitter,
    QStackedLayout,
    QStackedWidget,
    QStyle,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from seeding.auth import AuthUser
from seeding.config import (
    CALIBRATION_PIXELS_PER_MM_DEFAULT,
    CLASS_DISPLAY_NAMES,
    DEFAULT_CLASSIFY_WEIGHTS_PATH,
    DETECTION_CLASS_NAMES,
    DETECTION_CONFIDENCE_THRESHOLD,
    DETECTION_IOU_THRESHOLD,
    PANEL_LAYERS_MAX_WIDTH,
    PANEL_LAYERS_MIN_WIDTH,
    QSETTINGS_APP,
    QSETTINGS_ORG,
    ROTATE_ANGLE_DEG,
    ROTATE_K,
    SPLITTER_SIZES,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    WINDOW_X,
    WINDOW_Y,
    YOLO_BATCH_SIZE_CLASSIFY,
    ZOOM_FACTOR_INCREMENT,
)
from seeding.image_loader import (
    ImageLoadError,
    load_pages_from_path,
)
from seeding.inference import InferenceBackend, load_inference_backend
from seeding.mask_refiner import (
    bitmap_to_contours,
    bitmap_to_polygon,
    build_refine_context,
    polygon_to_bitmap,
    refine_mask_bitmap,
)

try:
    import torch  # noqa: F401 — загружает c10.dll в главном потоке до QThread
except Exception:
    pass
from seeding.models import (
    AllClassImage,
    AppState,
    ObjectImage,
    OriginalImage,
    SelectionPayload,
)
from seeding.services import (
    generate_report,
    orient_crop_upright,
    rotate_selection,
    run_classification_for_selection,
    run_detection,
)
from seeding.session_service import load_session
from seeding.ui.bbox_item import BBoxItem
from seeding.ui.detail_panel import SeedlingDetailPanel
from seeding.ui.icon_manager import IconManager
from seeding.ui.session_actions import (
    auto_save_current_session,
    restore_session_images,
    save_current_session,
    sync_current_user_to_state,
)
from seeding.ui.session_picker import SessionPickerDialog
from seeding.ui.settings_dialog import AppSettingsDialog
from seeding.ui.statistics_panel import StatisticsPanel
from seeding.ui.tree_widget import LayerTreeWidget
from seeding.user_service import record_user_action
from seeding.utils import clip_bbox_to_image, rotate_bbox

logger = logging.getLogger(__name__)

INPUT_FILE_FILTER = (
    "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;"
    "PDF Files (*.pdf);;"
    "All Files (*)"
)


class ModelLoaderThread(QThread):
    """Фоновый поток для предзагрузки моделей детекции и сегментации."""

    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, detect_path: str, classify_path: str, parent=None) -> None:
        super().__init__(parent)
        self._detect_path = detect_path
        self._classify_path = classify_path

    def run(self) -> None:
        try:
            detect = load_inference_backend(self._detect_path)
            classify = load_inference_backend(self._classify_path)
            self.finished.emit(detect, classify)
        except Exception as exc:
            self.error.emit(str(exc))


class DetectionWorkerThread(QThread):
    """Runs YOLO detection outside the UI thread and updates AppState when done."""

    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        model: InferenceBackend,
        state: AppState,
        page_indices: list[int],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._state = state
        self._page_indices = list(page_indices)
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            total = len(self._page_indices)
            processed = 0
            page_counts: dict[int, int] = {}
            for step, page_index in enumerate(self._page_indices, start=1):
                if self._cancel_requested:
                    break
                self.progress.emit(
                    processed,
                    total,
                    f"Страница {page_index + 1}... ({step}/{total})",
                )
                image = self._state.image_storage.images[page_index]
                results = self._model.predict(
                    image,
                    conf_threshold=DETECTION_CONFIDENCE_THRESHOLD,
                )
                objects = run_detection(
                    self._state,
                    page_index,
                    results,
                    detection_class_names=DETECTION_CLASS_NAMES,
                    iou_threshold=DETECTION_IOU_THRESHOLD,
                    rotate_k=ROTATE_K,
                )
                processed += 1
                page_counts[page_index] = len(objects)
                self.progress.emit(processed, total, "")
            self.finished_ok.emit(
                {
                    "processed": processed,
                    "total": total,
                    "page_counts": page_counts,
                    "canceled": self._cancel_requested,
                }
            )
        except Exception as exc:
            self.error.emit(str(exc))


class ClassificationWorkerThread(QThread):
    """Runs YOLO segmentation/classification outside the UI thread."""

    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        model: InferenceBackend,
        state: AppState,
        pending: list[tuple[int, int, ObjectImage]],
        batch_size: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._state = state
        self._pending = list(pending)
        self._batch_size = max(1, int(batch_size))
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            processed = 0
            total = len(self._pending)
            for batch in self._iter_batches(self._pending, self._batch_size):
                if self._cancel_requested:
                    break

                self.progress.emit(processed, total, f"Сеянец {processed + 1} из {total}...")
                orient_results_batch = self._model.predict_batch(
                    [obj.image[0] for _, _, obj in batch]
                )

                final_results_by_item: dict[int, list] = {}
                rotated_items: list[tuple[int, ObjectImage]] = []
                rotated_images: list[np.ndarray] = []
                probe_items: list[tuple[int, ObjectImage]] = []
                probe_images: list[np.ndarray] = []

                for local_index, (_, _, obj) in enumerate(batch):
                    orient_results = orient_results_batch[local_index]
                    oriented_crop, was_rotated, uncertain = orient_crop_upright(
                        obj.image[0], orient_results
                    )
                    if was_rotated:
                        obj.image[0] = oriented_crop
                        obj.rotation_k = (obj.rotation_k + 2) % 4
                        rotated_items.append((local_index, obj))
                        rotated_images.append(obj.image[0])
                    else:
                        final_results_by_item[local_index] = orient_results
                    obj.orientation_uncertain = uncertain
                    if (
                        not was_rotated
                        and self._orientation_needs_rotated_probe(orient_results)
                    ):
                        probe_items.append((local_index, obj))
                        probe_images.append(np.rot90(obj.image[0], k=2))

                if probe_images and not self._cancel_requested:
                    probe_results_batch = self._model.predict_batch(probe_images)
                    for probe_index, (local_index, obj) in enumerate(probe_items):
                        if self._results_have_boxes(probe_results_batch[probe_index]):
                            obj.image[0] = probe_images[probe_index]
                            obj.rotation_k = (obj.rotation_k + 2) % 4
                            rotated_items.append((local_index, obj))
                            rotated_images.append(obj.image[0])
                        else:
                            final_results_by_item[local_index] = orient_results_batch[
                                local_index
                            ]

                if rotated_images and not self._cancel_requested:
                    rotated_results_batch = self._model.predict_batch(rotated_images)
                    for rotated_index, (local_index, _) in enumerate(rotated_items):
                        final_results_by_item[local_index] = rotated_results_batch[
                            rotated_index
                        ]

                for local_index, (page_index, object_index, _) in enumerate(batch):
                    if self._cancel_requested:
                        break
                    run_classification_for_selection(
                        self._state,
                        page_index,
                        object_index,
                        final_results_by_item.get(local_index, []),
                    )
                    processed += 1
                    self.progress.emit(processed, total, "")

            self.finished_ok.emit(
                {
                    "processed": processed,
                    "total": total,
                    "canceled": self._cancel_requested,
                }
            )
        except Exception as exc:
            self.error.emit(str(exc))

    @staticmethod
    def _iter_batches(items, batch_size: int):
        for start in range(0, len(items), batch_size):
            yield items[start : start + batch_size]

    @staticmethod
    def _orientation_needs_rotated_probe(results) -> bool:
        if not results:
            return True
        marker_classes = {
            "root",
            "rootcedr",
            "rootpinus",
            "flower",
            "inflorescence",
            "flowercedr",
            "flowerpinus",
        }
        for result in results:
            for box in result.boxes:
                class_name = str(result.names[int(box.cls)]).lower()
                if class_name in marker_classes:
                    return False
        return True

    @staticmethod
    def _results_have_boxes(results) -> bool:
        return any(bool(result.boxes) for result in results or [])


class AnalysisProgressDialog(QDialog):
    """Стилизованный диалог прогресса анализа в тёмной теме приложения."""

    canceled = pyqtSignal()

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent, Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setModal(True)
        self.setFixedWidth(420)

        self._canceled = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        self._title_label = QLabel(title)
        self._title_label.setProperty("progressTitle", "true")
        font = self._title_label.font()
        font.setPointSize(font.pointSize() + 1)
        font.setBold(True)
        self._title_label.setFont(font)
        layout.addWidget(self._title_label)

        self._step_label = QLabel("")
        self._step_label.setObjectName("progressStepLabel")
        layout.addWidget(self._step_label)

        self._bar = QProgressBar()
        self._bar.setFixedHeight(18)
        self._bar.setTextVisible(True)
        self._bar.setFormat("%v / %m")
        layout.addWidget(self._bar)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._cancel_btn = QPushButton("Отмена")
        self._cancel_btn.setFixedWidth(100)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)
        layout.addLayout(btn_row)

    def _on_cancel(self) -> None:
        self._canceled = True
        self.canceled.emit()

    def wasCanceled(self) -> bool:
        return self._canceled

    def setRange(self, minimum: int, maximum: int) -> None:
        self._bar.setRange(minimum, maximum)

    def setValue(self, value: int) -> None:
        self._bar.setValue(value)
        QApplication.processEvents()

    def setStep(self, text: str) -> None:
        self._step_label.setText(text)
        QApplication.processEvents()


class CanvasGraphicsView(QGraphicsView):
    """Графический viewport с поддержкой прокрутки средней кнопкой мыши."""

    box_drawn = pyqtSignal(object)
    draw_mode_cancelled = pyqtSignal()

    def __init__(self, scene: QGraphicsScene, parent=None) -> None:
        """Создаёт viewport и подготавливает поля для ручного перемещения холста."""
        super().__init__(scene, parent)
        self._drag_active = False
        self._drag_start_pos = QPoint()
        self._scroll_start_pos = QPoint()
        self._draw_mode = False
        self._draw_active = False
        self._draw_start_pos = QPointF()
        self._rubber_item = None

    def set_draw_mode(self, enabled: bool) -> None:
        """Переключает режим рисования боксов."""
        self._draw_mode = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        if not enabled and self._rubber_item is not None:
            self.scene().removeItem(self._rubber_item)
            self._rubber_item = None
        self._draw_active = False

    def mousePressEvent(self, event) -> None:
        """Включает режим перетаскивания холста при нажатии средней кнопки мыши."""
        if self._draw_mode and event.button() == Qt.LeftButton:
            self._draw_active = True
            self._draw_start_pos = self.mapToScene(event.pos())
            event.accept()
            return
        if event.button() == Qt.MiddleButton:
            self._drag_active = True
            self.setCursor(Qt.ClosedHandCursor)
            self._drag_start_pos = event.pos()
            self._scroll_start_pos = QPoint(
                self.horizontalScrollBar().value(),
                self.verticalScrollBar().value(),
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """Перемещает холст вслед за курсором, пока активен режим ручной прокрутки."""
        if self._draw_active:
            current = self.mapToScene(event.pos())
            rect = QRectF(self._draw_start_pos, current).normalized()
            if self._rubber_item is None:
                self._rubber_item = QGraphicsRectItem()
                pen = QPen(QColor(255, 255, 0), 2, Qt.DashLine)
                self._rubber_item.setPen(pen)
                self._rubber_item.setBrush(Qt.transparent)
                self._rubber_item.setZValue(10)
                self.scene().addItem(self._rubber_item)
            self._rubber_item.setRect(rect)
            event.accept()
            return
        if self._drag_active:
            delta = event.pos() - self._drag_start_pos
            self.horizontalScrollBar().setValue(self._scroll_start_pos.x() - delta.x())
            self.verticalScrollBar().setValue(self._scroll_start_pos.y() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """Завершает ручное перемещение холста после отпускания средней кнопки."""
        if self._draw_active and event.button() == Qt.LeftButton:
            self._draw_active = False
            current = self.mapToScene(event.pos())
            rect = QRectF(self._draw_start_pos, current).normalized()
            if self._rubber_item is not None:
                self.scene().removeItem(self._rubber_item)
                self._rubber_item = None
            if rect.width() > 5 and rect.height() > 5:
                self.box_drawn.emit(rect)
            event.accept()
            return
        if event.button() == Qt.MiddleButton:
            self._drag_active = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if self._draw_mode and event.key() == Qt.Key_Escape:
            self.set_draw_mode(False)
            self.draw_mode_cancelled.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class TaskProgressDialog(QDialog):
    """Компактный диалог прогресса в стиле приложения."""

    def __init__(
        self,
        title: str,
        message: str,
        *,
        maximum: int,
        parent: QWidget | None = None,
        cancel_text: str = "Отмена",
    ) -> None:
        super().__init__(parent)
        self._canceled = False
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setObjectName("panelCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        self.title_label = QLabel(title, self)
        self.title_label.setObjectName("panelCardTitle")
        layout.addWidget(self.title_label)

        self.message_label = QLabel(message, self)
        self.message_label.setObjectName("panelHint")
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, maximum)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m")
        layout.addWidget(self.progress_bar)

        self.detail_label = QLabel("", self)
        self.detail_label.setObjectName("panelHint")
        self.detail_label.setWordWrap(True)
        self.detail_label.setVisible(False)
        layout.addWidget(self.detail_label)

        actions_layout = QHBoxLayout()
        actions_layout.addStretch(1)
        self.cancel_button = QPushButton(cancel_text, self)
        self.cancel_button.setProperty("variant", "secondary")
        self.cancel_button.clicked.connect(self._handle_cancel)
        actions_layout.addWidget(self.cancel_button)
        layout.addLayout(actions_layout)

    def _handle_cancel(self) -> None:
        """Помечает операцию как отменённую и закрывает диалог."""

        self._canceled = True
        self.close()

    def set_message(self, message: str, *, detail: str | None = None) -> None:
        """Обновляет подпись и дополнительный статус."""

        self.message_label.setText(message)
        if detail:
            self.detail_label.setText(detail)
            self.detail_label.setVisible(True)
        else:
            self.detail_label.clear()
            self.detail_label.setVisible(False)
        QApplication.processEvents()

    def set_progress(
        self,
        value: int,
        *,
        message: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Обновляет значение прогресса и подписи."""

        if message is not None or detail is not None:
            self.set_message(message or self.message_label.text(), detail=detail)
        self.progress_bar.setValue(value)
        QApplication.processEvents()

    def was_canceled(self) -> bool:
        """Возвращает `True`, если пользователь отменил операцию."""

        return self._canceled


class ImageEditor(QMainWindow):
    """Главное окно приложения с загрузкой файлов, анализом и просмотром результатов."""

    logout_requested = pyqtSignal()

    def __init__(
        self,
        weights_path: str,
        classify_weights_path: str | None = None,
        current_user: AuthUser | None = None,
    ) -> None:
        """Инициализирует состояние окна, модели, настройки и основные элементы UI."""
        super().__init__()
        self.weights_path = str(weights_path)
        self.classify_weights_path = str(
            classify_weights_path or DEFAULT_CLASSIFY_WEIGHTS_PATH
        )
        self.current_user = current_user
        self.detect_model: InferenceBackend | None = None
        self.classify_model: InferenceBackend | None = None
        self.app_state = AppState(
            image_storage=OriginalImage(),
            current_user_id=current_user.id if current_user else None,
        )
        self.image_storage = self.app_state.image_storage
        self._active_image_index = 0
        self._display_target: tuple[Any, ...] = ("page", 0)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._original_pixmap: QPixmap | None = None
        self.rect_items: list[BBoxItem] = []
        self.mask_items: list[QGraphicsItem] = []
        self._show_boxes = True
        self._show_masks = True
        self._interaction_mode = "view"
        self._box_draw_active = False
        self._box_draw_start: QPointF | None = None
        self._box_draw_item: QGraphicsRectItem | None = None
        self._pending_box_item: BBoxItem | None = None
        self._pending_box_target: tuple[Any, ...] | None = None
        self.zoom_factor = 1.0
        self.min_fit_zoom = 1.0
        self.pixels_per_mm = CALIBRATION_PIXELS_PER_MM_DEFAULT
        self._measure_mode = False
        self._calibration_pending = False
        self._measure_start_scene_pos: QPointF | None = None
        self._measure_line_item: QGraphicsLineItem | None = None
        self._measure_text_item: QGraphicsTextItem | None = None
        self._resize_active = False
        self._resize_edges = Qt.Edges()
        self._resize_start_geometry = QRectF()
        self._resize_start_global_pos = QPoint()
        self._resize_margin = 8
        self._window_drag_active = False
        self._window_drag_offset = QPoint()
        self._settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        self._analysis_batch_size = self._settings.value(
            "analysis/yolo_batch_size",
            YOLO_BATCH_SIZE_CLASSIFY,
            type=int,
        )
        self._analysis_device = self._settings.value(
            "analysis/yolo_device",
            os.getenv("YOLO_DEVICE", "auto"),
            type=str,
        ) or "auto"
        self._auto_save_enabled = self._settings.value(
            "app/auto_save_enabled",
            True,
            type=bool,
        )
        self._report_output_dir = self._settings.value(
            "report/output_dir",
            "",
            type=str,
        ) or ""
        self._analysis_thread: QThread | None = None
        self._analysis_progress: AnalysisProgressDialog | None = None
        self._apply_inference_settings()
        self.app_state.zoom_factor = self.zoom_factor
        self.app_state.pixels_per_mm = self.pixels_per_mm

        self.setWindowTitle("Seeding")
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setGeometry(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)

        self.icon_manager = IconManager(self)
        self._build_ui()
        self._build_toolbar()
        self._build_menu()
        self._install_frameless_event_filters()
        self._update_window_control_buttons()
        QShortcut(QKeySequence("M"), self, self.toggle_measure_mode)
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(
            lambda: self._navigate_seedling(1)
        )
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(
            lambda: self._navigate_seedling(-1)
        )
        self.statusBar().showMessage("Откройте изображение или PDF", 3000)

    def _apply_inference_settings(self) -> None:
        """Применяет настройки инференса к окружению для новых backend-экземпляров."""
        device = str(self._analysis_device or "auto").strip() or "auto"
        os.environ["YOLO_DEVICE"] = device
        self._analysis_device = device
        self._analysis_batch_size = max(1, int(self._analysis_batch_size or 1))

    def _sync_current_user_to_state(self) -> None:
        """Гарантирует, что AppState знает текущего авторизованного пользователя."""
        sync_current_user_to_state(self.app_state, self.current_user)

    def _load_session_source_images(self, state: AppState) -> str | None:
        """Load source images for a restored session and rebuild saved crops."""
        return restore_session_images(state, logger=logger)

    def preload_models(self) -> None:
        """Запускает фоновую загрузку обеих моделей сразу после показа окна."""
        self._model_loader = ModelLoaderThread(
            self.weights_path, self.classify_weights_path, self
        )
        self._model_loader.finished.connect(self._on_models_loaded)
        self._model_loader.error.connect(self._on_models_load_error)
        self.statusBar().showMessage("Загрузка моделей…")
        self._model_loader.start()

    def _on_models_loaded(self, detect_model, classify_model) -> None:
        self.detect_model = detect_model
        self.classify_model = classify_model
        self.statusBar().showMessage("Модели загружены", 3000)

    def _on_models_load_error(self, error_text: str) -> None:
        self.statusBar().showMessage("Ошибка загрузки моделей", 5000)
        logger.error("Не удалось предзагрузить модели: %s", error_text)

    def _log_action(self, action: str, details: str | None = None) -> None:
        """Сохраняет действие пользователя, не прерывая UI при ошибках логирования."""

        if self.current_user is None:
            return
        try:
            record_user_action(self.current_user.id, action, details)
        except Exception:
            logger.exception("Не удалось записать лог действия %s", action)

    def _log_error(self, details: str) -> None:
        """Сохраняет минимальную запись об ошибке приложения для текущего пользователя."""

        self._log_action("application_error", details)

    def _build_ui(self) -> None:
        """Строит основную компоновку окна, панели, холст и вкладки боковой панели."""
        central = QWidget(self)
        central.setObjectName("appShell")
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        root_layout.addWidget(splitter)

        left_panel = QFrame(self)
        left_panel.setObjectName("panelCard")
        left_panel.setMinimumWidth(260)
        left_panel.setMaximumWidth(360)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)

        project_title = QLabel("Файлы проекта", left_panel)
        project_title.setObjectName("panelCardTitle")
        left_layout.addWidget(project_title)

        self.project_count_chip = QLabel("0 страниц", left_panel)
        self.project_count_chip.setObjectName("metricChip")
        left_layout.addWidget(self.project_count_chip, alignment=Qt.AlignLeft)

        self.tree_widget = LayerTreeWidget()
        self.tree_widget.itemSelectionChanged.connect(
            self._on_tree_selection_changed
        )
        left_layout.addWidget(self.tree_widget, 1)

        self.statistics_panel = StatisticsPanel(self)
        self.statistics_panel.setMaximumHeight(180)
        left_layout.addWidget(self.statistics_panel)

        interaction_card = QFrame(left_panel)
        interaction_card.setObjectName("imageInteractionCard")
        interaction_layout = QVBoxLayout(interaction_card)
        interaction_layout.setContentsMargins(12, 12, 12, 12)
        interaction_layout.setSpacing(10)

        interaction_title = QLabel("Взаимодействие с изображением", interaction_card)
        interaction_title.setObjectName("panelSubTitle")
        interaction_title.setWordWrap(True)
        interaction_layout.addWidget(interaction_title)

        # --- Переключатель режима (всегда виден) ---
        mode_bar = QFrame(interaction_card)
        mode_bar.setObjectName("interactionModeBar")
        mode_bar_layout = QHBoxLayout(mode_bar)
        mode_bar_layout.setContentsMargins(6, 6, 6, 6)
        mode_bar_layout.setSpacing(6)

        self.view_mode_button = QPushButton("Просмотр", mode_bar)
        self.view_mode_button.setCheckable(True)
        self.view_mode_button.setChecked(True)
        self.view_mode_button.setProperty("segmented", "true")
        self.view_mode_button.toggled.connect(
            lambda checked: checked and self._set_interaction_mode("view")
        )
        mode_bar_layout.addWidget(self.view_mode_button)

        self.edit_boxes_mode_button = QPushButton("Ред. боксов", mode_bar)
        self.edit_boxes_mode_button.setCheckable(True)
        self.edit_boxes_mode_button.setProperty("segmented", "true")
        self.edit_boxes_mode_button.setToolTip("Редактировать и добавлять bbox.")
        self.edit_boxes_mode_button.toggled.connect(
            lambda checked: checked and self._set_interaction_mode("edit_boxes")
        )
        mode_bar_layout.addWidget(self.edit_boxes_mode_button)

        # Stub: kept for backward compatibility with tests; not shown in UI
        self.edit_masks_mode_button = QPushButton("Ред. маски", mode_bar)
        self.edit_masks_mode_button.setCheckable(True)
        self.edit_masks_mode_button.setEnabled(False)
        self.edit_masks_mode_button.setVisible(False)

        self.add_box_mode_button = QPushButton("+ Бокс", mode_bar)
        self.add_box_mode_button.setCheckable(True)
        self.add_box_mode_button.setProperty("segmented", "true")
        self.add_box_mode_button.setToolTip("Нарисуй прямоугольник чтобы добавить объект вручную")
        self.add_box_mode_button.toggled.connect(self._toggle_add_box_mode)
        mode_bar_layout.addWidget(self.add_box_mode_button)

        interaction_layout.addWidget(mode_bar)

        # --- Контекстный стек ---
        self.tools_stack = QStackedWidget(interaction_card)
        interaction_layout.addWidget(self.tools_stack)

        # Страница 0: Просмотр
        view_page = QWidget()
        view_layout = QVBoxLayout(view_page)
        view_layout.setContentsMargins(0, 4, 0, 0)
        view_layout.setSpacing(6)

        overlays_label = QLabel("Показ оверлеев", view_page)
        overlays_label.setObjectName("panelHint")
        view_layout.addWidget(overlays_label)

        toggle_bar = QFrame(view_page)
        toggle_bar.setObjectName("overlayToggleBar")
        toggle_bar_layout = QHBoxLayout(toggle_bar)
        toggle_bar_layout.setContentsMargins(6, 6, 6, 6)
        toggle_bar_layout.setSpacing(6)

        self.show_boxes_button = QPushButton("Боксы", toggle_bar)
        self.show_boxes_button.setCheckable(True)
        self.show_boxes_button.setChecked(self._show_boxes)
        self.show_boxes_button.setProperty("segmented", "true")
        self.show_boxes_button.toggled.connect(self._set_show_boxes)
        toggle_bar_layout.addWidget(self.show_boxes_button)

        self.show_masks_button = QPushButton("Маски", toggle_bar)
        self.show_masks_button.setCheckable(True)
        self.show_masks_button.setChecked(self._show_masks)
        self.show_masks_button.setProperty("segmented", "true")
        self.show_masks_button.toggled.connect(self._set_show_masks)
        toggle_bar_layout.addWidget(self.show_masks_button)

        view_layout.addWidget(toggle_bar)
        view_layout.addStretch()
        self.tools_stack.addWidget(view_page)  # index 0

        # Страница 1: Ред. боксов (ожидание)
        edit_page = QWidget()
        edit_layout = QVBoxLayout(edit_page)
        edit_layout.setContentsMargins(0, 4, 0, 0)
        edit_layout.setSpacing(6)

        self.add_box_button = QPushButton("+ Добавить бокс", edit_page)
        self.add_box_button.setObjectName("primaryToolButton")
        self.add_box_button.setToolTip("Нарисовать новый bbox мышью.")
        self.add_box_button.clicked.connect(self._start_add_box_mode)
        edit_layout.addWidget(self.add_box_button)

        secondary_row = QFrame(edit_page)
        secondary_row_layout = QHBoxLayout(secondary_row)
        secondary_row_layout.setContentsMargins(0, 0, 0, 0)
        secondary_row_layout.setSpacing(6)

        self.rerun_selected_button = QToolButton(secondary_row)
        self.rerun_selected_button.setObjectName("annotationToolButton")
        self.rerun_selected_button.setText("Инференс")
        self.rerun_selected_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.rerun_selected_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.rerun_selected_button.setToolTip("Повторить сегментацию выбранного сеянца.")
        self.rerun_selected_button.clicked.connect(self.classify_selected)
        secondary_row_layout.addWidget(self.rerun_selected_button)

        self.delete_annotation_button = QToolButton(secondary_row)
        self.delete_annotation_button.setObjectName("annotationDangerButton")
        self.delete_annotation_button.setText("Удалить")
        self.delete_annotation_button.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        self.delete_annotation_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.delete_annotation_button.setToolTip("Удалить выбранный bbox или часть.")
        self.delete_annotation_button.clicked.connect(self._delete_selected_annotation)
        secondary_row_layout.addWidget(self.delete_annotation_button)

        edit_layout.addWidget(secondary_row)
        edit_layout.addStretch()
        self.tools_stack.addWidget(edit_page)  # index 1

        # Страница 2: Рисование bbox
        drawing_page = QWidget()
        drawing_layout = QVBoxLayout(drawing_page)
        drawing_layout.setContentsMargins(0, 4, 0, 0)
        drawing_layout.setSpacing(8)

        self.drawing_hint_label = QLabel("✏  Нарисуйте бокс на холсте", drawing_page)
        self.drawing_hint_label.setObjectName("drawingHint")
        self.drawing_hint_label.setWordWrap(True)
        drawing_layout.addWidget(self.drawing_hint_label)

        confirm_row = QFrame(drawing_page)
        confirm_row_layout = QHBoxLayout(confirm_row)
        confirm_row_layout.setContentsMargins(0, 0, 0, 0)
        confirm_row_layout.setSpacing(6)

        self.confirm_box_button = QToolButton(confirm_row)
        self.confirm_box_button.setObjectName("annotationConfirmButton")
        self.confirm_box_button.setText("✓ Принять")
        self.confirm_box_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.confirm_box_button.setToolTip("Подтвердить положение нового bbox.")
        self.confirm_box_button.clicked.connect(self._confirm_pending_box)
        confirm_row_layout.addWidget(self.confirm_box_button)

        self.cancel_box_button = QToolButton(confirm_row)
        self.cancel_box_button.setObjectName("annotationToolButton")
        self.cancel_box_button.setText("✗ Отменить")
        self.cancel_box_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.cancel_box_button.setToolTip("Отменить добавление bbox.")
        self.cancel_box_button.clicked.connect(self._cancel_add_box_mode)
        confirm_row_layout.addWidget(self.cancel_box_button)

        drawing_layout.addWidget(confirm_row)
        drawing_layout.addStretch()
        self.tools_stack.addWidget(drawing_page)  # index 2

        left_layout.addWidget(interaction_card)
        splitter.addWidget(left_panel)

        self.canvas_host = QFrame(self)
        self.canvas_host.setObjectName("canvasHost")
        canvas_layout = QVBoxLayout(self.canvas_host)
        canvas_layout.setContentsMargins(12, 12, 12, 12)
        canvas_layout.setSpacing(10)

        self.canvas_toolbar = QFrame(self.canvas_host)
        self.canvas_toolbar.setObjectName("canvasToolbar")
        toolbar_layout = QHBoxLayout(self.canvas_toolbar)
        toolbar_layout.setContentsMargins(10, 8, 10, 8)
        toolbar_layout.setSpacing(8)

        self.canvas_page_label = QLabel("Нет открытого изображения", self.canvas_toolbar)
        self.canvas_page_label.setObjectName("panelSubTitle")
        toolbar_layout.addWidget(self.canvas_page_label, 1)

        self.zoom_status_chip = QLabel("100%", self.canvas_toolbar)
        self.zoom_status_chip.setObjectName("metricChip")
        toolbar_layout.addWidget(self.zoom_status_chip)

        self.scale_status_chip = QLabel("px", self.canvas_toolbar)
        self.scale_status_chip.setObjectName("metricChip")
        toolbar_layout.addWidget(self.scale_status_chip)

        self.measure_toggle_button = QPushButton("Линейка", self.canvas_toolbar)
        self.measure_toggle_button.setObjectName("toolToggleButton")
        self.measure_toggle_button.setCheckable(True)
        self.measure_toggle_button.toggled.connect(self._set_measure_mode)
        toolbar_layout.addWidget(self.measure_toggle_button)

        self.calibration_button = QPushButton("Калибровка", self.canvas_toolbar)
        self.calibration_button.setProperty("variant", "secondary")
        self.calibration_button.clicked.connect(self._start_calibration)
        toolbar_layout.addWidget(self.calibration_button)

        self.calibration_settings_button = QPushButton("Коэфф.", self.canvas_toolbar)
        self.calibration_settings_button.setProperty("variant", "secondary")
        self.calibration_settings_button.clicked.connect(
            self._open_calibration_settings
        )
        toolbar_layout.addWidget(self.calibration_settings_button)

        canvas_layout.addWidget(self.canvas_toolbar)

        self.canvas_stack = QStackedLayout()

        self.graphics_scene = QGraphicsScene(self)
        self.graphics_scene.selectionChanged.connect(self._update_annotation_controls)
        self.graphics_view = CanvasGraphicsView(self.graphics_scene, self)
        self.graphics_view.box_drawn.connect(self._on_box_drawn)
        self.graphics_view.draw_mode_cancelled.connect(
            lambda: self.add_box_mode_button.setChecked(False)
        )
        self.graphics_view.setObjectName("centralView")
        self.graphics_view.setFrameShape(QFrame.NoFrame)
        self.graphics_view.setRenderHint(QPainter.Antialiasing, True)
        self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.graphics_view.setAlignment(Qt.AlignCenter)
        self.graphics_view.viewport().installEventFilter(self)

        self.empty_state = self._build_empty_state()
        self.canvas_stack.addWidget(self.empty_state)
        self.canvas_stack.addWidget(self.graphics_view)
        canvas_layout.addLayout(self.canvas_stack, 1)
        splitter.addWidget(self.canvas_host)

        splitter.setSizes([SPLITTER_SIZES[0], SPLITTER_SIZES[1] + SPLITTER_SIZES[2]])
        right_panel = QFrame(self)
        right_panel.setObjectName("panelCard")
        right_panel.setMinimumWidth(PANEL_LAYERS_MIN_WIDTH)
        right_panel.setMaximumWidth(PANEL_LAYERS_MAX_WIDTH)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)

        self.sidebar_page_label = QLabel("Нет выбранной страницы", right_panel)
        self.sidebar_page_label.setObjectName("panelSubTitle")
        self.sidebar_page_label.setWordWrap(True)
        right_layout.addWidget(self.sidebar_page_label)

        self.right_tabs = QTabWidget(right_panel)
        self.right_tabs.setObjectName("rightTabs")
        self.right_tabs.setDocumentMode(True)
        self.tree_widget = LayerTreeWidget()
        self.tree_widget.itemSelectionChanged.connect(
            self._on_tree_selection_changed
        )
        self.statistics_panel = StatisticsPanel(self)

        self.right_tabs.addTab(self.tree_widget, "Слои")
        self.right_tabs.addTab(self.statistics_panel, "Статистика")
        right_layout.addWidget(self.right_tabs, 1)

        separator = QFrame(right_panel)
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        right_layout.addWidget(separator)

        self.detail_panel = SeedlingDetailPanel(right_panel)
        self.detail_panel.navigate.connect(self._navigate_seedling)
        right_layout.addWidget(self.detail_panel)

        splitter.addWidget(right_panel)
        splitter.setSizes(SPLITTER_SIZES)
        self._set_canvas_empty(True)
        self._update_canvas_status()
        self._update_annotation_controls()

    def _build_empty_state(self) -> QWidget:
        """Создаёт заглушку, показываемую на холсте до открытия первого файла."""
        empty_state = QFrame(self)
        empty_state.setObjectName("emptyState")
        layout = QVBoxLayout(empty_state)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(12)
        layout.addStretch()

        title = QLabel("Откройте изображение или PDF", empty_state)
        title.setObjectName("emptyStateTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        hint = QLabel(
            "После загрузки можно найти растения, сегментировать части и "
            "сохранить PDF-отчет.",
            empty_state,
        )
        hint.setObjectName("emptyStateHint")
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        layout.addWidget(hint)

        shortcuts = QLabel(
            "Ctrl+O - открыть, Ctrl+Shift+F - анализ всех страниц",
            empty_state,
        )
        shortcuts.setObjectName("emptyStateHint")
        shortcuts.setAlignment(Qt.AlignCenter)
        layout.addWidget(shortcuts)

        open_button = QPushButton("Открыть файлы", empty_state)
        open_button.setObjectName("emptyOpenButton")
        open_button.clicked.connect(self.open_image)
        layout.addWidget(open_button, alignment=Qt.AlignCenter)
        layout.addStretch()
        return empty_state

    def _set_canvas_empty(self, is_empty: bool) -> None:
        """Переключает центральную область между пустым состоянием и холстом."""
        self.canvas_stack.setCurrentWidget(
            self.empty_state if is_empty else self.graphics_view
        )

    def _update_project_summary(self) -> None:
        """Обновляет краткую сводку проекта по числу страниц и исходных файлов."""
        pages_count = len(self.image_storage.images)
        if pages_count == 0:
            self.project_count_chip.setText("0 страниц")
            return
        sources_count = len({path for path in self.image_storage.source_files if path})
        self.project_count_chip.setText(
            f"{pages_count} стр. / {max(1, sources_count)} файл."
        )

    def _page_position_for_source(self, page_index: int) -> tuple[int, int]:
        """Возвращает локальный номер страницы и их общее число внутри одного источника."""
        source = self._source_file(page_index)
        if not source:
            return page_index + 1, len(self.image_storage.images)

        positions = [
            idx for idx, value in enumerate(self.image_storage.source_files) if value == source
        ]
        if not positions:
            return page_index + 1, len(self.image_storage.images)
        return positions.index(page_index) + 1, len(positions)

    def _update_canvas_status(self) -> None:
        """Обновляет подписи текущей страницы, масштаба и калибровки на панели статуса."""
        if not self.image_storage.images:
            self.canvas_page_label.setText("Нет открытого изображения")
            self.zoom_status_chip.setText("100%")
            self.scale_status_chip.setText("px")
            return

        source = self._source_file(self._active_image_index)
        source_name = Path(source).name if source else f"Страница {self._active_image_index + 1}"
        if source.lower().endswith(".pdf"):
            position, total = self._page_position_for_source(self._active_image_index)
            page_text = f"{source_name} · страница {position}/{total}"
        else:
            page_text = source_name

        self.canvas_page_label.setText(page_text)
        self.zoom_status_chip.setText(f"{int(round(self.zoom_factor * 100))}%")
        if self.pixels_per_mm > 0:
            self.scale_status_chip.setText(f"{self.pixels_per_mm:.2f} px/мм")
        else:
            self.scale_status_chip.setText("Без масштаба")

    def _normalize_source_key(self, source_file: str | Path | None) -> str | None:
        """Приводит путь источника к стабильному ключу для хранения калибровок."""
        if not source_file:
            return None
        path = Path(str(source_file).strip()).expanduser()
        try:
            return str(path.resolve())
        except OSError:
            return str(path)

    def _load_calibrations_payload(self) -> dict[str, float]:
        """Загружает сохранённые коэффициенты калибровки из `QSettings`."""
        raw_value = self._settings.value("file_calibrations", "{}", type=str) or "{}"
        try:
            payload = json.loads(raw_value)
        except json.JSONDecodeError:
            return {}
        result: dict[str, float] = {}
        if not isinstance(payload, dict):
            return result
        for key, value in payload.items():
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                result[str(key)] = parsed
        return result

    def _save_calibrations_payload(self, payload: dict[str, float]) -> None:
        """Сохраняет словарь калибровок в `QSettings` в виде JSON."""
        self._settings.setValue(
            "file_calibrations",
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )
        self._settings.sync()

    def _current_source_file(self, index: int | None = None) -> str:
        """Возвращает путь активного исходного файла или общий путь проекта."""
        source_index = self._active_image_index if index is None else int(index)
        if 0 <= source_index < len(self.image_storage.source_files):
            source = str(self.image_storage.source_files[source_index]).strip()
            if source:
                return source
        return str(self.image_storage.file_path or "")

    def _save_calibration_for_current_source(self, pixels_per_mm: float) -> None:
        """Сохраняет калибровку для текущего файла и обновляет состояние окна."""
        self.pixels_per_mm = max(float(pixels_per_mm), 0.0)
        self.app_state.pixels_per_mm = self.pixels_per_mm
        self._settings.setValue("pixels_per_mm", self.pixels_per_mm)

        source_key = self._normalize_source_key(self._current_source_file())
        if source_key is not None:
            calibrations = self._load_calibrations_payload()
            if self.pixels_per_mm > 0:
                calibrations[source_key] = self.pixels_per_mm
            else:
                calibrations.pop(source_key, None)
            self._save_calibrations_payload(calibrations)
        else:
            self._settings.sync()

        self._update_canvas_status()

    def _restore_calibration_for_index(self, index: int | None = None) -> None:
        """Восстанавливает калибровку для выбранной страницы по пути исходного файла."""
        pixels_per_mm = CALIBRATION_PIXELS_PER_MM_DEFAULT
        source_key = self._normalize_source_key(self._current_source_file(index))
        if source_key is not None:
            pixels_per_mm = self._load_calibrations_payload().get(
                source_key,
                CALIBRATION_PIXELS_PER_MM_DEFAULT,
            )
        self.pixels_per_mm = max(float(pixels_per_mm), 0.0)
        self.app_state.pixels_per_mm = self.pixels_per_mm
        self._update_canvas_status()

    def _set_measure_mode(self, enabled: bool) -> None:
        """Включает или выключает режим измерения на изображении."""
        if enabled and not self.image_storage.images:
            self._measure_mode = False
            if self.measure_toggle_button.isChecked():
                self.measure_toggle_button.blockSignals(True)
                self.measure_toggle_button.setChecked(False)
                self.measure_toggle_button.blockSignals(False)
            self._show_info("Нет данных", "Сначала откройте изображение или PDF.")
            return

        self._measure_mode = bool(enabled)
        if self.measure_toggle_button.isChecked() != self._measure_mode:
            self.measure_toggle_button.blockSignals(True)
            self.measure_toggle_button.setChecked(self._measure_mode)
            self.measure_toggle_button.blockSignals(False)

        if self._measure_mode:
            self.graphics_view.viewport().setCursor(Qt.CrossCursor)
            if not self._calibration_pending:
                self.statusBar().showMessage(
                    "Линейка: отметьте две точки на изображении.",
                    3000,
                )
        else:
            self.graphics_view.viewport().unsetCursor()
            if not self._calibration_pending:
                self._reset_measure_state(clear_items=True)

    def toggle_measure_mode(self) -> None:
        """Переключает режим измерения между активным и неактивным состоянием."""
        self._set_measure_mode(not self._measure_mode)

    def _open_calibration_settings(self) -> None:
        """Открывает диалог ручного ввода коэффициента калибровки для текущего файла."""
        if not self.image_storage.images:
            self._show_info("Нет данных", "Сначала откройте изображение или PDF.")
            return

        current_value = self.pixels_per_mm if self.pixels_per_mm > 0 else 10.0
        value, ok = QInputDialog.getDouble(
            self,
            "Калибровка",
            "Введите коэффициент px/мм.\n0 = сброс для текущего файла.",
            current_value,
            0.0,
            1_000_000.0,
            4,
        )
        if not ok:
            return

        self._save_calibration_for_current_source(value)
        self._refresh_current_view()
        if value > 0:
            self.statusBar().showMessage(
                f"Коэффициент сохранен: {value:.4f} px/мм",
                3000,
            )
        else:
            self.statusBar().showMessage(
                "Калибровка для текущего файла сброшена.",
                3000,
            )

    def _open_app_settings(self) -> None:
        """Open inference and performance settings."""
        dialog = AppSettingsDialog(
            current_device=self._analysis_device,
            current_batch_size=self._analysis_batch_size,
            current_auto_save_enabled=self._auto_save_enabled,
            current_report_output_dir=self._report_output_dir,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return

        selected = dialog.selected_settings()
        self._analysis_device = selected.analysis_device
        self._analysis_batch_size = selected.analysis_batch_size
        self._auto_save_enabled = selected.auto_save_enabled
        self._report_output_dir = selected.report_output_dir
        self._settings.setValue("analysis/yolo_device", self._analysis_device)
        self._settings.setValue("analysis/yolo_batch_size", self._analysis_batch_size)
        self._settings.setValue("app/auto_save_enabled", self._auto_save_enabled)
        self._settings.setValue("report/output_dir", self._report_output_dir)
        self._settings.sync()
        self._apply_inference_settings()

        self.detect_model = None
        self.classify_model = None
        self.statusBar().showMessage(
            f"Настройки применены: устройство {self._analysis_device}, batch {self._analysis_batch_size}",
            5000,
        )


    def _start_calibration(self) -> None:
        """Запускает интерактивную калибровку по измеренному отрезку на изображении."""
        if not self.image_storage.images:
            self._show_info("Калибровка", "Сначала откройте изображение или PDF.")
            return
        self._calibration_pending = True
        self._reset_measure_state(clear_items=True)
        self._set_measure_mode(True)
        self.statusBar().showMessage(
            (
                "Калибровка: отметьте две точки эталонного отрезка, "
                "после второго клика введите длину в мм."
            ),
            6000,
        )

    def _reset_measure_state(self, *, clear_items: bool) -> None:
        """Сбрасывает временные данные измерения и при необходимости удаляет графику."""
        self._measure_start_scene_pos = None
        for attr_name in ("_measure_line_item", "_measure_text_item"):
            item = getattr(self, attr_name, None)
            if clear_items and item is not None:
                try:
                    scene = item.scene()
                    if scene is not None:
                        scene.removeItem(item)
                except RuntimeError:
                    pass
            setattr(self, attr_name, None)

    def _image_scene_rect(self) -> QRectF:
        """Возвращает прямоугольник изображения на сцене или пустую область."""
        if self._pixmap_item is None:
            return QRectF()
        return self._pixmap_item.sceneBoundingRect()

    def _clamp_scene_pos_to_image(self, scene_pos: QPointF) -> QPointF | None:
        """Ограничивает точку сцены границами отображаемого изображения."""
        rect = self._image_scene_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return None
        return QPointF(
            min(max(scene_pos.x(), rect.left()), rect.right()),
            min(max(scene_pos.y(), rect.top()), rect.bottom()),
        )

    def _start_manual_measure(self, scene_pos: QPointF) -> None:
        """Создаёт линию измерения и запоминает первую точку на сцене."""
        self._reset_measure_state(clear_items=True)
        self._measure_start_scene_pos = scene_pos
        pen = QPen(QColor(46, 226, 201))
        pen.setWidth(2)
        self._measure_line_item = self.graphics_scene.addLine(
            scene_pos.x(),
            scene_pos.y(),
            scene_pos.x(),
            scene_pos.y(),
            pen,
        )
        self._measure_line_item.setZValue(50)
        self._measure_text_item = self.graphics_scene.addText("")
        self._measure_text_item.setDefaultTextColor(QColor(46, 226, 201))
        self._measure_text_item.setFlag(
            QGraphicsItem.ItemIgnoresTransformations,
            True,
        )
        self._measure_text_item.setZValue(51)
        self._update_manual_measure(scene_pos)

    def _update_manual_measure(self, scene_pos: QPointF) -> None:
        """Обновляет линию измерения и подпись с текущей длиной отрезка."""
        start = self._measure_start_scene_pos
        if start is None:
            return
        if self._measure_line_item is not None:
            self._measure_line_item.setLine(
                start.x(),
                start.y(),
                scene_pos.x(),
                scene_pos.y(),
            )

        dx = scene_pos.x() - start.x()
        dy = scene_pos.y() - start.y()
        diagonal_px = float((dx ** 2 + dy ** 2) ** 0.5)
        label = f"{diagonal_px:.2f}px"
        if self.pixels_per_mm > 0:
            label += f" / {diagonal_px / self.pixels_per_mm:.2f} мм"

        if self._measure_text_item is not None:
            mid_x = (start.x() + scene_pos.x()) / 2.0
            mid_y = (start.y() + scene_pos.y()) / 2.0
            self._measure_text_item.setPlainText(label)
            self._measure_text_item.setPos(mid_x + 6.0, mid_y + 6.0)

    def _apply_calibration_from_measurement(self, diagonal_px: float) -> None:
        """Переводит измеренный отрезок в коэффициент пиксели-на-миллиметр."""
        if diagonal_px <= 0:
            self.statusBar().showMessage(
                "Калибровка: отрезок нулевой длины, повторите измерение.",
                3500,
            )
            return

        mm_value, ok = QInputDialog.getDouble(
            self,
            "Калибровка",
            (
                f"Измерено: {diagonal_px:.2f} px.\n"
                "Введите реальную длину отрезка в миллиметрах:"
            ),
            10.0,
            0.0001,
            1_000_000.0,
            4,
        )
        if not ok:
            self.statusBar().showMessage(
                "Калибровка отменена пользователем.",
                3000,
            )
            return
        if mm_value <= 0:
            self._show_error(
                "Калибровка",
                "Длина в миллиметрах должна быть больше нуля.",
            )
            return

        self._save_calibration_for_current_source(diagonal_px / float(mm_value))
        self.statusBar().showMessage(
            f"Калибровка применена: {self.pixels_per_mm:.4f} px/мм",
            4500,
        )
        self._refresh_current_view()

    def _finish_manual_measure(self, scene_pos: QPointF) -> None:
        """Завершает измерение и либо применяет калибровку, либо показывает результат."""
        start = self._measure_start_scene_pos
        if start is None:
            return

        self._update_manual_measure(scene_pos)
        width_px = int(round(abs(scene_pos.x() - start.x())))
        height_px = int(round(abs(scene_pos.y() - start.y())))
        diagonal_px = float((width_px ** 2 + height_px ** 2) ** 0.5)

        if self._calibration_pending:
            self._calibration_pending = False
            self._measure_start_scene_pos = None
            self._apply_calibration_from_measurement(diagonal_px)
            return

        self._measure_start_scene_pos = None
        if self.pixels_per_mm > 0:
            diagonal_mm = diagonal_px / self.pixels_per_mm
            self.statusBar().showMessage(
                f"Измерение: {diagonal_px:.2f}px / {diagonal_mm:.2f} мм",
                4000,
            )
        else:
            self.statusBar().showMessage(
                f"Измерение: {diagonal_px:.2f}px",
                4000,
            )

    def eventFilter(self, watched, event):
        """Обрабатывает колесо мыши и события линейки в области просмотра изображения."""
        if self._handle_frameless_window_event(watched, event):
            return True

        if watched is self.graphics_view.viewport():
            if event.type() == QEvent.Wheel and event.modifiers() & Qt.ControlModifier:
                if event.angleDelta().y() > 0:
                    self.zoom_in(anchor_view_pos=event.pos())
                elif event.angleDelta().y() < 0:
                    self.zoom_out(anchor_view_pos=event.pos())
                return True

            if self._box_draw_active:
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    scene_pos = self._clamp_scene_pos_to_image(
                        self.graphics_view.mapToScene(event.pos())
                    )
                    if scene_pos is None:
                        return True
                    self._box_draw_start = scene_pos
                    self._box_draw_item = self.graphics_scene.addRect(
                        QRectF(scene_pos, scene_pos),
                        QPen(QColor(255, 255, 255), 2, Qt.DashLine),
                        QBrush(Qt.transparent),
                    )
                    self._box_draw_item.setZValue(5)
                    return True

                if (
                    event.type() == QEvent.MouseMove
                    and self._box_draw_start is not None
                    and self._box_draw_item is not None
                ):
                    scene_pos = self._clamp_scene_pos_to_image(
                        self.graphics_view.mapToScene(event.pos())
                    )
                    if scene_pos is not None:
                        self._box_draw_item.setRect(
                            QRectF(self._box_draw_start, scene_pos).normalized()
                        )
                    return True

                if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                    rect = (
                        self._box_draw_item.rect().normalized()
                        if self._box_draw_item is not None
                        else QRectF()
                    )
                    self._finish_add_box(rect)
                    return True

                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
                    self._cancel_add_box_mode()
                    return True

            if self._measure_mode:
                if event.type() == QEvent.MouseButtonPress:
                    if event.button() == Qt.RightButton:
                        was_calibration = self._calibration_pending
                        self._calibration_pending = False
                        self._reset_measure_state(clear_items=True)
                        self._set_measure_mode(False)
                        self.statusBar().showMessage(
                            "Калибровка отменена."
                            if was_calibration
                            else "Линейка: измерение отменено.",
                            2000,
                        )
                        return True

                    if event.button() == Qt.LeftButton:
                        raw_scene_pos = self.graphics_view.mapToScene(event.pos())
                        scene_pos = self._clamp_scene_pos_to_image(raw_scene_pos)
                        if scene_pos is None:
                            return True
                        if self._measure_start_scene_pos is None:
                            self._start_manual_measure(scene_pos)
                        else:
                            self._finish_manual_measure(scene_pos)
                        return True

                if (
                    event.type() == QEvent.MouseMove
                    and self._measure_start_scene_pos is not None
                ):
                    raw_scene_pos = self.graphics_view.mapToScene(event.pos())
                    scene_pos = self._clamp_scene_pos_to_image(raw_scene_pos)
                    if scene_pos is not None:
                        self._update_manual_measure(scene_pos)
                    return True

        return super().eventFilter(watched, event)

    def keyPressEvent(self, event) -> None:
        """Отменяет измерение по `Esc` и передаёт остальные клавиши стандартной обработке."""
        if event.key() == Qt.Key_Delete and self._interaction_mode == "edit_boxes":
            self._delete_selected_annotation()
            event.accept()
            return
        if event.key() == Qt.Key_Escape and self._box_draw_active:
            self._cancel_add_box_mode()
            event.accept()
            return
        if event.key() == Qt.Key_Escape and (
            self._measure_mode or self._measure_start_scene_pos is not None
        ):
            self._calibration_pending = False
            self._reset_measure_state(clear_items=True)
            self._set_measure_mode(False)
            self.statusBar().showMessage("Линейка: измерение отменено.", 2000)
            event.accept()
            return
        super().keyPressEvent(event)

    def changeEvent(self, event) -> None:
        """Обновляет кнопки окна при смене состояния развёрнуто/обычно."""

        if event.type() == QEvent.WindowStateChange:
            self._update_window_control_buttons()
        super().changeEvent(event)

    def _refresh_current_view(self) -> None:
        """Перерисовывает текущее изображение или кроп без смены выбранного контекста."""
        if not self.image_storage.images:
            return
        self._restore_display(preserve_view=True)

    def _set_show_boxes(self, visible: bool) -> None:
        """Включает или выключает отображение bbox на текущем изображении."""
        self._show_boxes = bool(visible)
        self._refresh_current_view()

    def _set_show_masks(self, visible: bool) -> None:
        """Включает или выключает отображение масок частей на текущем изображении."""
        self._show_masks = bool(visible)
        self._refresh_current_view()

    def _set_interaction_mode(self, mode: str) -> None:
        """Переключает режим взаимодействия с графическими элементами на сцене."""
        if mode not in {"view", "edit_boxes", "edit_masks"}:
            return
        self._interaction_mode = mode
        self._box_draw_active = False
        self._box_draw_start = None
        if self._box_draw_item is not None:
            try:
                scene = self._box_draw_item.scene()
                if scene is not None:
                    scene.removeItem(self._box_draw_item)
            except RuntimeError:
                pass
            self._box_draw_item = None
        if mode != "edit_boxes" and hasattr(self, "_clear_pending_box"):
            self._clear_pending_box()
        for button, button_mode in (
            (self.view_mode_button, "view"),
            (self.edit_boxes_mode_button, "edit_boxes"),
        ):
            button.blockSignals(True)
            button.setChecked(mode == button_mode)
            button.blockSignals(False)
        self._apply_interaction_mode_to_rect_items()

    def _apply_interaction_mode_to_rect_items(self) -> None:
        """Применяет текущий режим редактирования ко всем рамкам на сцене."""
        editable = self._interaction_mode == "edit_boxes"
        for item in self.rect_items:
            item.setEditable(editable and bool(getattr(item, "_manual_edit_enabled", True)))
        self._update_annotation_controls()

    def _update_annotation_controls(self) -> None:
        """Updates annotation toolbar availability for the current view and mode."""
        if not hasattr(self, "add_box_button"):
            return
        has_image = self._pixmap_item is not None
        editing_boxes = self._interaction_mode == "edit_boxes"
        selected_payload = self.app_state.selected_item or {}
        can_rerun = self._current_crop_context() is not None or selected_payload.get("type") == "seeding"
        has_pending_box = self._pending_box_item is not None
        drawing_active = self._box_draw_active or has_pending_box

        # Переключение страницы стека
        if not editing_boxes:
            self.tools_stack.setCurrentIndex(0)
        elif drawing_active:
            self.tools_stack.setCurrentIndex(2)
        else:
            self.tools_stack.setCurrentIndex(1)

        # Состояние кнопок
        self.add_box_button.setEnabled(has_image and editing_boxes and not drawing_active)
        self.rerun_selected_button.setEnabled(has_image and can_rerun)
        self.delete_annotation_button.setEnabled(
            has_image and not drawing_active and self._selected_annotation_object() is not None
        )
        self.confirm_box_button.setEnabled(has_pending_box)
        self.cancel_box_button.setEnabled(True)

    def _toggle_add_box_mode(self, enabled: bool) -> None:
        """Включает/выключает режим рисования боксов на канвасе."""
        self.graphics_view.set_draw_mode(enabled)
        if enabled:
            self.view_mode_button.setChecked(False)
            self.edit_boxes_mode_button.setChecked(False)
            self.edit_masks_mode_button.setChecked(False)
        else:
            self.view_mode_button.setChecked(True)

    def _on_box_drawn(self, scene_rect) -> None:
        """Обрабатывает нарисованный прямоугольник — показывает диалог выбора класса."""
        from seeding.ui.add_box_dialog import AddBoxDialog

        self.add_box_mode_button.setChecked(False)

        page_idx = self._active_image_index
        if not self.image_storage.images or page_idx >= len(self.image_storage.images):
            return

        offset_x = offset_y = 0.0
        if self._pixmap_item is not None:
            p = self._pixmap_item.pos()
            offset_x, offset_y = p.x(), p.y()

        page_img = self.image_storage.images[page_idx]
        if not isinstance(page_img, np.ndarray):
            logger.warning("_on_box_drawn: page image is not ndarray, skipping")
            return
        img_h, img_w = page_img.shape[:2]

        x1 = max(0, int(scene_rect.left() - offset_x))
        y1 = max(0, int(scene_rect.top() - offset_y))
        x2 = min(img_w, int(scene_rect.right() - offset_x))
        y2 = min(img_h, int(scene_rect.bottom() - offset_y))

        if x2 <= x1 or y2 <= y1:
            return

        dlg = AddBoxDialog(self)
        if dlg.exec_() != dlg.Accepted:
            return
        class_name = dlg.selected_class()
        if class_name is None:
            return

        self._create_manual_object(page_idx, class_name, (x1, y1, x2, y2))

    def _create_manual_object(
        self,
        page_idx: int,
        class_name: str,
        bbox: tuple[int, int, int, int],
    ) -> None:
        """Создаёт ObjectImage или AllClassImage по нарисованному боксу."""
        from seeding.mask_refiner import bitmap_to_polygon, refine_mask_bitmap

        page_img = self.image_storage.images[page_idx]
        if not isinstance(page_img, np.ndarray):
            return

        x1, y1, x2, y2 = bbox
        crop = page_img[y1:y2, x1:x2].copy()

        h, w = crop.shape[:2]
        coarse = np.full((h, w), 255, dtype=np.uint8)
        mask_bitmap = refine_mask_bitmap(crop, coarse)
        mask_polygon = bitmap_to_polygon(mask_bitmap) if mask_bitmap is not None else None

        if class_name == "seeding":
            obj = ObjectImage(
                class_name=class_name,
                confidence=1.0,
                image=[crop],
                image_all_class=[],
                bbox=bbox,
                manual=True,
            )
            if self.image_storage.class_object_image is None:
                self.image_storage.class_object_image = [
                    [] for _ in self.image_storage.images
                ]
            while len(self.image_storage.class_object_image) <= page_idx:
                self.image_storage.class_object_image.append([])
            self.image_storage.class_object_image[page_idx].append(obj)
        else:
            page_objects = []
            if (self.image_storage.class_object_image
                    and page_idx < len(self.image_storage.class_object_image)):
                page_objects = self.image_storage.class_object_image[page_idx]

            target_obj = None
            selected = self.app_state.selected_item or {}
            if selected.get("type") in {"seeding", "class"}:
                seed_idx = int(selected.get("seeding_index", selected.get("index", 0)))
                if seed_idx < len(page_objects):
                    target_obj = page_objects[seed_idx]

            if target_obj is None and page_objects:
                target_obj = page_objects[-1]

            if target_obj is None:
                QMessageBox.information(
                    self, "Нет сеянца",
                    "Сначала выбери сеянец в дереве, к которому добавить эту часть."
                )
                return

            part = AllClassImage(
                class_name=class_name,
                confidence=1.0,
                image=crop,
                bbox=bbox,
                mask_polygon=mask_polygon,
                mask_bitmap=mask_bitmap,
                manual=True,
            )
            if target_obj.image_all_class is None:
                target_obj.image_all_class = []
            target_obj.image_all_class.append(part)

        self._refresh_tree()
        self.display_image_with_boxes(page_idx)

    def _build_toolbar(self) -> None:
        """Создаёт верхний toolbar с основными действиями приложения."""
        toolbar = QToolBar("Main", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.main_toolbar = toolbar
        self.addToolBar(toolbar)

        self.action_open = self._create_action(
            "action_open.svg",
            "Открыть",
            self.open_image,
            shortcut=QKeySequence.Open,
            fallback_standard_icon=QStyle.SP_DialogOpenButton,
        )
        self.action_add = self._create_action(
            "action_add.svg",
            "Добавить",
            self.add_files,
            shortcut="Ctrl+Shift+O",
            fallback_standard_icon=QStyle.SP_FileIcon,
        )
        self.action_open_session = self._create_action(
            "action_open.svg",
            "Открыть сессию",
            self._open_session_dialog,
            shortcut="Ctrl+Alt+O",
            fallback_standard_icon=QStyle.SP_FileDialogDetailedView,
        )
        self.action_find = self._create_action(
            "action_detect.svg",
            "Найти",
            self.find_seedlings,
            shortcut="Ctrl+F",
            fallback_standard_icon=QStyle.SP_MediaPlay,
        )
        self.btn_find_all = self._create_split_button(
            "action_detect_all.svg",
            "Найти на всех",
            self.find_all_seedlings_new_only,
            [
                ("Найти только на новых страницах", self.find_all_seedlings_new_only),
                ("Найти на всех страницах заново", self.find_all_seedlings),
            ],
            fallback_standard_icon=QStyle.SP_BrowserReload,
        )
        self.btn_classify = self._create_split_button(
            "action_classify.svg",
            "Сегментировать",
            self.classify_new_only,
            [
                ("Сегментировать только новые", self.classify_new_only),
                ("Сегментировать все заново", self.classify),
            ],
            fallback_standard_icon=QStyle.SP_FileDialogDetailedView,
        )
        self.action_rotate = self._create_action(
            "action_rotate.svg",
            "Повернуть",
            self.rotate_image,
            shortcut="Ctrl+R",
            fallback_standard_icon=QStyle.SP_BrowserReload,
        )
        self.action_save = self._create_action(
            "action_report.svg",
            "Сохранить",
            self._on_save,
            shortcut="Ctrl+S",
            fallback_standard_icon=QStyle.SP_DialogSaveButton,
        )
        self.action_report = self._create_action(
            "action_report.svg",
            "Отчёт",
            self.create_report,
            shortcut="Ctrl+P",
            fallback_standard_icon=QStyle.SP_FileDialogContentsView,
        )
        self.action_settings = self._create_action(
            "action_fit.svg",
            "Настройки",
            self._open_app_settings,
            shortcut="Ctrl+,",
            fallback_standard_icon=QStyle.SP_FileDialogDetailedView,
        )
        self.action_logout = self._create_action(
            "action_close.svg",
            "Выйти из аккаунта",
            self._request_logout,
            fallback_standard_icon=QStyle.SP_DialogCloseButton,
        )
        self.action_zoom_in = self._create_action(
            "action_zoom_in.svg",
            "Приблизить",
            self.zoom_in,
            shortcut="Ctrl++",
            fallback_standard_icon=QStyle.SP_ArrowUp,
        )
        self.action_zoom_out = self._create_action(
            "action_zoom_out.svg",
            "Отдалить",
            self.zoom_out,
            shortcut="Ctrl+-",
            fallback_standard_icon=QStyle.SP_ArrowDown,
        )
        self.action_fit = self._create_action(
            "action_fit.svg",
            "Вписать",
            self.fit_to_window,
            shortcut="Ctrl+0",
            fallback_standard_icon=QStyle.SP_DesktopIcon,
        )

        toolbar.addAction(self.action_open)
        toolbar.addAction(self.action_add)
        toolbar.addSeparator()
        toolbar.addAction(self.action_find)
        toolbar.addWidget(self.btn_find_all)
        toolbar.addWidget(self.btn_classify)
        toolbar.addAction(self.action_rotate)
        toolbar.addSeparator()
        toolbar.addAction(self.action_save)
        toolbar.addAction(self.action_report)
        toolbar.addAction(self.action_settings)
        toolbar.addSeparator()
        toolbar.addAction(self.action_zoom_in)
        toolbar.addAction(self.action_zoom_out)
        toolbar.addAction(self.action_fit)
        toolbar.addSeparator()

        self.window_drag_handle = QLabel("Seeding", toolbar)
        self.window_drag_handle.setObjectName("toolbarBrand")
        self.window_drag_handle.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.window_drag_handle.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
        self.window_drag_handle.installEventFilter(self)
        toolbar.addWidget(self.window_drag_handle)

    def _build_menu(self) -> None:
        """Создаёт верхнее меню и раскладывает по нему основные действия окна."""
        self.menuBar().setNativeMenuBar(False)
        file_menu = self.menuBar().addMenu("Файл")
        file_menu.addAction(self.action_open)
        file_menu.addAction(self.action_add)
        file_menu.addAction(self.action_open_session)
        file_menu.addSeparator()
        file_menu.addAction(self.action_report)
        file_menu.addSeparator()
        file_menu.addAction(self.action_logout)

        analysis_menu = self.menuBar().addMenu("Анализ")
        analysis_menu.addAction(self.action_find)
        find_all_menu = analysis_menu.addMenu("Найти на всех")
        find_all_menu.addAction("Найти только на новых страницах", self.find_all_seedlings_new_only)
        find_all_menu.addAction("Найти на всех страницах заново", self.find_all_seedlings)
        classify_menu = analysis_menu.addMenu("Сегментировать")
        classify_menu.addAction("Сегментировать выбранный заново", self.classify_selected)
        classify_menu.addAction("Сегментировать только новые", self.classify_new_only)
        classify_menu.addAction("Сегментировать все заново", self.classify)
        analysis_menu.addAction(self.action_rotate)

        view_menu = self.menuBar().addMenu("Вид")
        view_menu.addAction(self.action_zoom_in)
        view_menu.addAction(self.action_zoom_out)
        view_menu.addAction(self.action_fit)

        settings_menu = self.menuBar().addMenu("Настройки")
        settings_menu.addAction(self.action_settings)

        self.window_controls_widget = QWidget(self.menuBar())
        controls_layout = QHBoxLayout(self.window_controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        self.window_minimize_button = self._create_window_control_button(
            QStyle.SP_TitleBarMinButton,
            "Свернуть",
            self.showMinimized,
        )
        controls_layout.addWidget(self.window_minimize_button)

        self.window_maximize_button = self._create_window_control_button(
            QStyle.SP_TitleBarMaxButton,
            "Развернуть",
            self._toggle_maximized,
        )
        controls_layout.addWidget(self.window_maximize_button)

        self.window_close_button = self._create_window_control_button(
            QStyle.SP_TitleBarCloseButton,
            "Закрыть",
            self.close,
            role="close",
        )
        controls_layout.addWidget(self.window_close_button)
        self.menuBar().setCornerWidget(self.window_controls_widget, Qt.TopRightCorner)

    def _create_action(
        self,
        icon_name: str,
        text: str,
        handler,
        *,
        shortcut: str | QKeySequence | None = None,
        fallback_standard_icon: QStyle.StandardPixmap | None = None,
    ) -> QAction:
        """Создаёт действие меню или toolbar с иконкой, хоткеем и обработчиком."""
        action = QAction(
            self.icon_manager.get_icon(
                icon_name,
                fallback_standard_icon=fallback_standard_icon,
            ),
            text,
            self,
        )
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(handler)
        return action

    def _create_split_button(
        self,
        icon_name: str,
        text: str,
        default_handler,
        menu_items: list[tuple[str, object]],
        *,
        fallback_standard_icon: QStyle.StandardPixmap | None = None,
    ) -> QToolButton:
        """Создаёт кнопку со стрелкой: клик = default_handler, стрелка = меню вариантов."""
        btn = QToolButton(self.main_toolbar)
        btn.setIcon(
            self.icon_manager.get_icon(
                icon_name, fallback_standard_icon=fallback_standard_icon
            )
        )
        btn.setText(text)
        btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        btn.setPopupMode(QToolButton.MenuButtonPopup)

        menu = QMenu(btn)
        for item_text, handler in menu_items:
            action = menu.addAction(item_text)
            action.triggered.connect(handler)
        btn.setMenu(menu)
        btn.clicked.connect(default_handler)
        return btn

    def _create_window_control_button(
        self,
        standard_icon: QStyle.StandardPixmap,
        tooltip: str,
        handler,
        *,
        role: str = "default",
    ) -> QPushButton:
        """Создаёт кнопку управления безрамочным окном."""

        button = QPushButton(self.main_toolbar)
        button.setProperty("windowControl", "true")
        button.setProperty("controlRole", role)
        button.setCursor(Qt.PointingHandCursor)
        button.setToolTip(tooltip)
        button.setIcon(self.style().standardIcon(standard_icon))
        button.setFixedSize(36, 36)
        button.clicked.connect(handler)
        return button

    def _toggle_maximized(self) -> None:
        """Переключает состояние окна между обычным и развёрнутым."""

        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
        self._update_window_control_buttons()

    def _update_window_control_buttons(self) -> None:
        """Синхронизирует кнопку разворота с текущим состоянием окна."""

        if not hasattr(self, "window_maximize_button"):
            return
        icon_type = (
            QStyle.SP_TitleBarNormalButton
            if self.isMaximized()
            else QStyle.SP_TitleBarMaxButton
        )
        tooltip = "Восстановить размер" if self.isMaximized() else "Развернуть"
        self.window_maximize_button.setIcon(self.style().standardIcon(icon_type))
        self.window_maximize_button.setToolTip(tooltip)

    def _install_frameless_event_filters(self) -> None:
        """Включает обработку drag/resize для безрамочного окна."""

        widgets: list[QWidget] = [self]
        widgets.extend(self.findChildren(QWidget))
        installed: set[int] = set()
        for widget in widgets:
            widget_id = id(widget)
            if widget_id in installed:
                continue
            installed.add(widget_id)
            widget.setMouseTracking(True)
            if widget is not self.graphics_view.viewport():
                widget.installEventFilter(self)

    def _handle_frameless_window_event(self, watched, event) -> bool:
        """Обрабатывает перетаскивание и изменение размера безрамочного окна."""

        if not isinstance(watched, QWidget):
            return False
        if watched is not self and not self.isAncestorOf(watched):
            return False
        if event.type() not in {
            QEvent.MouseButtonPress,
            QEvent.MouseButtonRelease,
            QEvent.MouseButtonDblClick,
            QEvent.MouseMove,
            QEvent.Leave,
        }:
            return False

        window_pos = self._window_pos_from_widget_event(watched, event)
        if window_pos is None:
            return False

        if event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
            if self._is_window_drag_source(watched, event):
                self._toggle_maximized()
                event.accept()
                return True
            return False

        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            edges = self._resize_edges_for_pos(window_pos)
            if edges and not self.isMaximized():
                self._resize_active = True
                self._resize_edges = edges
                self._resize_start_geometry = QRectF(self.geometry())
                self._resize_start_global_pos = event.globalPos()
                event.accept()
                return True

            if self._is_window_drag_source(watched, event) and not self.isMaximized():
                self._window_drag_active = True
                self._window_drag_offset = (
                    event.globalPos() - self.frameGeometry().topLeft()
                )
                event.accept()
                return True
            return False

        if event.type() == QEvent.MouseMove:
            if self._resize_active and event.buttons() & Qt.LeftButton:
                self._perform_window_resize(event.globalPos())
                event.accept()
                return True
            if self._window_drag_active and event.buttons() & Qt.LeftButton:
                self.move(event.globalPos() - self._window_drag_offset)
                event.accept()
                return True
            self._update_resize_cursor(watched, window_pos)
            return False

        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            released = self._resize_active or self._window_drag_active
            self._resize_active = False
            self._resize_edges = Qt.Edges()
            self._window_drag_active = False
            self._update_resize_cursor(watched, window_pos)
            if released:
                event.accept()
                return True
            return False

        if event.type() == QEvent.Leave and not (self._resize_active or self._window_drag_active):
            watched.unsetCursor()
        return False

    def _window_pos_from_widget_event(self, watched: QWidget, event) -> QPoint | None:
        """Преобразует позицию события виджета в координаты окна."""

        if not hasattr(event, "pos"):
            return None
        if watched is self:
            return event.pos()
        return self.mapFromGlobal(watched.mapToGlobal(event.pos()))

    def _is_window_drag_source(self, watched: QWidget, event) -> bool:
        """Определяет, можно ли начать перетаскивание окна из этой области."""

        if watched is getattr(self, "window_drag_handle", None):
            return True

        if watched is getattr(self, "main_toolbar", None):
            child = watched.childAt(event.pos())
            return child is None or child is self.window_drag_handle

        if watched is self.menuBar():
            corner_widget = self.menuBar().cornerWidget(Qt.TopRightCorner)
            if corner_widget is not None and corner_widget.geometry().contains(event.pos()):
                return False
            return self.menuBar().actionAt(event.pos()) is None

        return False

    def _resize_edges_for_pos(self, pos: QPoint) -> Qt.Edges:
        """Определяет активные края окна для изменения размера."""

        if self.isMaximized():
            return Qt.Edges()

        edges = Qt.Edges()
        rect = self.rect()
        margin = self._resize_margin

        if pos.x() <= rect.left() + margin:
            edges |= Qt.LeftEdge
        elif pos.x() >= rect.right() - margin:
            edges |= Qt.RightEdge

        if pos.y() <= rect.top() + margin:
            edges |= Qt.TopEdge
        elif pos.y() >= rect.bottom() - margin:
            edges |= Qt.BottomEdge

        return edges

    def _cursor_for_edges(self, edges: Qt.Edges):
        """Подбирает курсор под активные края ресайза."""

        if edges in (Qt.LeftEdge, Qt.RightEdge):
            return Qt.SizeHorCursor
        if edges in (Qt.TopEdge, Qt.BottomEdge):
            return Qt.SizeVerCursor
        if edges in (
            Qt.TopEdge | Qt.LeftEdge,
            Qt.BottomEdge | Qt.RightEdge,
        ):
            return Qt.SizeFDiagCursor
        if edges in (
            Qt.TopEdge | Qt.RightEdge,
            Qt.BottomEdge | Qt.LeftEdge,
        ):
            return Qt.SizeBDiagCursor
        return None

    def _update_resize_cursor(self, watched: QWidget, pos: QPoint) -> None:
        """Обновляет курсор при наведении на границы окна."""

        if watched is self.graphics_view.viewport() and self._measure_mode:
            edges = self._resize_edges_for_pos(pos)
            if not edges:
                return
        edges = self._resize_edges_for_pos(pos)
        cursor = self._cursor_for_edges(edges)
        if cursor is None:
            watched.unsetCursor()
            return
        watched.setCursor(cursor)

    def _perform_window_resize(self, global_pos: QPoint) -> None:
        """Изменяет размер окна по активным краям."""

        delta = global_pos - self._resize_start_global_pos
        start = self._resize_start_geometry.toRect()
        new_rect = QRectF(start)
        minimum = self.minimumSizeHint().expandedTo(self.minimumSize())
        min_width = max(480, minimum.width())
        min_height = max(360, minimum.height())

        if self._resize_edges & Qt.LeftEdge:
            new_left = min(
                start.right() - min_width + 1,
                start.left() + delta.x(),
            )
            new_rect.setLeft(new_left)
        if self._resize_edges & Qt.RightEdge:
            new_right = max(
                start.left() + min_width - 1,
                start.right() + delta.x(),
            )
            new_rect.setRight(new_right)
        if self._resize_edges & Qt.TopEdge:
            new_top = min(
                start.bottom() - min_height + 1,
                start.top() + delta.y(),
            )
            new_rect.setTop(new_top)
        if self._resize_edges & Qt.BottomEdge:
            new_bottom = max(
                start.top() + min_height - 1,
                start.bottom() + delta.y(),
            )
            new_rect.setBottom(new_bottom)

        self.setGeometry(new_rect.toRect())

    def _show_error(self, title: str, text: str) -> None:
        """Показывает модальное сообщение об ошибке."""
        QMessageBox.critical(self, title, text)

    def _show_info(self, title: str, text: str) -> None:
        """Показывает модальное информационное сообщение."""
        QMessageBox.information(self, title, text)

    def _request_logout(self) -> None:
        """Инициирует завершение текущей пользовательской сессии."""

        self.logout_requested.emit()

    def _ensure_detect_model(self) -> InferenceBackend | None:
        """Лениво загружает модель детекции и возвращает её экземпляр."""
        if self.detect_model is not None:
            return self.detect_model
        try:
            self.detect_model = load_inference_backend(self.weights_path)
            return self.detect_model
        except Exception as error:
            self._log_error(
                f"Не удалось загрузить модель детекции: {self.weights_path}. Ошибка: {error}"
            )
            logger.exception("Не удалось загрузить модель детекции")
            self._show_error(
                "Ошибка модели детекции",
                f"Не удалось загрузить модель:\n{self.weights_path}\n\n{error}",
            )
            return None

    def _ensure_classify_model(self) -> InferenceBackend | None:
        """Лениво загружает модель сегментации частей и возвращает её экземпляр."""
        if self.classify_model is not None:
            return self.classify_model
        try:
            self.classify_model = load_inference_backend(self.classify_weights_path)
            return self.classify_model
        except Exception as error:
            self._log_error(
                "Не удалось загрузить модель сегментации: "
                f"{self.classify_weights_path}. Ошибка: {error}"
            )
            logger.exception("Не удалось загрузить модель сегментации")
            self._show_error(
                "Ошибка модели сегментации",
                (
                    "Не удалось загрузить модель:\n"
                    f"{self.classify_weights_path}\n\n{error}"
                ),
            )
            return None

    def clear_project(self) -> None:
        """Полностью очищает проект, состояние окна и текущие результаты анализа."""
        self.app_state = AppState(
            image_storage=OriginalImage(),
            current_user_id=self.current_user.id if self.current_user else None,
        )
        self.image_storage = self.app_state.image_storage
        self._active_image_index = 0
        self._display_target = ("page", 0)
        self.zoom_factor = 1.0
        self.min_fit_zoom = 1.0
        self.app_state.zoom_factor = self.zoom_factor
        self.tree_widget.clear()
        self.statistics_panel.set_summary(
            StatisticsPanel.build_summary(self.image_storage)
        )
        self.graphics_scene.clear()
        self._pixmap_item = None
        self._original_pixmap = None
        self.rect_items = []
        self._calibration_pending = False
        self.pixels_per_mm = CALIBRATION_PIXELS_PER_MM_DEFAULT
        self.app_state.pixels_per_mm = self.pixels_per_mm
        self._set_canvas_empty(True)
        self._set_measure_mode(False)
        self._update_project_summary()
        self._update_canvas_status()

    def open_image(self) -> None:
        """Открывает диалог выбора файлов и загружает новый проект с нуля."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Открыть файлы",
            os.getcwd(),
            INPUT_FILE_FILTER,
        )
        if not paths:
            return
        self.clear_project()
        self._add_files_from_paths(paths)
        self._log_action(
            "upload_image",
            f"Загружено файлов в новый проект: {len(paths)}",
        )

    def add_files(self) -> None:
        """Открывает диалог выбора файлов и добавляет их в текущий проект."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Добавить файлы",
            os.getcwd(),
            INPUT_FILE_FILTER,
        )
        if not paths:
            return
        self._add_files_from_paths(paths)
        self._log_action(
            "upload_image",
            f"Добавлено файлов в текущий проект: {len(paths)}",
        )

    def _add_files_from_paths(self, paths: list[str]) -> None:
        """Загружает набор путей и переключается на первую успешно добавленную страницу."""
        first_new_index: int | None = None
        for path in paths:
            new_indices = self._load_path(path)
            if first_new_index is None and new_indices:
                first_new_index = new_indices[0]

        if self.image_storage.images and first_new_index is not None:
            self._select_page(first_new_index)
        self._refresh_tree()
        self._refresh_statistics_panel()

    def _load_path(self, file_path: str) -> list[int]:
        """Выбирает способ загрузки файла по его расширению."""
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(file_path)
        return self._load_image(file_path)

    def _load_image(self, file_path: str) -> list[int]:
        """Загружает одно изображение с диска и добавляет его как страницу проекта."""
        try:
            pages = load_pages_from_path(file_path)
        except ImageLoadError:
            self._log_error(f"Не удалось открыть изображение: {file_path}")
            self._show_error(
                "Ошибка открытия",
                f"Не удалось открыть изображение:\n{file_path}",
            )
            return []
        return [self._append_page(page.image, page.source_path) for page in pages]

    def _load_pdf(self, pdf_path: str) -> list[int]:
        """Render a PDF file into project pages."""
        pages: list[int] = []
        progress = AnalysisProgressDialog("Loading PDF", self)
        try:
            progress.setRange(0, 0)
            progress.show()

            def on_progress(done: int, total: int) -> None:
                progress.setRange(0, total)
                progress.setStep(f"Page {done} of {total}...")
                progress.setValue(done)
            loaded_pages = load_pages_from_path(
                pdf_path,
                on_progress=on_progress,
                should_cancel=progress.wasCanceled,
            )
            pages = [
                self._append_page(page.image, page.source_path)
                for page in loaded_pages
            ]
        except Exception as error:
            logger.exception("Failed to load PDF %s", pdf_path)
            self._log_error(f"Failed to load PDF {pdf_path}. Error: {error}")
            self._show_error(
                "PDF error",
                f"Failed to load PDF:\n{pdf_path}\n\n{error}",
            )
        finally:
            progress.close()
        return pages


    def _append_page(self, image: np.ndarray, source_path: str) -> int:
        """Добавляет страницу в хранилище проекта и список файлов интерфейса."""
        index = len(self.image_storage.images)
        self.image_storage.images.append(image)
        self.image_storage.source_files.append(source_path)
        if not self.image_storage.file_path:
            self.image_storage.file_path = source_path
        if self.image_storage.class_object_image is None:
            self.image_storage.class_object_image = []
        while len(self.image_storage.class_object_image) < len(
            self.image_storage.images
        ):
            self.image_storage.class_object_image.append([])

        self._update_project_summary()
        return index

    def _page_title(self, page_index: int) -> str:
        """Формирует заголовок страницы для списка проекта и дерева слоёв."""
        source = self._source_file(page_index)
        source_name = Path(source).name if source else f"page_{page_index + 1}"
        if source.lower().endswith(".pdf"):
            position, total = self._page_position_for_source(page_index)
            return f"{page_index + 1}. {source_name} · стр. {position}/{total}"
        return f"{page_index + 1}. {source_name}"

    def _source_file(self, page_index: int) -> str:
        """Возвращает путь исходного файла для указанной страницы."""
        if 0 <= page_index < len(self.image_storage.source_files):
            return self.image_storage.source_files[page_index]
        return self.image_storage.file_path

    def _select_page(self, page_index: int) -> None:
        """Делает страницу активной и обновляет связанные элементы интерфейса."""
        if page_index < 0 or page_index >= len(self.image_storage.images):
            return
        self._restore_calibration_for_index(page_index)
        self.display_image_with_boxes(page_index)
        self._sync_tree_to_active_page()
        self._refresh_statistics_panel()
        self._update_canvas_status()

    def _refresh_tree(self) -> None:
        """Перестраивает дерево проекта: все страницы, сеянцы и части."""
        self.tree_widget.blockSignals(True)
        self.tree_widget.clear()

        active = min(
            max(self._active_image_index, 0),
            max(len(self.image_storage.images) - 1, 0),
        )

        for page_index, image in enumerate(self.image_storage.images):
            source = self._source_file(page_index)
            image_type = "pdf" if source.lower().endswith(".pdf") else "original"
            root = self.tree_widget.add_root_item(
                name=self._page_title(page_index),
                description=Path(source).name,
                index=page_index,
                image_type=image_type,
                image=image,
            )
            page_objects = []
            if self.image_storage.class_object_image and page_index < len(
                self.image_storage.class_object_image
            ):
                page_objects = self.image_storage.class_object_image[page_index]
            for object_index, obj in enumerate(page_objects):
                child = self.tree_widget.add_child_item(
                    root,
                    self._seedling_title(object_index, obj),
                    self._object_description(obj),
                    page_index,
                    object_index,
                    "seeding",
                    obj.image[0] if obj.image else None,
                    confidence=obj.confidence,
                    manual=getattr(obj, "manual", False),
                )
                if getattr(obj, "orientation_uncertain", False):
                    child.setIcon(
                        0,
                        self.style().standardIcon(QStyle.SP_MessageBoxWarning),
                    )
                for class_index, part in enumerate(obj.image_all_class or []):
                    self.tree_widget.add_class_item(
                        child,
                        self._display_part_name(part.class_name),
                        self._part_description(part),
                        page_index,
                        object_index,
                        class_index,
                        confidence=part.confidence,
                        manual=getattr(part, "manual", False),
                        class_name=part.class_name,
                    )
                child.setExpanded(False)
            root.setExpanded(page_index == active)

        self.tree_widget.blockSignals(False)
        self._sync_tree_to_active_page()

    def _sync_tree_to_active_page(self) -> None:
        """Выделяет в дереве узел активной страницы без триггера сигналов."""
        self.tree_widget.blockSignals(True)
        for i in range(self.tree_widget.topLevelItemCount()):
            item = self.tree_widget.topLevelItem(i)
            payload = item.data(0, Qt.UserRole) or {}
            if payload.get("index") == self._active_image_index:
                self.tree_widget.setCurrentItem(item)
                break
        self.tree_widget.blockSignals(False)

    def _refresh_statistics_panel(self) -> None:
        """Пересчитывает и обновляет панель статистики для текущего контекста."""
        if not self.image_storage.images:
            summary = StatisticsPanel.build_summary(self.image_storage)
        else:
            page_index = min(
                max(self._active_image_index, 0),
                len(self.image_storage.images) - 1,
            )
            page_objects = []
            if self.image_storage.class_object_image and page_index < len(
                self.image_storage.class_object_image
            ):
                page_objects = self.image_storage.class_object_image[page_index]
            page_data = OriginalImage(
                file_path=self._source_file(page_index),
                source_files=[self._source_file(page_index)],
                images=[self.image_storage.images[page_index]],
                class_object_image=[page_objects],
            )
            summary = StatisticsPanel.build_summary(page_data)
        self.statistics_panel.set_summary(summary)

    def _on_tree_selection_changed(self) -> None:
        """Переключает отображение по выбору пользователя в дереве слоёв."""
        item = self.tree_widget.currentItem()
        if item is None:
            self.detail_panel.clear()
            return
        payload = item.data(0, Qt.UserRole) or {}
        item_type = payload.get("type")
        self.app_state.selected_item = payload

        if item_type in {"original", "pdf"}:
            self._select_page(int(payload["index"]))
            self.detail_panel.clear()
            return
        if item_type == "seeding":
            self.display_image_with_boxes(
                int(payload["parent_index"]),
                seeding_idx=int(payload["index"]),
            )
            self._update_detail_panel(payload)
            return
        if item_type == "class":
            self.display_image_with_boxes(
                int(payload["parent_index"]),
                seeding_idx=int(payload["seeding_index"]),
            )
            self._update_detail_panel(payload)

    def _update_detail_panel(self, payload: dict) -> None:
        """Заполняет детальную панель данными выбранного объекта."""
        item_type = payload.get("type")
        page_idx = int(payload.get("parent_index", 0))

        page_objects = []
        if (self.image_storage.class_object_image
                and page_idx < len(self.image_storage.class_object_image)):
            page_objects = self.image_storage.class_object_image[page_idx]

        if item_type == "seeding":
            seed_idx = int(payload["index"])
            if seed_idx >= len(page_objects):
                self.detail_panel.clear()
                return
            obj = page_objects[seed_idx]
            crop = obj.image[0] if obj.image and isinstance(obj.image[0], np.ndarray) else None
            self.detail_panel.set_object(
                name=self._display_part_name(obj.class_name),
                confidence=obj.confidence,
                manual=getattr(obj, "manual", False),
                crop_image=crop,
                mask_bitmap=None,
                mask_color=(78, 200, 100),
                bbox=obj.bbox,
                pixels_per_mm=self.pixels_per_mm,
            )
            return

        if item_type == "class":
            seed_idx = int(payload["seeding_index"])
            class_idx = int(payload["class_index"])
            if seed_idx >= len(page_objects):
                self.detail_panel.clear()
                return
            obj = page_objects[seed_idx]
            parts = obj.image_all_class or []
            if class_idx >= len(parts):
                self.detail_panel.clear()
                return
            part = parts[class_idx]

            crop = None
            if obj.image and isinstance(obj.image[0], np.ndarray) and part.bbox:
                seed_crop = obj.image[0]
                x1, y1, x2, y2 = part.bbox
                sx1, sy1 = (obj.bbox[0], obj.bbox[1]) if obj.bbox else (0, 0)
                rx1 = max(0, x1 - sx1)
                ry1 = max(0, y1 - sy1)
                rx2 = min(seed_crop.shape[1], x2 - sx1)
                ry2 = min(seed_crop.shape[0], y2 - sy1)
                if rx2 > rx1 and ry2 > ry1:
                    crop = seed_crop[ry1:ry2, rx1:rx2]

            fill_color, _ = self._part_mask_colors(part.class_name)
            mask_color = (fill_color.red(), fill_color.green(), fill_color.blue())
            fallback_crop = obj.image[0] if obj.image and isinstance(obj.image[0], np.ndarray) else None
            self.detail_panel.set_object(
                name=self._display_part_name(part.class_name),
                confidence=part.confidence,
                manual=getattr(part, "manual", False),
                crop_image=crop if (crop is not None and crop.size > 0) else fallback_crop,
                mask_bitmap=getattr(part, "mask_bitmap", None),
                mask_color=mask_color,
                bbox=part.bbox,
                pixels_per_mm=self.pixels_per_mm,
            )

    def _navigate_seedling(self, delta: int) -> None:
        """Переключает выбор на следующий/предыдущий сеянец (delta = +1 или -1)."""
        item = self.tree_widget.currentItem()
        if item is None:
            return
        payload = item.data(0, Qt.UserRole) or {}
        item_type = payload.get("type")

        if item_type == "class":
            item = item.parent()
            if item is None:
                return
            payload = item.data(0, Qt.UserRole) or {}

        if payload.get("type") != "seeding":
            return

        parent_item = item.parent()
        if parent_item is None:
            return
        current_row = parent_item.indexOfChild(item)
        new_row = current_row + delta
        if 0 <= new_row < parent_item.childCount():
            new_item = parent_item.child(new_row)
            self.tree_widget.setCurrentItem(new_item)

    def display_image(
        self,
        image: np.ndarray,
        *,
        preserve_view: bool = False,
        previous_zoom: float | None = None,
        previous_center: QPointF | None = None,
    ) -> None:
        """Отображает изображение на сцене без добавления bbox и масок."""
        self.graphics_scene.clear()
        self.rect_items = []
        self.mask_items = []
        self._pending_box_item = None
        self._pending_box_target = None
        self._pixmap_item = None
        self._original_pixmap = None
        self._reset_measure_state(clear_items=True)

        if image is None or not isinstance(image, np.ndarray):
            self._set_canvas_empty(True)
            return
        self._set_canvas_empty(False)

        if image.ndim == 2:
            q_image = QImage(
                image.data,
                image.shape[1],
                image.shape[0],
                image.shape[1],
                QImage.Format_Grayscale8,
            )
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            q_image = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                rgb.shape[1] * 3,
                QImage.Format_RGB888,
            )

        pixmap = QPixmap.fromImage(q_image)
        self._original_pixmap = pixmap
        self._pixmap_item = self.graphics_scene.addPixmap(pixmap)
        self.graphics_scene.setSceneRect(QRectF(pixmap.rect()))
        viewport = self.graphics_view.viewport().size()
        ratio_w = max(1, viewport.width()) / max(1, pixmap.width())
        ratio_h = max(1, viewport.height()) / max(1, pixmap.height())
        self.min_fit_zoom = max(0.05, min(ratio_w, ratio_h, 1.0))

        if preserve_view and previous_zoom is not None:
            self.zoom_factor = max(float(previous_zoom), self.min_fit_zoom)
            self.update_image_zoom()
            if previous_center is not None:
                scene_rect = self.graphics_scene.sceneRect()
                clamped_center = QPointF(
                    min(max(previous_center.x(), scene_rect.left()), scene_rect.right()),
                    min(max(previous_center.y(), scene_rect.top()), scene_rect.bottom()),
                )
                self.graphics_view.centerOn(clamped_center)
        else:
            self.fit_to_window()

    def fit_to_window(self) -> None:
        """Подгоняет масштаб изображения под размер области просмотра."""
        if self._original_pixmap is None:
            return
        self.zoom_factor = self.min_fit_zoom
        self.update_image_zoom()
        self.graphics_view.centerOn(self.graphics_scene.sceneRect().center())

    def update_image_zoom(self) -> None:
        """Применяет текущий коэффициент масштаба к виду и сцене."""
        if self._original_pixmap is None:
            return
        self.app_state.zoom_factor = self.zoom_factor
        transform = QTransform()
        transform.scale(self.zoom_factor, self.zoom_factor)
        self.graphics_view.setTransform(transform)
        self.graphics_scene.setSceneRect(
            0,
            0,
            self._original_pixmap.width(),
            self._original_pixmap.height(),
        )
        self._update_canvas_status()

    def _apply_zoom(
        self,
        factor: float,
        *,
        anchor_view_pos: QPoint | None = None,
    ) -> None:
        """Изменяет масштаб изображения, при необходимости закрепляя его по курсору."""
        if self._original_pixmap is None:
            return

        previous_center = self.graphics_view.mapToScene(
            self.graphics_view.viewport().rect().center()
        )
        anchor_scene_pos = (
            self.graphics_view.mapToScene(anchor_view_pos)
            if anchor_view_pos is not None
            else None
        )

        next_zoom = self.zoom_factor * factor
        next_zoom = max(self.min_fit_zoom, min(next_zoom, 20.0))
        if abs(next_zoom - self.zoom_factor) < 1e-9:
            return

        self.zoom_factor = next_zoom
        self.update_image_zoom()
        if anchor_view_pos is not None and anchor_scene_pos is not None:
            new_anchor_scene_pos = self.graphics_view.mapToScene(anchor_view_pos)
            delta = new_anchor_scene_pos - anchor_scene_pos
            self.graphics_view.centerOn(previous_center - delta)
        else:
            self.graphics_view.centerOn(previous_center)

    def zoom_in(self, *, anchor_view_pos: QPoint | None = None) -> None:
        """Увеличивает масштаб изображения на фиксированный шаг."""
        self._apply_zoom(
            ZOOM_FACTOR_INCREMENT,
            anchor_view_pos=anchor_view_pos,
        )

    def zoom_out(self, *, anchor_view_pos: QPoint | None = None) -> None:
        """Уменьшает масштаб изображения на фиксированный шаг."""
        self._apply_zoom(
            1.0 / ZOOM_FACTOR_INCREMENT,
            anchor_view_pos=anchor_view_pos,
        )

    def _part_bbox_to_global(
        self,
        page_width: int,
        page_height: int,
        seed_obj: ObjectImage,
        part_obj: AllClassImage,
    ) -> tuple[int, int, int, int] | None:
        """Переводит локальный bbox части из кропа в координаты исходной страницы."""
        if not seed_obj.bbox or not part_obj.bbox:
            return None
        sx1, sy1, sx2, sy2 = seed_obj.bbox
        lx1, ly1, lx2, ly2 = part_obj.bbox

        rotation_k = int(getattr(seed_obj, "rotation_k", 0)) % 4
        ux1, uy1, ux2, uy2 = lx1, ly1, lx2, ly2
        if rotation_k:
            crop_h = max(1, sy2 - sy1)
            crop_w = max(1, sx2 - sx1)
            if seed_obj.image and isinstance(seed_obj.image[0], np.ndarray):
                crop_h, crop_w = seed_obj.image[0].shape[:2]
            ux1, uy1, ux2, uy2 = rotate_bbox(
                lx1,
                ly1,
                lx2,
                ly2,
                crop_w,
                crop_h,
                (-rotation_k) % 4,
            )

        global_bbox = (sx1 + ux1, sy1 + uy1, sx1 + ux2, sy1 + uy2)
        return clip_bbox_to_image(global_bbox, page_width, page_height)

    def _part_mask_colors(self, class_name: str | None) -> tuple[QColor, QColor]:
        """Возвращает цвета заливки и контура маски по типу части растения."""
        value = (class_name or "").strip().lower()
        if value.startswith("root"):
            return QColor(52, 199, 89, 80), QColor(52, 199, 89, 210)
        if value.startswith("stem"):
            return QColor(255, 159, 10, 80), QColor(255, 159, 10, 210)
        if value.startswith("flower") or value == "inflorescence":
            return QColor(64, 156, 255, 80), QColor(64, 156, 255, 210)
        return QColor(46, 226, 201, 70), QColor(46, 226, 201, 190)

    def _add_part_mask_item(self, part_obj: AllClassImage) -> None:
        """Отображает маску части растения.

        Приоритет у попиксельного bitmap (точное покрытие тонких структур),
        fallback — полигональный контур (совместимость со старыми данными).
        """
        fill_color, outline_color = self._part_mask_colors(part_obj.class_name)

        bitmap = getattr(part_obj, "mask_bitmap", None)
        if isinstance(bitmap, np.ndarray) and bitmap.ndim == 2 and bitmap.size > 0:
            self._add_bitmap_mask_item(bitmap, fill_color, outline_color)
            return

        polygon = getattr(part_obj, "mask_polygon", None)
        if polygon is None:
            return

        points = np.asarray(polygon, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
            return

        q_polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in points])
        item = self.graphics_scene.addPolygon(
            q_polygon,
            QPen(outline_color, 2),
            QBrush(fill_color),
        )
        item.setAcceptedMouseButtons(Qt.NoButton)
        item.setZValue(0.5)
        self.mask_items.append(item)

    def _add_bitmap_mask_item(
        self,
        bitmap: np.ndarray,
        fill_color: QColor,
        outline_color: QColor,
    ) -> None:
        """Рендерит bitmap-маску как QPainterPath с OddEvenFill.

        Дыры (просветы между хвоинками/корнями) автоматически прозрачны —
        путь строится по всем контурам через RETR_CCOMP.
        """
        binary = (bitmap > 0).astype(np.uint8) * 255
        if not np.any(binary):
            return

        contours = bitmap_to_contours(binary)
        if not contours:
            return

        path = QPainterPath()
        path.setFillRule(Qt.OddEvenFill)
        for contour in contours:
            poly = QPolygonF([QPointF(float(x), float(y)) for x, y in contour])
            sub = QPainterPath()
            sub.addPolygon(poly)
            sub.closeSubpath()
            path.addPath(sub)

        path_item = QGraphicsPathItem(path)
        path_item.setPen(QPen(outline_color, 1.5))
        path_item.setBrush(QBrush(fill_color))
        path_item.setAcceptedMouseButtons(Qt.NoButton)
        path_item.setZValue(0.5)
        self.graphics_scene.addItem(path_item)
        self.mask_items.append(path_item)

    def _selection_payload_for_display(
        self,
        img_idx: int,
        seeding_idx: int | None = None,
    ) -> SelectionPayload:
        """Формирует payload выбора действия для текущего отображаемого элемента."""
        if seeding_idx is not None:
            return {"type": "seeding", "parent_index": img_idx, "index": seeding_idx}

        source_path = self._source_file(img_idx)
        item_type = "pdf" if source_path.lower().endswith(".pdf") else "original"
        return {"type": item_type, "index": img_idx}

    def display_image_with_boxes(
        self,
        img_idx: int,
        seeding_idx: int | None = None,
        *,
        preserve_view: bool = False,
    ) -> None:
        """Показывает страницу или кроп вместе с bbox и масками, если они доступны."""
        if img_idx < 0 or img_idx >= len(self.image_storage.images):
            return

        self._active_image_index = img_idx
        self.app_state.active_image_index = img_idx

        if seeding_idx is None:
            base_img = self.image_storage.images[img_idx]
            objects_to_draw = (
                self.image_storage.class_object_image[img_idx]
                if self.image_storage.class_object_image
                else []
            )
        else:
            if not self.image_storage.class_object_image:
                return
            page_objects = self.image_storage.class_object_image[img_idx]
            if seeding_idx < 0 or seeding_idx >= len(page_objects):
                return
            obj = page_objects[seeding_idx]
            if not obj.image:
                return
            base_img = obj.image[0]
            objects_to_draw = obj.image_all_class or []

        self._display_target = (
            ("page", img_idx) if seeding_idx is None else ("crop", img_idx, seeding_idx)
        )
        self.app_state.selected_item = self._selection_payload_for_display(
            img_idx,
            seeding_idx,
        )

        previous_zoom = None
        previous_center = None
        if preserve_view and self._original_pixmap is not None:
            previous_zoom = float(self.zoom_factor)
            previous_center = self.graphics_view.mapToScene(
                self.graphics_view.viewport().rect().center()
            )

        self.display_image(
            base_img,
            preserve_view=preserve_view,
            previous_zoom=previous_zoom,
            previous_center=previous_center,
        )
        if self._pixmap_item is None:
            return

        if seeding_idx is None:
            page_height, page_width = base_img.shape[:2]
            for seed_obj in objects_to_draw:
                if self._show_boxes and seed_obj.bbox:
                    x1, y1, x2, y2 = seed_obj.bbox
                    rect = QRectF(x1, y1, x2 - x1, y2 - y1)
                    item = BBoxItem(
                        rect,
                        seed_obj,
                        bbox_commit_callback=lambda obj, page=img_idx: self._commit_seedling_bbox(
                            page,
                            obj,
                            auto_segment=True,
                        ),
                        class_label=self._display_part_name(seed_obj.class_name),
                        pixels_per_mm=self.pixels_per_mm,
                    )
                    item._manual_edit_enabled = True
                    item.setZValue(1.0)
                    self.graphics_scene.addItem(item)
                    self.rect_items.append(item)

                if self._show_boxes:
                    for part_obj in seed_obj.image_all_class or []:
                        global_bbox = self._part_bbox_to_global(
                            page_width,
                            page_height,
                            seed_obj,
                            part_obj,
                        )
                        if global_bbox is None:
                            continue
                        px1, py1, px2, py2 = global_bbox
                        rect = QRectF(px1, py1, px2 - px1, py2 - py1)
                        item = BBoxItem(
                            rect,
                            part_obj,
                            class_label=self._display_part_name(part_obj.class_name),
                            pixels_per_mm=self.pixels_per_mm,
                        )
                        item._manual_edit_enabled = False
                        item.setZValue(1.0)
                        self.graphics_scene.addItem(item)
                        self.rect_items.append(item)
        else:
            for part_obj in objects_to_draw:
                if self._show_masks:
                    self._add_part_mask_item(part_obj)
                if not self._show_boxes or not part_obj.bbox:
                    continue
                x1, y1, x2, y2 = part_obj.bbox
                rect = QRectF(x1, y1, x2 - x1, y2 - y1)
                item = BBoxItem(
                    rect,
                    part_obj,
                    bbox_commit_callback=lambda obj: self._commit_part_bbox(obj),
                    class_label=self._display_part_name(part_obj.class_name),
                    pixels_per_mm=self.pixels_per_mm,
                )
                item._manual_edit_enabled = True
                item.setZValue(1.0)
                self.graphics_scene.addItem(item)
                self.rect_items.append(item)

        self._apply_interaction_mode_to_rect_items()
        self._update_canvas_status()


    def _current_page_objects(self, page_index: int) -> list[ObjectImage]:
        if self.image_storage.class_object_image is None:
            self.image_storage.class_object_image = []
        while len(self.image_storage.class_object_image) < len(self.image_storage.images):
            self.image_storage.class_object_image.append([])
        return self.image_storage.class_object_image[page_index]

    def _current_crop_context(self) -> tuple[int, int, ObjectImage] | None:
        if len(self._display_target) != 3 or self._display_target[0] != "crop":
            return None
        page_index = int(self._display_target[1])
        object_index = int(self._display_target[2])
        if not self.image_storage.class_object_image:
            return None
        if page_index >= len(self.image_storage.class_object_image):
            return None
        objects = self.image_storage.class_object_image[page_index]
        if object_index >= len(objects):
            return None
        return page_index, object_index, objects[object_index]

    def _start_add_box_mode(self) -> None:
        if self._pixmap_item is None:
            return
        if self._interaction_mode != "edit_boxes":
            self._set_interaction_mode("edit_boxes")
        self._clear_pending_box()
        self._box_draw_active = True
        self._box_draw_start = None
        self.statusBar().showMessage(
            "Нарисуйте bbox, затем подгоните его ручками и нажмите ОК.",
            4000,
        )
        self._update_annotation_controls()

    def _clear_pending_box(self) -> None:
        if self._pending_box_item is not None:
            try:
                scene = self._pending_box_item.scene()
                if scene is not None:
                    scene.removeItem(self._pending_box_item)
            except RuntimeError:
                pass
        self._pending_box_item = None
        self._pending_box_target = None

    def _cancel_add_box_mode(self) -> None:
        self._box_draw_active = False
        self._box_draw_start = None
        if self._box_draw_item is not None:
            try:
                scene = self._box_draw_item.scene()
                if scene is not None:
                    scene.removeItem(self._box_draw_item)
            except RuntimeError:
                pass
            self._box_draw_item = None
        self._clear_pending_box()
        self._update_annotation_controls()

    def _finish_add_box(self, rect: QRectF) -> None:
        self._box_draw_active = False
        self._box_draw_start = None
        if self._box_draw_item is not None:
            try:
                scene = self._box_draw_item.scene()
                if scene is not None:
                    scene.removeItem(self._box_draw_item)
            except RuntimeError:
                pass
            self._box_draw_item = None
        if rect.width() < 3 or rect.height() < 3:
            self._update_annotation_controls()
            return
        self._create_pending_box(rect)

    def _create_pending_box(self, rect: QRectF) -> None:
        self._clear_pending_box()
        bbox = self._bbox_from_scene_rect(rect)
        pending = ObjectImage(class_name="new", confidence=1.0, bbox=bbox)
        item = BBoxItem(
            QRectF(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]),
            pending,
            class_label="Новый",
            pixels_per_mm=self.pixels_per_mm,
        )
        item._manual_edit_enabled = True
        item.setEditable(True)
        item.setSelected(True)
        item.setZValue(3.0)
        self.graphics_scene.addItem(item)
        self._pending_box_item = item
        self._pending_box_target = tuple(self._display_target)
        self._update_annotation_controls()
        self.statusBar().showMessage(
            "Подгоните bbox и нажмите ОК, чтобы построить маску.",
            5000,
        )

    def _bbox_from_scene_rect(self, rect: QRectF) -> tuple[int, int, int, int]:
        normalized = rect.normalized()
        return (
            int(round(normalized.left())),
            int(round(normalized.top())),
            int(round(normalized.right())),
            int(round(normalized.bottom())),
        )

    def _confirm_pending_box(self) -> None:
        if self._pending_box_item is None or self._pending_box_target is None:
            return
        self._pending_box_item.update_bbox()
        bbox = self._pending_box_item.obj.bbox
        target = self._pending_box_target
        self._clear_pending_box()
        self._update_annotation_controls()
        if target[0] == "page":
            self._add_seedling_box(int(target[1]), bbox, auto_segment=True)
            return
        if target[0] == "crop":
            context = self._current_crop_context()
            if context is not None:
                self._add_part_box(context[2], bbox)

    def _rect_mask_for_bbox(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        polygon = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
        coarse = polygon_to_bitmap(polygon, h, w)
        if coarse is None:
            return polygon, None
        refined = refine_mask_bitmap(image, coarse, build_refine_context(image))
        mask = refined if refined is not None else coarse
        refined_polygon = bitmap_to_polygon(mask)
        return (refined_polygon if refined_polygon is not None else polygon), mask

    def _choose_annotation_class(self, classes: list[str], title: str) -> str | None:
        value, ok = QInputDialog.getItem(self, title, "Класс:", classes, 0, False)
        if not ok:
            return None
        return str(value)

    def _add_seedling_box(
        self,
        page_index: int,
        bbox,
        *,
        auto_segment: bool = False,
    ) -> None:
        page_img = self.image_storage.images[page_index]
        height, width = page_img.shape[:2]
        clipped = clip_bbox_to_image(bbox, width, height)
        if clipped is None:
            return
        class_name = self._choose_annotation_class(["cedr", "pinus", "seeding"], "Новый сеянец")
        if class_name is None:
            return
        x1, y1, x2, y2 = clipped
        crop = page_img[y1:y2, x1:x2].copy()
        obj = ObjectImage(
            class_name=class_name,
            confidence=1.0,
            image=[crop],
            image_all_class=None,
            bbox=clipped,
        )
        objects = self._current_page_objects(page_index)
        objects.append(obj)
        object_index = len(objects) - 1
        self._after_manual_annotation_change(page_index, seeding_idx=object_index)
        if auto_segment:
            self._segment_single_seedling(page_index, object_index, obj)

    def _add_part_box(self, seed_obj: ObjectImage, bbox) -> None:
        if not seed_obj.image:
            return
        crop_img = seed_obj.image[0]
        height, width = crop_img.shape[:2]
        clipped = clip_bbox_to_image(bbox, width, height)
        if clipped is None:
            return
        class_name = self._choose_annotation_class(
            ["root", "stem", "inflorescence"],
            "Новая часть растения",
        )
        if class_name is None:
            return
        x1, y1, x2, y2 = clipped
        mask_polygon, mask_bitmap = self._rect_mask_for_bbox(crop_img, clipped)
        part = AllClassImage(
            class_name=class_name,
            confidence=1.0,
            image=crop_img[y1:y2, x1:x2].copy(),
            bbox=clipped,
            mask_polygon=mask_polygon,
            mask_bitmap=mask_bitmap,
        )
        if seed_obj.image_all_class is None:
            seed_obj.image_all_class = []
        seed_obj.image_all_class.append(part)
        context = self._current_crop_context()
        if context is not None:
            self._after_manual_annotation_change(context[0], seeding_idx=context[1])

    def _commit_seedling_bbox(
        self,
        page_index: int,
        obj: ObjectImage,
        *,
        auto_segment: bool = False,
    ) -> None:
        if page_index >= len(self.image_storage.images) or not obj.bbox:
            return
        page_img = self.image_storage.images[page_index]
        height, width = page_img.shape[:2]
        clipped = clip_bbox_to_image(obj.bbox, width, height)
        if clipped is None:
            return
        obj.bbox = clipped
        x1, y1, x2, y2 = clipped
        obj.image = [page_img[y1:y2, x1:x2].copy()]
        obj.image_all_class = None
        self._after_manual_annotation_change(page_index)
        if auto_segment:
            try:
                object_index = self._current_page_objects(page_index).index(obj)
            except ValueError:
                return
            self._segment_single_seedling(page_index, object_index, obj)

    def _commit_part_bbox(self, part: AllClassImage) -> None:
        context = self._current_crop_context()
        if context is None or not part.bbox:
            return
        page_index, seeding_idx, seed_obj = context
        if not seed_obj.image:
            return
        crop_img = seed_obj.image[0]
        height, width = crop_img.shape[:2]
        clipped = clip_bbox_to_image(part.bbox, width, height)
        if clipped is None:
            return
        part.bbox = clipped
        x1, y1, x2, y2 = clipped
        part.image = crop_img[y1:y2, x1:x2].copy()
        part.mask_polygon, part.mask_bitmap = self._rect_mask_for_bbox(crop_img, clipped)
        self._after_manual_annotation_change(page_index, seeding_idx=seeding_idx)

    def _after_manual_annotation_change(
        self,
        page_index: int,
        *,
        seeding_idx: int | None = None,
    ) -> None:
        self._refresh_tree()
        self._refresh_statistics_panel()
        self.display_image_with_boxes(page_index, seeding_idx, preserve_view=True)
        self._auto_save_session()

    def _selected_annotation_object(self):
        for item in self.graphics_scene.selectedItems():
            if isinstance(item, BBoxItem):
                return item.obj
        item = self.tree_widget.currentItem()
        payload = item.data(0, Qt.UserRole) if item is not None else None
        if not payload:
            return None
        item_type = payload.get("type")
        if item_type == "seeding" and self.image_storage.class_object_image:
            objects = self.image_storage.class_object_image[int(payload["parent_index"])]
            return objects[int(payload["index"])]
        if item_type == "class" and self.image_storage.class_object_image:
            objects = self.image_storage.class_object_image[int(payload["parent_index"])]
            seed_obj = objects[int(payload["seeding_index"])]
            return (seed_obj, seed_obj.image_all_class[int(payload["class_index"])])
        return None

    def _delete_selected_annotation(self) -> None:
        selected = self._selected_annotation_object()
        if selected is None:
            self.statusBar().showMessage("Выберите bbox для удаления.", 2500)
            return
        reply = QMessageBox.question(
            self,
            "Удаление аннотации",
            "Удалить выбранную аннотацию?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        if isinstance(selected, tuple):
            seed_obj, part = selected
            if seed_obj.image_all_class and part in seed_obj.image_all_class:
                seed_obj.image_all_class.remove(part)
                context = self._current_crop_context()
                page_index = context[0] if context is not None else self._active_image_index
                seeding_idx = context[1] if context is not None else None
                self._after_manual_annotation_change(page_index, seeding_idx=seeding_idx)
                self.statusBar().showMessage("Часть растения удалена.", 2500)
            return
        if isinstance(selected, AllClassImage) and self.image_storage.class_object_image:
            for page_index, objects in enumerate(self.image_storage.class_object_image):
                for seeding_idx, seed_obj in enumerate(objects):
                    if seed_obj.image_all_class and selected in seed_obj.image_all_class:
                        seed_obj.image_all_class.remove(selected)
                        self._after_manual_annotation_change(
                            page_index,
                            seeding_idx=seeding_idx,
                        )
                        self.statusBar().showMessage("Часть растения удалена.", 2500)
                        return
        if not isinstance(selected, ObjectImage) or not self.image_storage.class_object_image:
            return
        for page_index, objects in enumerate(self.image_storage.class_object_image):
            if selected in objects:
                objects.remove(selected)
                self._after_manual_annotation_change(page_index)
                self.statusBar().showMessage("Сеянец удалён.", 2500)
                return

    def _seedling_title(self, object_index: int, obj: ObjectImage) -> str:
        """Возвращает заголовок узла сеянца для дерева слоёв."""
        return f"{self._display_part_name(obj.class_name)} {object_index + 1}"

    def _object_description(self, obj: ObjectImage) -> str:
        """Формирует краткое описание объекта с уверенностью и bbox."""
        bbox = obj.bbox if obj.bbox else ("-", "-", "-", "-")
        return f"conf={obj.confidence:.2f}, bbox={bbox}"

    def _part_description(self, part: AllClassImage) -> str:
        """Формирует краткое описание части растения с уверенностью и bbox."""
        bbox = part.bbox if part.bbox else ("-", "-", "-", "-")
        return f"conf={part.confidence:.2f}, bbox={bbox}"

    def _display_part_name(self, class_name: str | None) -> str:
        """Преобразует внутреннее имя класса в человекочитаемое название."""
        value = (class_name or "").strip().lower()
        return CLASS_DISPLAY_NAMES.get(value, class_name or "Часть")

    def _restore_display(self, *, preserve_view: bool = False) -> None:
        """Восстанавливает ранее выбранный режим показа страницы или кропа."""
        target = self._display_target
        if not target or not self.image_storage.images:
            return
        if target[0] == "crop" and len(target) == 3:
            self.display_image_with_boxes(
                int(target[1]),
                seeding_idx=int(target[2]),
                preserve_view=preserve_view,
            )
        else:
            self.display_image_with_boxes(
                int(target[1]),
                preserve_view=preserve_view,
            )

    def _analysis_is_running(self) -> bool:
        return self._analysis_thread is not None and self._analysis_thread.isRunning()

    def _cleanup_analysis_worker(self) -> None:
        self._analysis_thread = None
        self._analysis_progress = None

    def _on_analysis_progress(
        self,
        progress: AnalysisProgressDialog,
        done: int,
        total: int,
        text: str,
    ) -> None:
        progress.setRange(0, total)
        if text:
            progress.setStep(text)
        progress.setValue(done)

    def _start_detection_worker(
        self,
        page_indices: list[int],
        *,
        title: str,
        on_complete,
    ) -> None:
        if self._analysis_is_running():
            self.statusBar().showMessage("Анализ уже выполняется", 3000)
            return
        model = self._ensure_detect_model()
        if model is None:
            return

        progress = AnalysisProgressDialog(title, self)
        progress.setRange(0, len(page_indices))
        progress.show()

        thread = DetectionWorkerThread(model, self.app_state, page_indices, self)
        progress.canceled.connect(thread.request_cancel)
        thread.progress.connect(
            lambda done, total, text: self._on_analysis_progress(
                progress, done, total, text
            )
        )
        thread.finished_ok.connect(
            lambda result: self._on_detection_worker_done(
                progress, result, on_complete
            )
        )
        thread.error.connect(lambda error: self._on_analysis_worker_error(progress, error))
        thread.finished.connect(thread.deleteLater)

        self._analysis_progress = progress
        self._analysis_thread = thread
        thread.start()

    def _start_classification_worker(
        self,
        pending: list[tuple[int, int, ObjectImage]],
        *,
        title: str,
        on_complete,
    ) -> None:
        if self._analysis_is_running():
            self.statusBar().showMessage("Анализ уже выполняется", 3000)
            return
        model = self._ensure_classify_model()
        if model is None:
            return

        progress = AnalysisProgressDialog(title, self)
        progress.setRange(0, len(pending))
        progress.show()

        thread = ClassificationWorkerThread(
            model,
            self.app_state,
            pending,
            self._analysis_batch_size,
            self,
        )
        progress.canceled.connect(thread.request_cancel)
        thread.progress.connect(
            lambda done, total, text: self._on_analysis_progress(
                progress, done, total, text
            )
        )
        thread.finished_ok.connect(
            lambda result: self._on_classification_worker_done(
                progress, result, on_complete
            )
        )
        thread.error.connect(lambda error: self._on_analysis_worker_error(progress, error))
        thread.finished.connect(thread.deleteLater)

        self._analysis_progress = progress
        self._analysis_thread = thread
        thread.start()

    def _on_detection_worker_done(self, progress, result, on_complete) -> None:
        progress.close()
        self._refresh_tree()
        self._refresh_statistics_panel()
        self._restore_display(preserve_view=True)
        on_complete(result)
        self._auto_save_session()
        self._cleanup_analysis_worker()

    def _on_classification_worker_done(self, progress, result, on_complete) -> None:
        progress.close()
        self._refresh_tree()
        self._refresh_statistics_panel()
        self._restore_display(preserve_view=True)
        on_complete(result)
        self._auto_save_session()
        self._cleanup_analysis_worker()

    def _on_analysis_worker_error(self, progress, error: str) -> None:
        progress.close()
        self._cleanup_analysis_worker()
        self._log_error(f"Ошибка фонового анализа: {error}")
        self._show_error("Ошибка анализа", error)

    def find_seedlings(self) -> None:
        """Запускает детекцию сеянцев на активной странице в фоне."""
        if not self.image_storage.images:
            self._show_info("Нет данных", "Сначала откройте изображение или PDF.")
            return
        page_index = self._active_image_index

        def on_complete(result) -> None:
            count = result["page_counts"].get(page_index, 0)
            self._log_action(
                "run_analysis",
                f"Детекция завершена для страницы {page_index}; объектов: {count}",
            )
            self.statusBar().showMessage(f"Найдено растений: {count}", 3000)

        self._start_detection_worker(
            [page_index],
            title="Поиск сеянцев",
            on_complete=on_complete,
        )

    def find_all_seedlings(self) -> None:
        """Выполняет детекцию сеянцев на всех страницах в фоне."""
        if not self.image_storage.images:
            self._show_info("Нет данных", "Сначала откройте изображение или PDF.")
            return
        total = len(self.image_storage.images)

        def on_complete(result) -> None:
            processed = result["processed"]
            self._log_action(
                "run_analysis",
                f"Пакетная детекция завершена; обработано {processed} из {total}",
            )
            self.statusBar().showMessage(
                f"Пакетная детекция растений завершена: {processed}/{total}",
                3000,
            )

        self._start_detection_worker(
            list(range(total)),
            title="Поиск сеянцев",
            on_complete=on_complete,
        )

    def find_all_seedlings_new_only(self) -> None:
        """Выполняет детекцию только на страницах без результатов, в фоне."""
        if not self.image_storage.images:
            self._show_info("Нет данных", "Сначала откройте изображение или PDF.")
            return

        pages_to_process = [
            page_index
            for page_index, image in enumerate(self.image_storage.images)
            if not (
                self.image_storage.class_object_image
                and page_index < len(self.image_storage.class_object_image)
                and self.image_storage.class_object_image[page_index]
            )
        ]
        if not pages_to_process:
            self.statusBar().showMessage("Все страницы уже обработаны", 3000)
            return

        total = len(pages_to_process)

        def on_complete(result) -> None:
            processed = result["processed"]
            self._log_action(
                "run_analysis",
                f"Детекция новых страниц завершена; обработано: {processed}/{total}",
            )
            self.statusBar().showMessage(
                f"Поиск завершён: обработано {processed} новых страниц", 3000
            )

        self._start_detection_worker(
            pages_to_process,
            title="Поиск сеянцев (новые)",
            on_complete=on_complete,
        )

    def _segment_single_seedling(
        self,
        page_index: int,
        object_index: int,
        obj: ObjectImage,
    ) -> None:
        if self._analysis_is_running():
            self.statusBar().showMessage("Анализ уже выполняется", 3000)
            return
        if not obj.image:
            self._commit_seedling_bbox(page_index, obj)
        if not obj.image:
            self._show_info("Сегментация", "У выбранного сеянца нет crop-изображения.")
            return

        def on_complete(result) -> None:
            processed = result["processed"]
            self._log_action(
                "run_analysis",
                f"Сегментация выбранного сеянца завершена; обработано: {processed}",
            )
            self.statusBar().showMessage(
                "Маска построена по новому bbox.",
                3000,
            )

        self._start_classification_worker(
            [(page_index, object_index, obj)],
            title="Сегментация выбранного",
            on_complete=on_complete,
        )

    def classify_selected(self) -> None:
        """Runs segmentation again only for the currently selected seedling."""
        context = self._current_crop_context()
        if context is None:
            payload = self.app_state.selected_item or {}
            if payload.get("type") == "seeding" and self.image_storage.class_object_image:
                page_index = int(payload["parent_index"])
                object_index = int(payload["index"])
                objects = self.image_storage.class_object_image[page_index]
                if object_index < len(objects):
                    context = (page_index, object_index, objects[object_index])
        if context is None:
            self._show_info("Сегментация", "Выберите сеянец в дереве или на холсте.")
            return
        self._segment_single_seedling(*context)

    def classify(self) -> None:
        """Классифицирует части растений в найденных объектах в фоне."""
        """Сегментирует части растений внутри найденных объектов проекта."""
        if not self.image_storage.class_object_image or not any(
            self.image_storage.class_object_image
        ):
            self._show_info(
                "Нет детекций",
                "Сначала выполните детекцию растений.",
            )
            return

        pending = [
            (page_index, object_index, obj)
            for page_index, objects in enumerate(
                self.image_storage.class_object_image or []
            )
            for object_index, obj in enumerate(objects)
            if obj.image
        ]
        if not pending:
            self.statusBar().showMessage("Нет объектов для сегментации", 3000)
            return

        def on_complete(result) -> None:
            processed = result["processed"]
            self._log_action(
                "run_analysis",
                f"Сегментация завершена; обработано объектов: {processed}",
            )
            self.statusBar().showMessage("Сегментация завершена", 3000)

        self._start_classification_worker(
            pending,
            title="Сегментация частей",
            on_complete=on_complete,
        )

    def classify_new_only(self) -> None:
        """Сегментирует только сеянцы без существующих результатов, в фоне."""
        if not self.image_storage.class_object_image or not any(
            self.image_storage.class_object_image
        ):
            self._show_info(
                "Нет детекций",
                "Сначала выполните детекцию растений.",
            )
            return

        pending = [
            (page_index, object_index, obj)
            for page_index, objects in enumerate(
                self.image_storage.class_object_image or []
            )
            for object_index, obj in enumerate(objects)
            if obj.image and not obj.image_all_class
        ]
        if not pending:
            self.statusBar().showMessage("Все сеянцы уже сегментированы", 3000)
            return

        total = len(pending)

        def on_complete(result) -> None:
            processed = result["processed"]
            self._log_action(
                "run_analysis",
                f"Сегментация новых объектов завершена; обработано: {processed}/{total}",
            )
            self.statusBar().showMessage(
                f"Сегментация завершена: обработано {processed} новых сеянцев", 3000
            )

        self._start_classification_worker(
            pending,
            title="Сегментация (новые)",
            on_complete=on_complete,
        )
    def rotate_image(self) -> None:
        """Поворачивает текущую страницу или выбранный кроп и обновляет отображение."""
        if not self.image_storage.images:
            return

        selection = self.app_state.selected_item
        if selection is None:
            selection = {"type": "original", "index": self._active_image_index}
        elif selection.get("type") == "class":
            selection = {
                "type": "seeding",
                "parent_index": int(selection["parent_index"]),
                "index": int(selection["seeding_index"]),
            }

        result = rotate_selection(
            self.app_state,
            selection,
            angle=ROTATE_ANGLE_DEG,
            rotate_k=ROTATE_K,
        )
        if result is None:
            return

        self._refresh_tree()
        self._refresh_statistics_panel()
        if result.target == "crop" and result.crop_index is not None:
            self.display_image_with_boxes(result.page_index, result.crop_index)
        else:
            self.display_image_with_boxes(result.page_index)
        self.statusBar().showMessage("Изображение повернуто", 2000)

    def _default_report_dir(self) -> str:
        """Подбирает каталог по умолчанию для сохранения итогового PDF-отчёта."""
        if self._report_output_dir and os.path.isdir(self._report_output_dir):
            return self._report_output_dir
        current_source = self._source_file(self._active_image_index)
        if current_source:
            source_dir = os.path.dirname(current_source)
            if source_dir and os.path.isdir(source_dir):
                return source_dir
        return os.getcwd()

    def create_report(self) -> None:
        """Запрашивает путь сохранения и создаёт PDF-отчёт по текущему проекту."""
        if not self.image_storage.images:
            self._show_info("Нет данных", "Сначала откройте изображение или PDF.")
            return

        default_dir = self._default_report_dir()
        default_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        initial_path = os.path.join(default_dir, default_name)
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить PDF-отчёт",
            initial_path,
            "PDF Files (*.pdf)",
        )
        if not output_path:
            return
        if not output_path.lower().endswith(".pdf"):
            output_path += ".pdf"

        try:
            saved_path = generate_report(
                self.app_state,
                output_path,
            )
            self.app_state.last_report_path = saved_path
            self._auto_save_session()
            self._log_action(
                "generate_report",
                f"Сформирован PDF-отчёт: {saved_path}",
            )
            self._show_info("Отчёт создан", f"Отчёт сохранён:\n{saved_path}")
        except Exception as error:
            logger.exception("Ошибка создания отчёта")
            self._log_error(
                f"Не удалось создать PDF-отчёт {output_path}. Ошибка: {error}"
            )
            self._show_error(
                "Ошибка отчёта",
                f"Не удалось создать PDF-отчёт:\n{error}",
            )

    def _on_save(self) -> None:
        """Сохраняет текущую сессию по запросу пользователя."""
        try:
            save_current_session(self.app_state, self.current_user)
            self.statusBar().showMessage("Сессия сохранена.", 3000)
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка сохранения", str(exc))

    def _open_session_dialog(self) -> None:
        """Открывает диалог выбора сохранённой сессии и восстанавливает её."""
        user_id = self.app_state.current_user_id
        if user_id is None:
            self._show_info("Нет пользователя", "Войдите в систему чтобы открыть сессию.")
            return

        dlg = SessionPickerDialog(user_id=user_id, parent=self)
        if dlg.exec_() != SessionPickerDialog.Accepted:
            return

        session_id = dlg.selected_session_id()
        if session_id is None:
            return

        try:
            state = load_session(session_id)
        except Exception as exc:
            self._show_error("Ошибка загрузки сессии", str(exc))
            return

        if state is None:
            self._show_error("Сессия не найдена", f"Сессия {session_id} не найдена в БД.")
            return

        missing = getattr(state, "_missing_source", None)
        loaded_missing = self._load_session_source_images(state)
        if loaded_missing:
            missing = loaded_missing
        if self.current_user is not None:
            state.current_user_id = self.current_user.id

        self.app_state = state
        self.image_storage = state.image_storage
        self.pixels_per_mm = state.pixels_per_mm

        self._refresh_tree()
        self._refresh_statistics_panel()
        self._restore_display(preserve_view=False)

        if missing:
            self._show_info(
                "Файл не найден",
                f"Исходный файл не найден:\n{missing}\n\n"
                "Данные анализа восстановлены, изображения недоступны.",
            )
        else:
            self.statusBar().showMessage(
                f"Сессия восстановлена: {state.image_storage.file_path}", 4000
            )

    def _auto_save_session(self) -> None:
        """Автоматически сохраняет сессию после инференса."""
        if not self._auto_save_enabled:
            return
        auto_save_current_session(self.app_state, self.current_user, logger=logger)
