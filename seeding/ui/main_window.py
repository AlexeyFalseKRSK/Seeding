"""Главное окно приложения Seeding с загрузкой, анализом и просмотром результатов."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np
from PyQt5.QtCore import QEvent, QPoint, QPointF, QRectF, QSettings, QSize, Qt
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
)
from PyQt5.QtWidgets import (
    QAction,
    QFrame,
    QFileDialog,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QShortcut,
    QSplitter,
    QStackedLayout,
    QStyle,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from seeding.config import (
    CALIBRATION_PIXELS_PER_MM_DEFAULT,
    DETECTION_CLASS_NAME,
    DETECTION_CONFIDENCE_THRESHOLD,
    DETECTION_IOU_THRESHOLD,
    DEFAULT_CLASSIFY_WEIGHTS_PATH,
    PANEL_LAYERS_MAX_WIDTH,
    PANEL_LAYERS_MIN_WIDTH,
    PDF_RENDER_SCALE_DEFAULT,
    QSETTINGS_APP,
    QSETTINGS_ORG,
    ROTATE_ANGLE_DEG,
    ROTATE_K,
    SPLITTER_SIZES,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    WINDOW_X,
    WINDOW_Y,
    ZOOM_FACTOR_INCREMENT,
)
from seeding.controllers import AppController
from seeding.inference import InferenceBackend, load_inference_backend
from seeding.models import AllClassImage, AppState, ObjectImage, OriginalImage
from seeding.ui.bbox_item import BBoxItem
from seeding.ui.icon_manager import IconManager
from seeding.ui.statistics_panel import StatisticsPanel
from seeding.ui.thumbnails_panel import ThumbnailsPanel
from seeding.ui.tree_widget import LayerTreeWidget
from seeding.utils import clip_bbox_to_image, rotate_bbox

logger = logging.getLogger(__name__)

INPUT_FILE_FILTER = (
    "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;"
    "PDF Files (*.pdf);;"
    "All Files (*)"
)


class CanvasGraphicsView(QGraphicsView):
    """Графический viewport с поддержкой прокрутки средней кнопкой мыши."""

    def __init__(self, scene: QGraphicsScene, parent=None) -> None:
        """Создаёт viewport и подготавливает поля для ручного перемещения холста."""
        super().__init__(scene, parent)
        self._drag_active = False
        self._drag_start_pos = QPoint()
        self._scroll_start_pos = QPoint()

    def mousePressEvent(self, event) -> None:
        """Включает режим перетаскивания холста при нажатии средней кнопки мыши."""
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
        if self._drag_active:
            delta = event.pos() - self._drag_start_pos
            self.horizontalScrollBar().setValue(self._scroll_start_pos.x() - delta.x())
            self.verticalScrollBar().setValue(self._scroll_start_pos.y() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """Завершает ручное перемещение холста после отпускания средней кнопки."""
        if event.button() == Qt.MiddleButton:
            self._drag_active = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ImageEditor(QMainWindow):
    """Главное окно приложения с загрузкой файлов, анализом и просмотром результатов."""

    def __init__(
        self,
        weights_path: str,
        classify_weights_path: str | None = None,
    ) -> None:
        """Инициализирует состояние окна, модели, настройки и основные элементы UI."""
        super().__init__()
        self.weights_path = str(weights_path)
        self.classify_weights_path = str(
            classify_weights_path or DEFAULT_CLASSIFY_WEIGHTS_PATH
        )
        self.detect_model: InferenceBackend | None = None
        self.classify_model: InferenceBackend | None = None
        self.app_controller = AppController()
        self.app_state = AppState(image_storage=OriginalImage())
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
        self.zoom_factor = 1.0
        self.min_fit_zoom = 1.0
        self.pixels_per_mm = CALIBRATION_PIXELS_PER_MM_DEFAULT
        self._measure_mode = False
        self._calibration_pending = False
        self._measure_start_scene_pos: QPointF | None = None
        self._measure_line_item: QGraphicsLineItem | None = None
        self._measure_text_item: QGraphicsTextItem | None = None
        self._settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        self.app_state.zoom_factor = self.zoom_factor
        self.app_state.pixels_per_mm = self.pixels_per_mm

        self.setWindowTitle("Seeding")
        self.setGeometry(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)

        self.icon_manager = IconManager(self)
        self._build_ui()
        self._build_toolbar()
        self._build_menu()
        QShortcut(QKeySequence("M"), self, self.toggle_measure_mode)
        self.statusBar().showMessage("Откройте изображение или PDF", 3000)

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

        self.project_files_list = QListWidget(left_panel)
        self.project_files_list.setObjectName("projectFilesList")
        self.project_files_list.setMinimumHeight(240)
        self.project_files_list.currentRowChanged.connect(
            self._on_project_row_changed
        )
        left_layout.addWidget(self.project_files_list, 1)

        interaction_card = QFrame(left_panel)
        interaction_card.setObjectName("imageInteractionCard")
        interaction_layout = QVBoxLayout(interaction_card)
        interaction_layout.setContentsMargins(12, 12, 12, 12)
        interaction_layout.setSpacing(10)

        interaction_title = QLabel("Взаимодействие с изображением", interaction_card)
        interaction_title.setObjectName("panelSubTitle")
        interaction_title.setWordWrap(True)
        interaction_layout.addWidget(interaction_title)

        interaction_hint = QLabel(
            "Управление отображением слоев на изображении.",
            interaction_card,
        )
        interaction_hint.setObjectName("panelHint")
        interaction_hint.setWordWrap(True)
        interaction_layout.addWidget(interaction_hint)

        layers_title = QLabel("Отображение", interaction_card)
        layers_title.setObjectName("panelHint")
        interaction_layout.addWidget(layers_title)

        toggle_bar = QFrame(interaction_card)
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

        interaction_layout.addWidget(toggle_bar)

        mode_title = QLabel("Режим работы", interaction_card)
        mode_title.setObjectName("panelHint")
        interaction_layout.addWidget(mode_title)

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
        self.edit_boxes_mode_button.setEnabled(False)
        self.edit_boxes_mode_button.setToolTip("Режим будет добавлен позже.")
        mode_bar_layout.addWidget(self.edit_boxes_mode_button)

        self.edit_masks_mode_button = QPushButton("Ред. маски", mode_bar)
        self.edit_masks_mode_button.setCheckable(True)
        self.edit_masks_mode_button.setProperty("segmented", "true")
        self.edit_masks_mode_button.setEnabled(False)
        self.edit_masks_mode_button.setToolTip("Режим будет добавлен позже.")
        mode_bar_layout.addWidget(self.edit_masks_mode_button)

        interaction_layout.addWidget(mode_bar)
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
        self.graphics_view = CanvasGraphicsView(self.graphics_scene, self)
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
        self.thumbnails_panel = ThumbnailsPanel(self)
        self.thumbnails_panel.image_selected.connect(self._select_page)

        self.right_tabs.addTab(self.tree_widget, "Слои")
        self.right_tabs.addTab(self.statistics_panel, "Статистика")
        right_layout.addWidget(self.right_tabs, 1)
        splitter.addWidget(right_panel)
        splitter.setSizes(SPLITTER_SIZES)
        self._set_canvas_empty(True)
        self._update_canvas_status()

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
            "После загрузки можно найти сеянцы, классифицировать части и "
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
            self.sidebar_page_label.setText("Нет выбранной страницы")
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
        self.sidebar_page_label.setText(page_text)
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
        if watched is self.graphics_view.viewport():
            if event.type() == QEvent.Wheel and event.modifiers() & Qt.ControlModifier:
                if event.angleDelta().y() > 0:
                    self.zoom_in(anchor_view_pos=event.pos())
                elif event.angleDelta().y() < 0:
                    self.zoom_out(anchor_view_pos=event.pos())
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
        self._apply_interaction_mode_to_rect_items()

    def _apply_interaction_mode_to_rect_items(self) -> None:
        """Применяет текущий режим редактирования ко всем рамкам на сцене."""
        editable = self._interaction_mode == "edit_boxes"
        for item in self.rect_items:
            item.setEditable(editable)

    def _build_toolbar(self) -> None:
        """Создаёт верхний toolbar с основными действиями приложения."""
        toolbar = QToolBar("Main", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
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
        self.action_find = self._create_action(
            "action_detect.svg",
            "Найти",
            self.find_seedlings,
            shortcut="Ctrl+F",
            fallback_standard_icon=QStyle.SP_MediaPlay,
        )
        self.action_find_all = self._create_action(
            "action_detect_all.svg",
            "Найти на всех",
            self.find_all_seedlings,
            shortcut="Ctrl+Shift+F",
            fallback_standard_icon=QStyle.SP_BrowserReload,
        )
        self.action_classify = self._create_action(
            "action_classify.svg",
            "Классифицировать",
            self.classify,
            shortcut="Ctrl+C",
            fallback_standard_icon=QStyle.SP_FileDialogDetailedView,
        )
        self.action_rotate = self._create_action(
            "action_rotate.svg",
            "Повернуть",
            self.rotate_image,
            shortcut="Ctrl+R",
            fallback_standard_icon=QStyle.SP_BrowserReload,
        )
        self.action_report = self._create_action(
            "action_report.svg",
            "Отчёт",
            self.create_report,
            shortcut="Ctrl+P",
            fallback_standard_icon=QStyle.SP_FileDialogContentsView,
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
        toolbar.addAction(self.action_find_all)
        toolbar.addAction(self.action_classify)
        toolbar.addAction(self.action_rotate)
        toolbar.addSeparator()
        toolbar.addAction(self.action_report)
        toolbar.addSeparator()
        toolbar.addAction(self.action_zoom_in)
        toolbar.addAction(self.action_zoom_out)
        toolbar.addAction(self.action_fit)

    def _build_menu(self) -> None:
        """Создаёт верхнее меню и раскладывает по нему основные действия окна."""
        file_menu = self.menuBar().addMenu("Файл")
        file_menu.addAction(self.action_open)
        file_menu.addAction(self.action_add)
        file_menu.addSeparator()
        file_menu.addAction(self.action_report)

        analysis_menu = self.menuBar().addMenu("Анализ")
        analysis_menu.addAction(self.action_find)
        analysis_menu.addAction(self.action_find_all)
        analysis_menu.addAction(self.action_classify)
        analysis_menu.addAction(self.action_rotate)

        view_menu = self.menuBar().addMenu("Вид")
        view_menu.addAction(self.action_zoom_in)
        view_menu.addAction(self.action_zoom_out)
        view_menu.addAction(self.action_fit)

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

    def _show_error(self, title: str, text: str) -> None:
        """Показывает модальное сообщение об ошибке."""
        QMessageBox.critical(self, title, text)

    def _show_info(self, title: str, text: str) -> None:
        """Показывает модальное информационное сообщение."""
        QMessageBox.information(self, title, text)

    def _ensure_detection_storage(self) -> None:
        """Гарантирует наличие списка детекций для всех загруженных страниц."""
        if self.image_storage.class_object_image is None:
            self.image_storage.class_object_image = []
        while len(self.image_storage.class_object_image) < len(
            self.image_storage.images
        ):
            self.image_storage.class_object_image.append([])

    def _ensure_detect_model(self) -> InferenceBackend | None:
        """Лениво загружает модель детекции и возвращает её экземпляр."""
        if self.detect_model is not None:
            return self.detect_model
        try:
            self.detect_model = load_inference_backend(self.weights_path)
            return self.detect_model
        except Exception as error:
            logger.exception("Не удалось загрузить модель детекции")
            self._show_error(
                "Ошибка модели детекции",
                f"Не удалось загрузить модель:\n{self.weights_path}\n\n{error}",
            )
            return None

    def _ensure_classify_model(self) -> InferenceBackend | None:
        """Лениво загружает модель классификации частей и возвращает её экземпляр."""
        if self.classify_model is not None:
            return self.classify_model
        try:
            self.classify_model = load_inference_backend(
                self.classify_weights_path
            )
            return self.classify_model
        except Exception as error:
            logger.exception("Не удалось загрузить модель классификации")
            self._show_error(
                "Ошибка модели классификации",
                (
                    "Не удалось загрузить модель:\n"
                    f"{self.classify_weights_path}\n\n{error}"
                ),
            )
            return None

    def clear_project(self) -> None:
        """Полностью очищает проект, состояние окна и текущие результаты анализа."""
        self.app_state = AppState(image_storage=OriginalImage())
        self.image_storage = self.app_state.image_storage
        self._active_image_index = 0
        self._display_target = ("page", 0)
        self.zoom_factor = 1.0
        self.min_fit_zoom = 1.0
        self.app_state.zoom_factor = self.zoom_factor
        self.project_files_list.clear()
        self.tree_widget.clear()
        self.thumbnails_panel.set_images([])
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
        self._refresh_thumbnails_panel()

    def _load_path(self, file_path: str) -> list[int]:
        """Выбирает способ загрузки файла по его расширению."""
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(file_path)
        return self._load_image(file_path)

    def _load_image(self, file_path: str) -> list[int]:
        """Загружает одно изображение с диска и добавляет его как страницу проекта."""
        image = cv2.imread(file_path)
        if image is None:
            self._show_error(
                "Ошибка открытия",
                f"Не удалось открыть изображение:\n{file_path}",
            )
            return []
        index = self._append_page(image, file_path)
        return [index]

    def _load_pdf(self, pdf_path: str) -> list[int]:
        """Рендерит PDF постранично в изображения и добавляет их в проект."""
        pages: list[int] = []
        doc = None
        try:
            doc = fitz.open(pdf_path)
            total = int(doc.page_count)
            progress = QProgressDialog(
                "Загрузка PDF...",
                "Отмена",
                0,
                total,
                self,
            )
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            for page_num in range(total):
                if progress.wasCanceled():
                    break
                page = doc.load_page(page_num)
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(
                        PDF_RENDER_SCALE_DEFAULT,
                        PDF_RENDER_SCALE_DEFAULT,
                    )
                )
                image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height,
                    pix.width,
                    pix.n,
                )
                if pix.n == 4:
                    image = image[:, :, :3].copy()
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                pages.append(self._append_page(image, pdf_path))
                progress.setValue(page_num + 1)

            progress.close()
        except Exception as error:
            logger.exception("Не удалось загрузить PDF %s", pdf_path)
            self._show_error(
                "Ошибка PDF",
                f"Не удалось загрузить PDF:\n{pdf_path}\n\n{error}",
            )
        finally:
            if doc is not None:
                doc.close()
        return pages

    def _append_page(self, image: np.ndarray, source_path: str) -> int:
        """Добавляет страницу в хранилище проекта и список файлов интерфейса."""
        index = len(self.image_storage.images)
        self.image_storage.images.append(image)
        self.image_storage.source_files.append(source_path)
        if not self.image_storage.file_path:
            self.image_storage.file_path = source_path
        self._ensure_detection_storage()

        item = QListWidgetItem(self._page_title(index))
        item.setData(Qt.UserRole, index)
        self.project_files_list.addItem(item)
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

    def _on_project_row_changed(self, row: int) -> None:
        """Реагирует на смену выбранной строки в списке страниц проекта."""
        if row < 0 or row >= len(self.image_storage.images):
            return
        self._select_page(row)

    def _select_page(self, page_index: int) -> None:
        """Делает страницу активной и обновляет связанные элементы интерфейса."""
        if page_index < 0 or page_index >= len(self.image_storage.images):
            return
        self._restore_calibration_for_index(page_index)
        self.display_image_with_boxes(page_index)
        if self.project_files_list.currentRow() != page_index:
            self.project_files_list.blockSignals(True)
            self.project_files_list.setCurrentRow(page_index)
            self.project_files_list.blockSignals(False)
        self.thumbnails_panel.set_active_index(page_index)
        self._refresh_tree()
        self._refresh_statistics_panel()
        self._update_canvas_status()

    def _refresh_tree(self) -> None:
        """Перестраивает дерево слоёв для активной страницы проекта."""
        self.tree_widget.blockSignals(True)
        self.tree_widget.clear()

        if self.image_storage.images:
            page_index = min(
                max(self._active_image_index, 0),
                len(self.image_storage.images) - 1,
            )
            image = self.image_storage.images[page_index]
            root = self.tree_widget.add_root_item(
                name=self._page_title(page_index),
                description=Path(self._source_file(page_index)).name,
                index=page_index,
                image_type="pdf"
                if self._source_file(page_index).lower().endswith(".pdf")
                else "original",
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
                    )
                child.setExpanded(False)
            root.setExpanded(False)

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

    def _refresh_thumbnails_panel(self) -> None:
        """Перестраивает панель миниатюр и выделяет активную страницу."""
        images = [
            image
            for image in self.image_storage.images
            if isinstance(image, np.ndarray)
        ]
        self.thumbnails_panel.set_images(images)
        self.thumbnails_panel.set_active_index(self._active_image_index)

    def _on_tree_selection_changed(self) -> None:
        """Переключает отображение по выбору пользователя в дереве слоёв."""
        item = self.tree_widget.currentItem()
        if item is None:
            return
        payload = item.data(0, Qt.UserRole) or {}
        item_type = payload.get("type")
        self.app_state.selected_item = payload

        if item_type in {"original", "pdf"}:
            self._select_page(int(payload["index"]))
            return
        if item_type == "seeding":
            self.display_image_with_boxes(
                int(payload["parent_index"]),
                seeding_idx=int(payload["index"]),
            )
            return
        if item_type == "class":
            self.display_image_with_boxes(
                int(payload["parent_index"]),
                seeding_idx=int(payload["seeding_index"]),
            )

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
        if value == "root":
            return QColor(52, 199, 89, 80), QColor(52, 199, 89, 210)
        if value == "stem":
            return QColor(255, 159, 10, 80), QColor(255, 159, 10, 210)
        if value in {"flower", "inflorescence"}:
            return QColor(64, 156, 255, 80), QColor(64, 156, 255, 210)
        return QColor(46, 226, 201, 70), QColor(46, 226, 201, 190)

    def _add_part_mask_item(self, part_obj: AllClassImage) -> None:
        """Добавляет на сцену полигон маски классифицированной части растения."""
        polygon = getattr(part_obj, "mask_polygon", None)
        if polygon is None:
            return

        points = np.asarray(polygon, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
            return

        q_polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in points])
        fill_color, outline_color = self._part_mask_colors(part_obj.class_name)
        item = self.graphics_scene.addPolygon(
            q_polygon,
            QPen(outline_color, 2),
            QBrush(fill_color),
        )
        item.setAcceptedMouseButtons(Qt.NoButton)
        item.setZValue(0.5)
        self.mask_items.append(item)

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
        self._display_target = (
            ("page", img_idx) if seeding_idx is None else ("crop", img_idx, seeding_idx)
        )

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
                        class_label="Сеянец",
                        pixels_per_mm=self.pixels_per_mm,
                    )
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
                    class_label=self._display_part_name(part_obj.class_name),
                    pixels_per_mm=self.pixels_per_mm,
                )
                item.setZValue(1.0)
                self.graphics_scene.addItem(item)
                self.rect_items.append(item)

        self._apply_interaction_mode_to_rect_items()
        self.thumbnails_panel.set_active_index(img_idx)
        self._update_canvas_status()

    def _seedling_title(self, object_index: int, obj: ObjectImage) -> str:
        """Возвращает заголовок узла сеянца для дерева слоёв."""
        _ = obj
        return f"Сеянец {object_index + 1}"

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
        mapping = {
            "root": "Корень",
            "stem": "Стебель",
            "flower": "Соцветие",
            "inflorescence": "Соцветие",
            "seeding": "Сеянец",
            "seedling": "Сеянец",
        }
        return mapping.get(value, class_name or "Часть")

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

    def find_seedlings(self) -> None:
        """Запускает детекцию сеянцев на активной странице и обновляет интерфейс."""
        if not self.image_storage.images:
            self._show_info("Нет данных", "Сначала откройте изображение или PDF.")
            return
        model = self._ensure_detect_model()
        if model is None:
            return

        self._ensure_detection_storage()
        image = self.image_storage.images[self._active_image_index]
        results = model.predict(
            image,
            conf_threshold=DETECTION_CONFIDENCE_THRESHOLD,
        )
        objects = self.app_controller.run_detection(
            self.app_state,
            self._active_image_index,
            results,
            detection_class_name=DETECTION_CLASS_NAME,
            iou_threshold=DETECTION_IOU_THRESHOLD,
            rotate_k=ROTATE_K,
        )
        self._refresh_tree()
        self._refresh_statistics_panel()
        self._restore_display(preserve_view=True)
        self.statusBar().showMessage(f"Найдено объектов: {len(objects)}", 3000)

    def find_all_seedlings(self) -> None:
        """Выполняет детекцию сеянцев на всех страницах текущего проекта."""
        if not self.image_storage.images:
            self._show_info("Нет данных", "Сначала откройте изображение или PDF.")
            return
        model = self._ensure_detect_model()
        if model is None:
            return

        self._ensure_detection_storage()
        total = len(self.image_storage.images)
        progress = QProgressDialog(
            "Детекция на всех страницах...",
            "Отмена",
            0,
            total,
            self,
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        processed = 0
        for page_index, image in enumerate(self.image_storage.images):
            if progress.wasCanceled():
                break
            results = model.predict(
                image,
                conf_threshold=DETECTION_CONFIDENCE_THRESHOLD,
            )
            self.app_controller.run_detection(
                self.app_state,
                page_index,
                results,
                detection_class_name=DETECTION_CLASS_NAME,
                iou_threshold=DETECTION_IOU_THRESHOLD,
                rotate_k=ROTATE_K,
            )
            processed += 1
            progress.setValue(processed)

        progress.close()
        self._refresh_tree()
        self._refresh_statistics_panel()
        self._restore_display(preserve_view=True)
        self.statusBar().showMessage(
            f"Пакетная детекция завершена: {processed}/{total}",
            3000,
        )

    def classify(self) -> None:
        """Классифицирует части растений внутри найденных объектов проекта."""
        if not self.image_storage.class_object_image or not any(
            self.image_storage.class_object_image
        ):
            self._show_info(
                "Нет детекций",
                "Сначала выполните детекцию сеянцев.",
            )
            return
        model = self._ensure_classify_model()
        if model is None:
            return

        total_objects = sum(
            len(objects) for objects in self.image_storage.class_object_image or []
        )
        progress = QProgressDialog(
            "Классификация частей...",
            "Отмена",
            0,
            total_objects,
            self,
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        processed = 0
        for page_index, objects in enumerate(
            self.image_storage.class_object_image or []
        ):
            for object_index, obj in enumerate(objects):
                if progress.wasCanceled():
                    progress.close()
                    self._refresh_tree()
                    self._refresh_statistics_panel()
                    self._restore_display(preserve_view=True)
                    return
                if not obj.image:
                    continue
                results = model.predict(obj.image[0])
                self.app_controller.run_classification_for_selection(
                    self.app_state,
                    page_index,
                    object_index,
                    results,
                )
                processed += 1
                progress.setValue(processed)

        progress.close()
        self._refresh_tree()
        self._refresh_statistics_panel()
        self._restore_display(preserve_view=True)
        self.statusBar().showMessage("Классификация завершена", 3000)

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

        result = self.app_controller.rotate_current(
            self.app_state,
            selection,
            angle=ROTATE_ANGLE_DEG,
            rotate_k=ROTATE_K,
        )
        if result is None:
            return

        self._refresh_tree()
        self._refresh_statistics_panel()
        self._refresh_thumbnails_panel()
        if result.target == "crop" and result.crop_index is not None:
            self.display_image_with_boxes(result.page_index, result.crop_index)
        else:
            self.display_image_with_boxes(result.page_index)
        self.statusBar().showMessage("Изображение повернуто", 2000)

    def _default_report_dir(self) -> str:
        """Подбирает каталог по умолчанию для сохранения итогового PDF-отчёта."""
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
            saved_path = self.app_controller.generate_report(
                self.app_state,
                output_path,
            )
            self._show_info("Отчёт создан", f"Отчёт сохранён:\n{saved_path}")
        except Exception as error:
            logger.exception("Ошибка создания отчёта")
            self._show_error(
                "Ошибка отчёта",
                f"Не удалось создать PDF-отчёт:\n{error}",
            )
