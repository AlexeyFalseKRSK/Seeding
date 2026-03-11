"""Thumbnail list for quick page navigation."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout, QWidget


class ThumbnailsPanel(QWidget):
    image_selected = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.list_widget = QListWidget(self)
        self.list_widget.setObjectName("thumbnailList")
        self.list_widget.setViewMode(QListWidget.IconMode)
        self.list_widget.setResizeMode(QListWidget.Adjust)
        self.list_widget.setMovement(QListWidget.Static)
        self.list_widget.setSpacing(10)
        self.list_widget.setIconSize(QSize(88, 88))
        self.list_widget.setWordWrap(True)
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)

    def set_images(self, images: list[np.ndarray]) -> None:
        self.list_widget.clear()
        for idx, image in enumerate(images):
            item = QListWidgetItem(self._build_icon(image), str(idx + 1))
            item.setData(Qt.UserRole, idx)
            item.setToolTip(f"Изображение {idx + 1}")
            self.list_widget.addItem(item)

    def set_active_index(self, index: int) -> None:
        if 0 <= index < self.list_widget.count():
            self.list_widget.setCurrentRow(index)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        self.image_selected.emit(int(item.data(Qt.UserRole)))

    @staticmethod
    def _build_icon(image: np.ndarray) -> QIcon:
        if image is None or not isinstance(image, np.ndarray):
            pix = QPixmap(88, 88)
            pix.fill(Qt.transparent)
            return QIcon(pix)

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

        pixmap = QPixmap.fromImage(q_image).scaled(
            88,
            88,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        return QIcon(pixmap)
