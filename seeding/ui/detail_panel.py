"""Панель детального просмотра выбранного сеянца или части растения."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class SeedlingDetailPanel(QWidget):
    """Показывает crop выбранного объекта с маской и кнопки Пред/След."""

    navigate = pyqtSignal(int)  # -1 = пред, +1 = след

    PREVIEW_SIZE = (130, 110)  # ширина, высота превью в px

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.name_label = QLabel("—", self)
        self.name_label.setObjectName("panelSubTitle")
        self.name_label.setWordWrap(True)
        layout.addWidget(self.name_label)

        self.crop_label = QLabel(self)
        self.crop_label.setAlignment(Qt.AlignCenter)
        self.crop_label.setFixedHeight(self.PREVIEW_SIZE[1])
        self.crop_label.setStyleSheet(
            "background:#111; border-radius:3px; border:1px solid #253545"
        )
        layout.addWidget(self.crop_label)

        self.info_label = QLabel("—", self)
        self.info_label.setObjectName("panelHint")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Пред", self)
        self.prev_btn.setProperty("variant", "secondary")
        self.prev_btn.clicked.connect(lambda: self.navigate.emit(-1))
        self.next_btn = QPushButton("След ▶", self)
        self.next_btn.setProperty("variant", "secondary")
        self.next_btn.clicked.connect(lambda: self.navigate.emit(1))
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        layout.addLayout(nav)

    def clear(self) -> None:
        """Сбрасывает панель в пустое состояние."""
        self.name_label.setText("—")
        self.crop_label.clear()
        self.info_label.setText("—")

    def set_object(
        self,
        name: str,
        confidence: float | None,
        manual: bool,
        crop_image: np.ndarray | None,
        mask_bitmap: np.ndarray | None,
        mask_color: tuple[int, int, int],
        bbox: tuple[int, int, int, int] | None,
        pixels_per_mm: float,
    ) -> None:
        conf_text = "ручной" if manual else (
            f"{confidence:.2f}" if confidence is not None else "—"
        )
        self.name_label.setText(f"{name}  conf: {conf_text}")

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            w_px, h_px = max(0, x2 - x1), max(0, y2 - y1)
            if pixels_per_mm > 0:
                w_mm = w_px / pixels_per_mm
                h_mm = h_px / pixels_per_mm
                size_text = f"{w_mm:.1f} × {h_mm:.1f} mm"
            else:
                size_text = f"{w_px} × {h_px} px"
        else:
            size_text = "—"
        self.info_label.setText(size_text)

        self._set_crop_preview(crop_image, mask_bitmap, mask_color)

    def _set_crop_preview(
        self,
        crop: np.ndarray | None,
        mask: np.ndarray | None,
        mask_color: tuple[int, int, int],
    ) -> None:
        if crop is None or not isinstance(crop, np.ndarray):
            self.crop_label.clear()
            return

        preview = crop.copy()
        if preview.ndim == 2:
            preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

        if mask is not None and mask.shape[:2] == preview.shape[:2]:
            overlay = np.zeros_like(preview)
            overlay[:] = mask_color[::-1]  # RGB→BGR
            mask_bool = mask > 0
            alpha = 0.4
            preview[mask_bool] = (
                preview[mask_bool] * (1 - alpha) + overlay[mask_bool] * alpha
            ).astype(np.uint8)

        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3,
                      QImage.Format_RGB888).copy()
        pw, ph = self.PREVIEW_SIZE
        pix = QPixmap.fromImage(qimg).scaled(
            pw, ph, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.crop_label.setPixmap(pix)
