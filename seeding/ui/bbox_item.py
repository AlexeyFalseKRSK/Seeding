"""Интерактивный прямоугольник для отображения ограничивающей рамки.

Поддерживает изменение размера, перемещение и цветовую индикацию
по уровню уверенности детекции (зелёный / оранжевый / красный).
"""

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QStyleOptionGraphicsItem,
    QWidget,
)

import seeding.config as cfg


def get_color_by_confidence(conf: float) -> QColor:
    """Возвращает цвет рамки по уровню уверенности."""
    if conf >= cfg.CONF_THRESHOLD_HIGH:
        return QColor(Qt.green)
    if conf >= cfg.CONF_THRESHOLD_LOW:
        return QColor(*cfg.BBOX_COLOR_ORANGE_RGB)
    return QColor(Qt.red)


class BBoxItem(QGraphicsRectItem):
    """
    Универсальный интерактивный прямоугольник.
    Связывается с объектами ObjectImage (сеянцы) или AllClassImage (части).
    """

    HANDLE_SIZE = 8.0  # Размер ручек изменения размера (px)

    def __init__(
        self,
        rect: QRectF,
        obj,
        parent: QGraphicsItem | None = None,
        offset=(0, 0),
        bbox_update_callback=None,
        class_label: str | None = None,
        pixels_per_mm: float = 0.0,
    ):
        """Создаёт графический элемент bbox и связывает его с моделью данных.

        Параметры:
            rect: исходный прямоугольник в координатах сцены.
            obj: объект доменной модели, у которого будет обновляться ``bbox``.
            parent: родительский графический элемент.
            offset: смещение локальной области относительно исходного изображения.
        """
        super().__init__(rect, parent)
        self.obj = obj
        self.offset = offset
        self._bbox_update_callback = bbox_update_callback
        self._class_label = (
            str(class_label).strip()
            if class_label is not None
            else str(getattr(obj, "class_name", "")).strip()
        )
        try:
            self._pixels_per_mm = max(float(pixels_per_mm), 0.0)
        except (TypeError, ValueError):
            self._pixels_per_mm = 0.0

        color = get_color_by_confidence(getattr(obj, "confidence", 0.0))
        self.setPen(QPen(color, cfg.UI_BBOX_PEN_WIDTH))

        self.setFlags(
            QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self._editable = False
        self._highlighted = False
        self._handle = None
        self._handles = {}
        self._update_handles()

    def _update_handles(self) -> None:
        """Пересчет координат ручек изменения размера."""
        r = self.rect()
        s = self.HANDLE_SIZE
        self._handles = {
            "tl": QRectF(r.x() - s / 2, r.y() - s / 2, s, s),
            "tr": QRectF(r.right() - s / 2, r.y() - s / 2, s, s),
            "bl": QRectF(r.x() - s / 2, r.bottom() - s / 2, s, s),
            "br": QRectF(r.right() - s / 2, r.bottom() - s / 2, s, s),
        }

    def setEditable(self, state: bool) -> None:
        """Включает или выключает режим редактирования рамки."""
        self._editable = state
        self.setFlag(QGraphicsItem.ItemIsMovable, state)
        self.setFlag(QGraphicsItem.ItemIsSelectable, state)
        if not state:
            self.setSelected(False)
        self.update()

    def setHighlighted(self, state: bool) -> None:
        """Включает или выключает визуальное выделение bbox."""
        self._highlighted = bool(state)
        self.update()

    def paint(
        self,
        painter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ):
        """Отрисовывает рамку и маркеры редактирования в активном режиме."""
        super().paint(painter, option, widget)
        if self._highlighted:
            painter.save()
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255), cfg.UI_BBOX_PEN_WIDTH + 1))
            painter.drawRect(self.rect())
            painter.restore()
        self._draw_overlay_labels(painter)
        if self._editable:
            painter.setBrush(Qt.white)
            painter.setPen(QPen(Qt.black, 1))
            for handle_rect in self._handles.values():
                painter.drawRect(handle_rect)

    def mousePressEvent(self, event):
        """Определяет активную ручку изменения размера при клике."""
        if self._editable:
            for name, rect in self._handles.items():
                if rect.contains(event.pos()):
                    self._handle = name
                    break
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Обрабатывает перетаскивание рамки или ручек редактирования."""
        if self._editable and self._handle:
            r = self.rect()
            pos = event.pos()
            if self._handle == "tl":
                r.setTopLeft(pos)
            elif self._handle == "tr":
                r.setTopRight(pos)
            elif self._handle == "bl":
                r.setBottomLeft(pos)
            elif self._handle == "br":
                r.setBottomRight(pos)
            self.setRect(r)
        else:
            super().mouseMoveEvent(event)

        if self._editable:
            self._update_handles()
            self.update_bbox()

    def mouseReleaseEvent(self, event):
        """Завершает операцию редактирования и фиксирует новые координаты."""
        super().mouseReleaseEvent(event)
        self._handle = None
        if self._editable:
            self._update_handles()
            self.update_bbox()

    def update_bbox(self) -> None:
        """Обновляет координаты bbox в связанном объекте данных."""
        r = self.rect().normalized()
        ox, oy = self.offset
        new_bbox = (
            int(r.left() + ox),
            int(r.top() + oy),
            int(r.right() + ox),
            int(r.bottom() + oy),
        )
        if self._bbox_update_callback is not None:
            transformed_bbox = self._bbox_update_callback(new_bbox)
            if transformed_bbox is not None:
                new_bbox = transformed_bbox
        self.obj.bbox = new_bbox

    def _build_size_label(self) -> str:
        """Возвращает подпись размера bbox в мм (если калибровано) или в px."""
        rect = self.rect().normalized()
        width_px = max(0.0, float(rect.width()))
        height_px = max(0.0, float(rect.height()))
        if self._pixels_per_mm > 0:
            width_mm = width_px / self._pixels_per_mm
            height_mm = height_px / self._pixels_per_mm
            return f"{width_mm:.1f}/{height_mm:.1f} mm"
        return f"{int(round(width_px))}/{int(round(height_px))} px"

    def _draw_overlay_labels(self, painter) -> None:
        """Рисует подписи класса и размера поверх bbox."""
        rect = self.rect().normalized()
        if rect.width() <= 1 or rect.height() <= 1:
            return

        class_text = self._class_label or "Object"
        size_text = self._build_size_label()

        painter.save()
        font = painter.font()
        if font.pointSize() > 0:
            font.setPointSize(
                max(cfg.UI_BBOX_LABEL_FONT_POINT_SIZE, font.pointSize())
            )
        else:
            font.setPixelSize(cfg.UI_BBOX_LABEL_FONT_PIXEL_SIZE)
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()

        def draw_badge(text: str, x: float, y: float) -> QRectF:
            text_width = fm.horizontalAdvance(text)
            badge_w = float(text_width + 10)
            badge_h = float(fm.height() + 4)
            badge_rect = QRectF(x, y, badge_w, badge_h)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(10, 18, 26, 205))
            painter.drawRoundedRect(badge_rect, 3, 3)
            painter.setPen(QColor(245, 250, 255))
            painter.drawText(
                badge_rect.adjusted(5, 0, -5, 0),
                Qt.AlignLeft | Qt.AlignVCenter,
                text,
            )
            return badge_rect

        top_badge = draw_badge(class_text, rect.left() + 2, rect.top() + 2)
        bottom_y = max(rect.top() + 2, rect.bottom() - top_badge.height() - 2)
        draw_badge(size_text, rect.left() + 2, bottom_y)
        painter.restore()
