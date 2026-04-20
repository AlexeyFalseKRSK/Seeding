# Annotation & Navigation Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Улучшить визуализацию масок, добавить ручное создание боксов и удобную навигацию по сеянцам.

**Architecture:** Три независимых блока: (1) мультиконтурная отрисовка масок через QPainterPath вместо RGBA pixmap, (2) улучшение LayerTreeWidget с иконками/подсветкой + детальная панель в правой части, (3) режим рисования боксов в CanvasGraphicsView с диалогом выбора класса. Порядок: Блок 1 → Блок 3 → Блок 2.

**Tech Stack:** Python 3.13, PyQt5, OpenCV, NumPy, pytest

---

## Task 1: bitmap_to_contours в mask_refiner.py

**Files:**
- Modify: `seeding/mask_refiner.py`
- Test: `tests/test_mask_refiner.py` (новый файл)

- [ ] **Шаг 1: Написать падающий тест**

```python
# tests/test_mask_refiner.py
import numpy as np
import pytest
from seeding.mask_refiner import bitmap_to_contours


def _make_bitmap_with_hole():
    """Кольцо 20x20: внешний квадрат 20x20, внутренний белый квадрат 8x8 в центре."""
    bm = np.zeros((20, 20), dtype=np.uint8)
    bm[2:18, 2:18] = 255   # внешний контур
    bm[6:14, 6:14] = 0     # дыра
    return bm


def test_bitmap_to_contours_returns_list():
    bm = np.zeros((10, 10), dtype=np.uint8)
    bm[1:9, 1:9] = 255
    result = bitmap_to_contours(bm)
    assert isinstance(result, list)
    assert len(result) >= 1
    assert all(isinstance(c, np.ndarray) for c in result)
    assert all(c.ndim == 2 and c.shape[1] == 2 for c in result)


def test_bitmap_to_contours_detects_hole():
    bm = _make_bitmap_with_hole()
    result = bitmap_to_contours(bm)
    # Должно быть >= 2 контуров: внешний + дыра
    assert len(result) >= 2


def test_bitmap_to_contours_empty_returns_empty():
    assert bitmap_to_contours(np.zeros((10, 10), dtype=np.uint8)) == []
    assert bitmap_to_contours(None) == []
```

- [ ] **Шаг 2: Запустить тест, убедиться что падает**

```
python -m pytest tests/test_mask_refiner.py -v
```
Ожидается: `ImportError: cannot import name 'bitmap_to_contours'`

- [ ] **Шаг 3: Реализовать bitmap_to_contours**

В `seeding/mask_refiner.py` добавить после функции `bitmap_to_polygon` (строка ~52):

```python
def bitmap_to_contours(
    binary_mask: np.ndarray | None,
) -> list[np.ndarray]:
    """Возвращает все контуры маски — внешние + дыры — через RETR_CCOMP.

    Каждый контур — np.ndarray Nx2 float32 в координатах маски.
    Используется для QPainterPath с OddEvenFill: дыры прозрачны.
    """
    if binary_mask is None or binary_mask.size == 0:
        return []
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    result = []
    for contour in (contours or []):
        if len(contour) >= 3:
            result.append(
                np.ascontiguousarray(contour.reshape(-1, 2).astype(np.float32))
            )
    return result
```

В `__all__` в конце файла добавить `"bitmap_to_contours"`.

- [ ] **Шаг 4: Запустить тест, убедиться что проходит**

```
python -m pytest tests/test_mask_refiner.py -v
```
Ожидается: 4 passed

- [ ] **Шаг 5: Коммит**

```bash
git add seeding/mask_refiner.py tests/test_mask_refiner.py
git commit -m "feat: add bitmap_to_contours for multi-contour mask rendering"
```

---

## Task 2: QPainterPath-рендеринг масок в main_window.py

**Files:**
- Modify: `seeding/ui/main_window.py`

- [ ] **Шаг 1: Добавить импорты**

В `seeding/ui/main_window.py` в блок импортов PyQt5 добавить `QPainterPath`, `QGraphicsPathItem`:

```python
# В строке ~26 рядом с другими импортами из PyQt5.QtGui:
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,       # добавить
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
)
```

И в импортах из PyQt5.QtWidgets добавить `QGraphicsPathItem`:

```python
from PyQt5.QtWidgets import (
    ...
    QGraphicsPathItem,   # добавить
    ...
)
```

Также добавить импорт `bitmap_to_contours` рядом с существующим импортом из `seeding.mask_refiner` — его сейчас нет, добавить новый import:

```python
from seeding.mask_refiner import bitmap_to_contours
```

- [ ] **Шаг 2: Переписать `_add_bitmap_mask_item`**

Найти метод `_add_bitmap_mask_item` (строка ~2087) и заменить его целиком:

```python
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
    h, w = bitmap.shape[:2]
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
        path.addPath(sub)

    path_item = QGraphicsPathItem(path)
    path_item.setPen(QPen(outline_color, 1.5))
    path_item.setBrush(QBrush(fill_color))
    path_item.setAcceptedMouseButtons(Qt.NoButton)
    path_item.setZValue(0.5)
    self.graphics_scene.addItem(path_item)
    self.mask_items.append(path_item)
```

- [ ] **Шаг 3: Запустить все тесты**

```
python -m pytest tests/ -q
```
Ожидается: все тесты проходят (33 + 4 новых = 37 passed)

- [ ] **Шаг 4: Коммит**

```bash
git add seeding/ui/main_window.py
git commit -m "feat: render masks via QPainterPath with OddEvenFill for correct hole visualization"
```

---

## Task 3: Иконки и подсветка в LayerTreeWidget

**Files:**
- Modify: `seeding/ui/tree_widget.py`
- Modify: `seeding/ui/main_window.py` (передача флага manual)
- Test: `tests/test_tree_widget.py` (добавить тесты)

- [ ] **Шаг 1: Написать падающие тесты**

Добавить в `tests/test_tree_widget.py`:

```python
def test_low_confidence_child_has_warning_background():
    """Сеянец с confidence < 0.5 должен иметь ненулевой цвет фона."""
    app, created = _ensure_offscreen_qt()
    tree = LayerTreeWidget()
    root = tree.add_root_item("p", "d", 0, "original", None)
    child = tree.add_child_item(root, "S", "d", 0, 0, "seeding", None, confidence=0.3)

    bg = child.background(0)
    # Фон должен быть установлен (не дефолтный прозрачный)
    assert bg.color().alpha() > 0

    tree.deleteLater()
    if created:
        app.quit()


def test_normal_confidence_child_has_no_warning_background():
    """Сеянец с confidence >= 0.5 не должен иметь предупреждающего фона."""
    app, created = _ensure_offscreen_qt()
    tree = LayerTreeWidget()
    root = tree.add_root_item("p", "d", 0, "original", None)
    child = tree.add_child_item(root, "S", "d", 0, 0, "seeding", None, confidence=0.8)

    bg = child.background(0)
    assert bg.color().alpha() == 0  # дефолтный — прозрачный

    tree.deleteLater()
    if created:
        app.quit()


def test_manual_item_has_manual_prefix():
    """Ручной объект должен иметь префикс ✏ в имени."""
    app, created = _ensure_offscreen_qt()
    tree = LayerTreeWidget()
    root = tree.add_root_item("p", "d", 0, "original", None)
    child = tree.add_child_item(
        root, "Сеянец 1", "d", 0, 0, "seeding", None,
        confidence=1.0, manual=True
    )
    assert "✏" in child.text(0)

    tree.deleteLater()
    if created:
        app.quit()
```

- [ ] **Шаг 2: Запустить, убедиться что падают**

```
python -m pytest tests/test_tree_widget.py -v
```
Ожидается: 3 новых теста падают (TypeError или AssertionError)

- [ ] **Шаг 3: Обновить `add_child_item` в tree_widget.py**

Заменить сигнатуру и тело метода `add_child_item`:

```python
def add_child_item(
    self,
    parent: QTreeWidgetItem,
    name: str,
    description: str,
    parent_index: int,
    index: int,
    image_type: str,
    image,
    confidence: float | None = None,
    manual: bool = False,
) -> QTreeWidgetItem:
    """Добавляет узел сеянца с иконкой, подсветкой низкого confidence и флагом ручного объекта."""
    _ = (image_type, image)
    child = QTreeWidgetItem(parent)
    prefix = "✏ " if manual else "🌱 "
    child.setText(0, prefix + name)
    child.setText(1, description)
    child.setData(
        0,
        Qt.UserRole,
        {"type": "seeding", "parent_index": parent_index, "index": index},
    )
    if confidence is not None:
        child.setData(1, self.CONFIDENCE_ROLE, float(confidence))
        if float(confidence) < 0.5:
            warn_color = QColor(80, 40, 10, 180)
            child.setBackground(0, warn_color)
            child.setBackground(1, warn_color)
            child.setForeground(0, QColor(255, 150, 80))
            child.setText(0, "⚠ " + prefix + name)
    parent.addChild(child)
    return child
```

Для `add_class_item` добавить иконки по классу. Заменить начало метода (после `super().__init__` вызова — метод не вызывает super, просто `QTreeWidgetItem(parent)`):

```python
def add_class_item(
    self,
    parent: QTreeWidgetItem,
    name: str,
    description: str,
    parent_index: int,
    seeding_index: int,
    class_index: int,
    confidence: float | None = None,
    manual: bool = False,
) -> QTreeWidgetItem:
    """Добавляет узел части растения с иконкой по классу."""
    _CLASS_ICONS = {
        "root": "🫚",
        "stem": "🌿",
        "flower": "🌸",
        "inflorescence": "🌸",
    }
    child = QTreeWidgetItem(parent)
    icon = _CLASS_ICONS.get((name or "").strip().lower(), "🔹")
    prefix = "✏ " if manual else icon + " "
    child.setText(0, prefix + name)
    child.setText(1, description)
    child.setData(
        0,
        Qt.UserRole,
        {
            "type": "class",
            "parent_index": parent_index,
            "seeding_index": seeding_index,
            "class_index": class_index,
        },
    )
    if confidence is not None:
        child.setData(1, self.CONFIDENCE_ROLE, float(confidence))
        if float(confidence) < 0.5:
            warn_color = QColor(80, 40, 10, 180)
            child.setBackground(0, warn_color)
            child.setBackground(1, warn_color)
            child.setForeground(0, QColor(255, 150, 80))
    parent.addChild(child)
    return child
```

Добавить импорт `QColor` в начало `tree_widget.py`:

```python
from PyQt5.QtGui import QColor
```

- [ ] **Шаг 4: Обновить вызовы в main_window.py**

В методе `_populate_layer_tree` (строка ~1796) передать флаг `manual`:

```python
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
    )
```

- [ ] **Шаг 5: Запустить тесты**

```
python -m pytest tests/test_tree_widget.py -v
```
Ожидается: 5 passed (2 старых + 3 новых)

- [ ] **Шаг 6: Коммит**

```bash
git add seeding/ui/tree_widget.py seeding/ui/main_window.py tests/test_tree_widget.py
git commit -m "feat: add class icons, confidence highlighting and manual flag to layer tree"
```

---

## Task 4: Детальная панель SeedlingDetailPanel

**Files:**
- Create: `seeding/ui/detail_panel.py`
- Modify: `seeding/ui/main_window.py`

- [ ] **Шаг 1: Создать `seeding/ui/detail_panel.py`**

```python
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
        """Обновляет панель данными выбранного объекта.

        Args:
            name: отображаемое имя (например "🫚 Корень")
            confidence: уверенность детекции, None если неизвестна
            manual: True если объект добавлен вручную
            crop_image: BGR ndarray кропа объекта
            mask_bitmap: бинарная маска uint8 0/255 в координатах кропа
            mask_color: RGB цвет маски для overlay
            bbox: координаты (x1,y1,x2,y2) в пикселях страницы
            pixels_per_mm: масштаб для пересчёта в мм
        """
        conf_text = "ручной" if manual else (
            f"{confidence:.2f}" if confidence is not None else "—"
        )
        self.name_label.setText(f"{name}  conf: {conf_text}")

        # Размер bbox
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
        """Отрисовывает превью кропа с наложенной полупрозрачной маской."""
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
```

- [ ] **Шаг 2: Встроить деталь-панель в main_window.py**

В методе `_build_ui` (строка ~549) после создания `right_tabs` добавить деталь-панель:

```python
# После: right_layout.addWidget(self.right_tabs, 1)
# Добавить:
separator = QFrame(right_panel)
separator.setFrameShape(QFrame.HLine)
separator.setFrameShadow(QFrame.Sunken)
right_layout.addWidget(separator)

from seeding.ui.detail_panel import SeedlingDetailPanel
self.detail_panel = SeedlingDetailPanel(right_panel)
self.detail_panel.navigate.connect(self._navigate_seedling)
right_layout.addWidget(self.detail_panel)
```

- [ ] **Шаг 3: Обновить `_on_tree_selection_changed` для обновления деталь-панели**

Найти метод `_on_tree_selection_changed` (строка ~1855) и добавить вызов `_update_detail_panel` в конец каждой ветки:

```python
def _on_tree_selection_changed(self) -> None:
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
```

- [ ] **Шаг 4: Добавить метод `_update_detail_panel`**

Добавить новый метод после `_on_tree_selection_changed`:

```python
def _update_detail_panel(self, payload: dict) -> None:
    """Заполняет детальную панель данными выбранного объекта."""
    item_type = payload.get("type")
    page_idx = int(payload.get("parent_index", 0))

    page_objects = []
    if (self.image_storage.class_object_image
            and page_idx < len(self.image_storage.class_object_image)):
        page_objects = self.image_storage.class_object_image[page_idx]

    page_img = None
    if self.image_storage.images and page_idx < len(self.image_storage.images):
        page_img = self.image_storage.images[page_idx]
        if not isinstance(page_img, np.ndarray):
            page_img = None

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

        # Кроп части берём из кропа сеянца + bbox части
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
        self.detail_panel.set_object(
            name=self._display_part_name(part.class_name),
            confidence=part.confidence,
            manual=getattr(part, "manual", False),
            crop_image=crop if crop is not None and crop.size > 0 else
                       (obj.image[0] if obj.image and isinstance(obj.image[0], np.ndarray) else None),
            mask_bitmap=getattr(part, "mask_bitmap", None),
            mask_color=mask_color,
            bbox=part.bbox,
            pixels_per_mm=self.pixels_per_mm,
        )
```

- [ ] **Шаг 5: Добавить метод `_navigate_seedling`**

Добавить метод после `_update_detail_panel`:

```python
def _navigate_seedling(self, delta: int) -> None:
    """Переключает выбор на следующий/предыдущий сеянец (delta = +1 или -1)."""
    item = self.tree_widget.currentItem()
    if item is None:
        return
    payload = item.data(0, Qt.UserRole) or {}
    item_type = payload.get("type")

    # Нормализуем до уровня сеянца
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
```

- [ ] **Шаг 6: Добавить горячие клавиши ← →**

В `__init__` метода `ImageEditor` (после `self._build_ui()`) добавить:

```python
QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(
    lambda: self._navigate_seedling(1)
)
QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(
    lambda: self._navigate_seedling(-1)
)
```

- [ ] **Шаг 7: Запустить тесты**

```
python -m pytest tests/ -q
```
Ожидается: все тесты проходят

- [ ] **Шаг 8: Коммит**

```bash
git add seeding/ui/detail_panel.py seeding/ui/main_window.py
git commit -m "feat: add seedling detail panel with crop preview and prev/next navigation"
```

---

## Task 5: Поле manual в моделях данных

**Files:**
- Modify: `seeding/models.py`
- Test: `tests/test_models_manual_flag.py` (новый файл)

- [ ] **Шаг 1: Написать тест**

```python
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
```

- [ ] **Шаг 2: Запустить, убедиться что падает**

```
python -m pytest tests/test_models_manual_flag.py -v
```
Ожидается: TypeError (unexpected keyword argument 'manual')

- [ ] **Шаг 3: Добавить поле manual в models.py**

В `seeding/models.py` в датакласс `AllClassImage` добавить поле (строка ~29, после `mask_bitmap`):

```python
manual: bool = False
```

В датакласс `ObjectImage` добавить поле (строка ~42, после `rotation_k`):

```python
manual: bool = False
```

- [ ] **Шаг 4: Запустить тесты**

```
python -m pytest tests/test_models_manual_flag.py tests/ -q
```
Ожидается: все тесты проходят

- [ ] **Шаг 5: Коммит**

```bash
git add seeding/models.py tests/test_models_manual_flag.py
git commit -m "feat: add manual flag to ObjectImage and AllClassImage"
```

---

## Task 6: Диалог выбора класса AddBoxDialog

**Files:**
- Create: `seeding/ui/add_box_dialog.py`
- Test: `tests/test_add_box_dialog.py` (новый файл)

- [ ] **Шаг 1: Написать тест**

```python
# tests/test_add_box_dialog.py
import os
from PyQt5.QtWidgets import QApplication
from seeding.ui.add_box_dialog import AddBoxDialog


def _app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    return app or QApplication([])


def test_add_box_dialog_returns_none_on_reject():
    _app()
    dlg = AddBoxDialog()
    dlg.reject()
    assert dlg.selected_class() is None


def test_add_box_dialog_returns_class_on_accept():
    _app()
    dlg = AddBoxDialog()
    # Симулируем выбор класса "root" и accept
    dlg._select_class("root")
    dlg.accept()
    assert dlg.selected_class() == "root"


def test_add_box_dialog_has_all_classes():
    _app()
    dlg = AddBoxDialog()
    classes = [btn.property("class_name") for btn in dlg._class_buttons]
    assert "seeding" in classes
    assert "root" in classes
    assert "stem" in classes
    assert "inflorescence" in classes
```

- [ ] **Шаг 2: Запустить, убедиться что падает**

```
python -m pytest tests/test_add_box_dialog.py -v
```
Ожидается: ImportError

- [ ] **Шаг 3: Создать `seeding/ui/add_box_dialog.py`**

```python
"""Диалог выбора класса при ручном добавлении бокса."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

_CLASS_LABELS: list[tuple[str, str]] = [
    ("seeding",       "🌱 Сеянец"),
    ("root",          "🫚 Корень"),
    ("stem",          "🌿 Стебель"),
    ("inflorescence", "🌸 Соцветие"),
]


class AddBoxDialog(QDialog):
    """Позволяет оператору выбрать класс перед добавлением ручного бокса."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Добавить объект")
        self.setModal(True)
        self.setMinimumWidth(260)
        self.setObjectName("panelCard")
        self._chosen: str | None = None
        self._class_buttons: list[QPushButton] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Выбери тип объекта", self)
        title.setObjectName("panelCardTitle")
        layout.addWidget(title)

        for class_name, label in _CLASS_LABELS:
            btn = QPushButton(label, self)
            btn.setProperty("class_name", class_name)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, cn=class_name: self._select_class(cn))
            layout.addWidget(btn)
            self._class_buttons.append(btn)

        actions = QHBoxLayout()
        cancel_btn = QPushButton("Отмена", self)
        cancel_btn.setProperty("variant", "secondary")
        cancel_btn.clicked.connect(self.reject)
        self._confirm_btn = QPushButton("Добавить", self)
        self._confirm_btn.setEnabled(False)
        self._confirm_btn.clicked.connect(self.accept)
        actions.addWidget(cancel_btn)
        actions.addWidget(self._confirm_btn)
        layout.addLayout(actions)

    def _select_class(self, class_name: str) -> None:
        """Выбирает класс и подсвечивает кнопку."""
        self._chosen = class_name
        for btn in self._class_buttons:
            btn.setChecked(btn.property("class_name") == class_name)
        self._confirm_btn.setEnabled(True)

    def selected_class(self) -> str | None:
        """Возвращает выбранный класс или None если диалог отменён."""
        return self._chosen if self.result() == QDialog.Accepted else None
```

- [ ] **Шаг 4: Запустить тесты**

```
python -m pytest tests/test_add_box_dialog.py -v
```
Ожидается: 3 passed

- [ ] **Шаг 5: Коммит**

```bash
git add seeding/ui/add_box_dialog.py tests/test_add_box_dialog.py
git commit -m "feat: add AddBoxDialog for manual box class selection"
```

---

## Task 7: Режим рисования боксов в CanvasGraphicsView

**Files:**
- Modify: `seeding/ui/main_window.py`

- [ ] **Шаг 1: Добавить draw-mode в CanvasGraphicsView**

Найти класс `CanvasGraphicsView` (строка ~121) и добавить:

1. Импорт `pyqtSignal` уже есть в строке ~16. Добавить сигнал и поля в `__init__`:

```python
# В класс CanvasGraphicsView добавить сигнал после объявления класса:
box_drawn = pyqtSignal(object)  # испускает QRectF

# В __init__ после self._scroll_start_pos:
self._draw_mode = False
self._draw_active = False
self._draw_start_pos = QPointF()
self._rubber_item = None
```

2. Добавить метод `set_draw_mode`:

```python
def set_draw_mode(self, enabled: bool) -> None:
    """Переключает режим рисования боксов."""
    self._draw_mode = enabled
    self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
    if not enabled and self._rubber_item is not None:
        self.scene().removeItem(self._rubber_item)
        self._rubber_item = None
    self._draw_active = False
```

3. Переопределить mouse-события — добавить обработку draw-mode:

```python
def mousePressEvent(self, event) -> None:
    if self._draw_mode and event.button() == Qt.LeftButton:
        self._draw_active = True
        self._draw_start_pos = self.mapToScene(event.pos())
        event.accept()
        return
    super().mousePressEvent(event)

def mouseMoveEvent(self, event) -> None:
    if self._draw_active:
        current = self.mapToScene(event.pos())
        rect = QRectF(self._draw_start_pos, current).normalized()
        if self._rubber_item is None:
            from PyQt5.QtWidgets import QGraphicsRectItem
            from PyQt5.QtGui import QPen, QColor
            self._rubber_item = QGraphicsRectItem()
            pen = QPen(QColor(255, 255, 0), 2, Qt.DashLine)
            self._rubber_item.setPen(pen)
            self._rubber_item.setBrush(Qt.transparent)
            self._rubber_item.setZValue(10)
            self.scene().addItem(self._rubber_item)
        self._rubber_item.setRect(rect)
        event.accept()
        return
    super().mouseMoveEvent(event)

def mouseReleaseEvent(self, event) -> None:
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
    super().mouseReleaseEvent(event)

def keyPressEvent(self, event) -> None:
    if self._draw_mode and event.key() == Qt.Key_Escape:
        self.set_draw_mode(False)
        event.accept()
        return
    super().keyPressEvent(event)
```

- [ ] **Шаг 2: Добавить кнопку "Добавить бокс" в mode_bar**

Найти блок mode_bar (строка ~455) и добавить кнопку после `edit_masks_mode_button`:

```python
self.add_box_mode_button = QPushButton("+ Бокс", mode_bar)
self.add_box_mode_button.setCheckable(True)
self.add_box_mode_button.setProperty("segmented", "true")
self.add_box_mode_button.setToolTip("Нарисуй прямоугольник чтобы добавить объект вручную")
self.add_box_mode_button.toggled.connect(self._toggle_add_box_mode)
mode_bar_layout.addWidget(self.add_box_mode_button)
```

- [ ] **Шаг 3: Подключить canvas view к сигналу box_drawn**

В методе `_build_ui`, где создаётся `CanvasGraphicsView` (найти по `self.canvas_view = CanvasGraphicsView`), добавить подключение сигнала:

```python
self.canvas_view.box_drawn.connect(self._on_box_drawn)
```

- [ ] **Шаг 4: Добавить методы `_toggle_add_box_mode` и `_on_box_drawn`**

Добавить после метода `_set_interaction_mode`:

```python
def _toggle_add_box_mode(self, enabled: bool) -> None:
    """Включает/выключает режим рисования боксов на канвасе."""
    self.canvas_view.set_draw_mode(enabled)
    # Снимаем другие режимы если активируется draw
    if enabled:
        self.view_mode_button.setChecked(False)
        self.edit_boxes_mode_button.setChecked(False)
        self.edit_masks_mode_button.setChecked(False)

def _on_box_drawn(self, scene_rect) -> None:
    """Обрабатывает нарисованный прямоугольник — показывает диалог выбора класса."""
    from PyQt5.QtCore import QRectF
    from seeding.ui.add_box_dialog import AddBoxDialog

    # Выключаем режим рисования
    self.add_box_mode_button.setChecked(False)

    # Получаем текущий индекс страницы
    page_idx = self._active_image_index
    if not self.image_storage.images or page_idx >= len(self.image_storage.images):
        return

    # Переводим координаты сцены в координаты страницы
    # (учитываем смещение pixmap_item если он есть)
    offset_x = offset_y = 0.0
    if self._pixmap_item is not None:
        p = self._pixmap_item.pos()
        offset_x, offset_y = p.x(), p.y()

    page_img = self.image_storage.images[page_idx]
    img_h, img_w = page_img.shape[:2] if isinstance(page_img, np.ndarray) else (0, 0)

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
```

- [ ] **Шаг 5: Добавить метод `_create_manual_object`**

```python
def _create_manual_object(
    self,
    page_idx: int,
    class_name: str,
    bbox: tuple[int, int, int, int],
) -> None:
    """Создаёт ObjectImage или AllClassImage по нарисованному боксу."""
    import numpy as np
    from seeding.mask_refiner import refine_mask_bitmap, polygon_to_bitmap, bitmap_to_polygon
    from seeding.models import AllClassImage, ObjectImage

    page_img = self.image_storage.images[page_idx]
    if not isinstance(page_img, np.ndarray):
        return

    x1, y1, x2, y2 = bbox
    crop = page_img[y1:y2, x1:x2].copy()

    # Строим маску по яркости внутри bbox (coarse_mask = весь crop)
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
        # Добавляем как часть активного сеянца
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
            from PyQt5.QtWidgets import QMessageBox
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

    # Обновляем дерево и вид
    self._populate_layer_tree()
    self.display_image_with_boxes(page_idx)
```

- [ ] **Шаг 6: Запустить все тесты**

```
python -m pytest tests/ -q
```
Ожидается: все тесты проходят

- [ ] **Шаг 7: Коммит**

```bash
git add seeding/ui/main_window.py
git commit -m "feat: add draw-box mode for manual annotation with class dialog"
```

---

## Self-review checklist

- [x] Spec coverage: Блок 1 (Tasks 1–2) ✓, Блок 3 (Tasks 3–4) ✓, Блок 2 (Tasks 5–7) ✓
- [x] Нет TBD/TODO в шагах
- [x] Типы согласованы: `bitmap_to_contours` → `list[np.ndarray]` — используется в Task 2
- [x] `manual: bool` добавляется в Task 5 до того как используется в Tasks 3, 4, 7
- [x] `AddBoxDialog` создаётся в Task 6 до использования в Task 7
- [x] `SeedlingDetailPanel.navigate` сигнал подключается к `_navigate_seedling` из Task 4
