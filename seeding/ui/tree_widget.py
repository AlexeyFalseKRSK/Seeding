"""Дерево слоёв для страниц, сеянцев и найденных частей растений."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QAbstractItemView, QHeaderView, QTreeWidget, QTreeWidgetItem

_CLASS_ICONS: dict[str, str] = {
    "root": "🫚",
    "stem": "🌿",
    "flower": "🌸",
    "inflorescence": "🌸",
}


class LayerTreeWidget(QTreeWidget):
    """Отображает иерархию страниц, объектов и классифицированных частей."""

    CONFIDENCE_ROLE = Qt.UserRole + 100

    def __init__(self) -> None:
        """Настраивает дерево слоёв, заголовки колонок и режимы отображения."""
        super().__init__()
        self.setHeaderLabels(["Название", "Описание"])
        header = self.header()
        header.setMinimumSectionSize(80)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        self.setColumnWidth(1, 140)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def resizeEvent(self, event) -> None:
        """Подстраивает ширину колонки описания при изменении размера виджета."""
        super().resizeEvent(event)
        width = max(110, min(160, int(self.viewport().width() * 0.38)))
        self.setColumnWidth(1, width)

    def add_root_item(
        self,
        name: str,
        description: str,
        index: int,
        image_type: str,
        image,
    ) -> QTreeWidgetItem:
        """Добавляет корневой узел страницы и сохраняет его служебные данные."""
        _ = image
        root = QTreeWidgetItem(self)
        root.setText(0, name)
        root.setText(1, description)
        root.setData(0, Qt.UserRole, {"index": index, "type": image_type})
        self.addTopLevelItem(root)
        return root

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
            else:
                child.setBackground(0, QColor(0, 0, 0, 0))
                child.setBackground(1, QColor(0, 0, 0, 0))
        parent.addChild(child)
        return child

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
        class_name: str | None = None,
    ) -> QTreeWidgetItem:
        """Добавляет узел части растения с иконкой по классу."""
        child = QTreeWidgetItem(parent)
        lookup_key = (class_name or name or "").strip().lower()
        icon = _CLASS_ICONS.get(lookup_key, "🔹")
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
            else:
                child.setBackground(0, QColor(0, 0, 0, 0))
                child.setBackground(1, QColor(0, 0, 0, 0))
        parent.addChild(child)
        return child
