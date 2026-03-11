"""Simple tree widget for pages, seedlings and detected parts."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QAbstractItemView, QHeaderView, QTreeWidget, QTreeWidgetItem


class LayerTreeWidget(QTreeWidget):
    CONFIDENCE_ROLE = Qt.UserRole + 100

    def __init__(self) -> None:
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
    ) -> QTreeWidgetItem:
        _ = (image_type, image)
        child = QTreeWidgetItem(parent)
        child.setText(0, name)
        child.setText(1, description)
        child.setData(
            0,
            Qt.UserRole,
            {"type": "seeding", "parent_index": parent_index, "index": index},
        )
        if confidence is not None:
            child.setData(1, self.CONFIDENCE_ROLE, float(confidence))
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
    ) -> QTreeWidgetItem:
        child = QTreeWidgetItem(parent)
        child.setText(0, name)
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
        parent.addChild(child)
        return child
