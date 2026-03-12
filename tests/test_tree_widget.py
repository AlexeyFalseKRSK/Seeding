import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from seeding.ui.tree_widget import LayerTreeWidget


def _ensure_offscreen_qt() -> tuple[QApplication, bool]:
    """Создаёт `QApplication` в offscreen-режиме для тестов дерева слоёв."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    created = app is None
    if app is None:
        app = QApplication([])
    return app, created


def test_tree_widget_stores_payload_and_confidence_metadata():
    """Проверяет сохранение payload и confidence в пользовательских ролях дерева."""
    app, created = _ensure_offscreen_qt()

    tree = LayerTreeWidget()
    root = tree.add_root_item("image_1", "source", 0, "original", None)
    child = tree.add_child_item(
        root,
        "Seeding 1",
        "conf=0.20",
        0,
        0,
        "seeding",
        None,
        confidence=0.2,
    )
    cls = tree.add_class_item(
        child,
        "root",
        "conf=0.70",
        0,
        0,
        0,
        confidence=0.7,
    )

    assert root.data(0, Qt.UserRole) == {"index": 0, "type": "original"}
    assert child.data(0, Qt.UserRole) == {
        "type": "seeding",
        "parent_index": 0,
        "index": 0,
    }
    assert cls.data(0, Qt.UserRole) == {
        "type": "class",
        "parent_index": 0,
        "seeding_index": 0,
        "class_index": 0,
    }
    assert child.data(1, LayerTreeWidget.CONFIDENCE_ROLE) == 0.2
    assert cls.data(1, LayerTreeWidget.CONFIDENCE_ROLE) == 0.7

    tree.deleteLater()
    if created:
        app.quit()


def test_tree_widget_has_two_columns_and_non_editable_items():
    """Проверяет базовую конфигурацию дерева: колонки и запрет редактирования."""
    app, created = _ensure_offscreen_qt()

    tree = LayerTreeWidget()

    assert tree.columnCount() == 2
    assert tree.headerItem().text(0)
    assert tree.headerItem().text(1)
    assert tree.editTriggers() == tree.NoEditTriggers

    tree.deleteLater()
    if created:
        app.quit()
