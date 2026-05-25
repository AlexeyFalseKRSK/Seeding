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


def test_low_confidence_child_has_warning_background():
    """Сеянец с confidence < 0.5 должен иметь ненулевой цвет фона."""
    app, created = _ensure_offscreen_qt()
    tree = LayerTreeWidget()
    root = tree.add_root_item("p", "d", 0, "original", None)
    child = tree.add_child_item(root, "S", "d", 0, 0, "seeding", None, confidence=0.3)

    bg = child.background(0)
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
    assert bg.color().alpha() == 0

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
