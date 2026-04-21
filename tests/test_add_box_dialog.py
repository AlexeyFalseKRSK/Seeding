# tests/test_add_box_dialog.py
import os
from PyQt5.QtWidgets import QApplication
from seeding.ui.add_box_dialog import AddBoxDialog


def _ensure_offscreen_qt() -> tuple[QApplication, bool]:
    """Создаёт `QApplication` в offscreen-режиме для тестов диалогов."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    created = app is None
    if app is None:
        app = QApplication([])
    return app, created


def test_add_box_dialog_returns_none_on_reject():
    app, created = _ensure_offscreen_qt()
    dlg = AddBoxDialog()
    dlg.reject()
    assert dlg.selected_class() is None
    dlg.deleteLater()
    if created:
        app.quit()


def test_add_box_dialog_returns_class_on_accept():
    app, created = _ensure_offscreen_qt()
    dlg = AddBoxDialog()
    dlg._select_class("root")
    dlg.accept()
    assert dlg.selected_class() == "root"
    dlg.deleteLater()
    if created:
        app.quit()


def test_add_box_dialog_has_all_classes():
    app, created = _ensure_offscreen_qt()
    dlg = AddBoxDialog()
    classes = [btn.property("class_name") for btn in dlg._class_buttons]
    assert "seeding" in classes
    assert "root" in classes
    assert "stem" in classes
    assert "inflorescence" in classes
    dlg.deleteLater()
    if created:
        app.quit()
