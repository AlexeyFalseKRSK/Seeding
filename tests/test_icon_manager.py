import os

from PyQt5.QtWidgets import QApplication, QStyle, QWidget

from seeding.ui.icon_manager import IconManager


def test_icon_manager_has_expected_resources():
    """Проверяет наличие обязательных файлов иконок в ресурсах приложения."""
    assert IconManager.has_icon_resource("action_open.svg")
    assert IconManager.has_icon_resource("action_detect.svg")
    assert IconManager.has_icon_resource("action_report.svg")
    assert not IconManager.has_icon_resource("missing_icon.svg")


def test_icon_manager_returns_fallback_icon():
    """Проверяет загрузку fallback-иконки и корректную выдачу реального ресурса."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    created_app = app is None
    if app is None:
        app = QApplication([])
    manager = IconManager(QWidget())
    icon = manager.get_icon(
        "missing_icon.svg",
        fallback_standard_icon=QStyle.SP_FileIcon,
    )
    assert not icon.isNull()
    assert not manager.get_icon("action_open.svg").isNull()
    if created_app:
        app.quit()
