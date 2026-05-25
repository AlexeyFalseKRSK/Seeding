import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from seeding.ui.settings_dialog import AppSettingsDialog


def test_settings_dialog_returns_selected_values():
    app = QApplication.instance() or QApplication(sys.argv)
    _ = app
    dialog = AppSettingsDialog(
        current_device="cpu",
        current_batch_size=3,
        current_auto_save_enabled=False,
        current_report_output_dir="C:/Reports",
        parent=None,
    )

    settings = dialog.selected_settings()

    assert settings.analysis_device == "cpu"
    assert settings.analysis_batch_size == 3
    assert settings.auto_save_enabled is False
    assert settings.report_output_dir == "C:/Reports"


def test_settings_dialog_falls_back_to_auto_for_unknown_device():
    app = QApplication.instance() or QApplication(sys.argv)
    _ = app
    dialog = AppSettingsDialog(
        current_device="unknown",
        current_batch_size=0,
        parent=None,
    )

    settings = dialog.selected_settings()

    assert settings.analysis_device == "auto"
    assert settings.analysis_batch_size == 1
    assert settings.auto_save_enabled is True
    assert settings.report_output_dir == ""
