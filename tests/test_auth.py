import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QDialog

from seeding.auth import TEST_USER, authenticate
from seeding.ui.login_dialog import LoginDialog


def _ensure_offscreen_qt() -> tuple[QApplication, bool]:
    """Создаёт `QApplication` в offscreen-режиме для тестов авторизации."""

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    created = app is None
    if app is None:
        app = QApplication([])
    return app, created


def test_authenticate_accepts_single_stub_user():
    """Разрешает вход только тестовому пользователю и нормализует e-mail."""

    assert authenticate("  USER@USER.COM  ", "user") == TEST_USER


def test_authenticate_rejects_invalid_credentials():
    """Отклоняет неверный логин и неверный пароль."""

    assert authenticate(TEST_USER.email, "wrong-password") is None
    assert authenticate("wrong@user.com", "user") is None


def test_login_dialog_accepts_valid_credentials():
    """Закрывает диалог с успехом при корректных учётных данных."""

    app, created = _ensure_offscreen_qt()
    dialog = LoginDialog()
    dialog.email_input.setText(TEST_USER.email)
    dialog.password_input.setText("user")

    dialog._attempt_login()

    assert dialog.result() == QDialog.Accepted
    assert dialog.authenticated_user == TEST_USER

    dialog.close()
    if created:
        app.quit()


def test_login_dialog_uses_frameless_window_style():
    """Использует безрамочный стиль, чтобы оставалась только карточка входа."""

    app, created = _ensure_offscreen_qt()
    dialog = LoginDialog()

    assert dialog.windowFlags() & Qt.FramelessWindowHint
    assert dialog.testAttribute(Qt.WA_TranslucentBackground)

    dialog.close()
    if created:
        app.quit()


def test_login_dialog_shows_error_on_invalid_credentials():
    """Показывает ошибку и не пускает в приложение при неверном пароле."""

    app, created = _ensure_offscreen_qt()
    dialog = LoginDialog()
    dialog.email_input.setText(TEST_USER.email)
    dialog.password_input.setText("bad")

    dialog._attempt_login()

    assert dialog.result() != QDialog.Accepted
    assert dialog.authenticated_user is None
    assert not dialog.error_label.isHidden()
    assert "Неверный логин" in dialog.error_label.text()

    dialog.close()
    if created:
        app.quit()
