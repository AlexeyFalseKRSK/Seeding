import os
import sqlite3

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QDialog

import seeding.database as database_module
from seeding.database import get_database_path
from seeding.ui.login_dialog import LoginDialog, NO_USERS_MESSAGE
from seeding.user_service import (
    DuplicateUserError,
    authenticate_user,
    create_user,
    fetch_user_logs,
    initialize_user_storage,
    record_user_action,
    update_user_password_by_login,
)


def _ensure_offscreen_qt() -> tuple[QApplication, bool]:
    """Создаёт QApplication в offscreen-режиме для UI-тестов."""

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    created = app is None
    if app is None:
        app = QApplication([])
    return app, created


def _prepare_test_database(tmp_path, monkeypatch) -> None:
    """Настраивает изолированную SQLite-базу для текущего теста."""

    monkeypatch.setenv(
        "SEEDING_DB_PATH",
        str(tmp_path / "seeding-data" / "seeding-test.sqlite3"),
    )
    initialize_user_storage()


def test_default_database_path_uses_data_directory(tmp_path, monkeypatch):
    """Проверяет использование `seeding/data` как пути хранения по умолчанию."""

    monkeypatch.delenv("SEEDING_DB_PATH", raising=False)
    monkeypatch.setattr(database_module, "PROJECT_ROOT", tmp_path)

    database_path = database_module.get_database_path()

    assert database_path == tmp_path / "seeding" / "data" / "seeding.sqlite3"


def test_initialize_user_storage_creates_minimal_schema(tmp_path, monkeypatch):
    """Проверяет создание таблиц пользователей и логов в настроенной SQLite-базе."""

    _prepare_test_database(tmp_path, monkeypatch)

    database_path = get_database_path()
    assert database_path.exists()

    connection = sqlite3.connect(database_path)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        connection.close()

    assert {"users", "user_logs"} <= tables


def test_create_user_and_authenticate(tmp_path, monkeypatch):
    """Проверяет вход пользователя, явно созданного в базе данных."""

    _prepare_test_database(tmp_path, monkeypatch)
    create_user("operator", "secret12")

    user = authenticate_user("  OPERATOR  ", "secret12")

    assert user is not None
    assert user.login == "operator"


def test_duplicate_login_is_rejected(tmp_path, monkeypatch):
    """Проверяет отклонение повторного создания одного и того же логина."""

    _prepare_test_database(tmp_path, monkeypatch)
    create_user("operator", "secret12")

    try:
        create_user("Operator", "secret12")
    except DuplicateUserError:
        pass
    else:
        raise AssertionError("Expected DuplicateUserError for an existing login.")


def test_update_password_changes_credentials(tmp_path, monkeypatch):
    """Проверяет, что новый пароль принимается, а прежний отклоняется."""

    _prepare_test_database(tmp_path, monkeypatch)
    create_user("operator", "secret12")

    update_user_password_by_login("operator", "new-secret12")

    assert authenticate_user("operator", "secret12") is None
    assert authenticate_user("operator", "new-secret12") is not None


def test_user_actions_are_written_to_log_table(tmp_path, monkeypatch):
    """Проверяет запись простой записи пользовательского действия в лог."""

    _prepare_test_database(tmp_path, monkeypatch)
    user = create_user("operator", "secret12")

    record_user_action(user.id, "login", "Test sign in.")
    logs = fetch_user_logs(user.id)

    assert logs
    assert logs[0]["action"] == "login"
    assert logs[0]["details"] == "Test sign in."


def test_login_dialog_accepts_valid_credentials(tmp_path, monkeypatch):
    """Проверяет успешное закрытие диалога при корректном логине."""

    _prepare_test_database(tmp_path, monkeypatch)
    create_user("operator", "secret12")
    app, created = _ensure_offscreen_qt()
    dialog = LoginDialog()
    dialog.login_input.setText("operator")
    dialog.password_input.setText("secret12")

    dialog._attempt_login()

    assert dialog.result() == QDialog.Accepted
    assert dialog.authenticated_user is not None
    assert dialog.authenticated_user.login == "operator"

    dialog.close()
    if created:
        app.quit()


def test_login_dialog_uses_frameless_window_style(tmp_path, monkeypatch):
    """Проверяет использование безрамочного полупрозрачного окна для карточки входа."""

    _prepare_test_database(tmp_path, monkeypatch)
    app, created = _ensure_offscreen_qt()
    dialog = LoginDialog()

    assert dialog.windowFlags() & Qt.FramelessWindowHint
    assert dialog.testAttribute(Qt.WA_TranslucentBackground)

    dialog.close()
    if created:
        app.quit()


def test_login_dialog_shows_guidance_when_no_users_exist(tmp_path, monkeypatch):
    """Проверяет показ понятной инструкции, когда в базе нет пользователей."""

    _prepare_test_database(tmp_path, monkeypatch)
    app, created = _ensure_offscreen_qt()
    dialog = LoginDialog()
    dialog.login_input.setText("operator")
    dialog.password_input.setText("secret12")

    dialog._attempt_login()

    assert dialog.result() != QDialog.Accepted
    assert dialog.authenticated_user is None
    assert not dialog.error_label.isHidden()
    assert "создайте первого пользователя" in dialog.error_label.text().casefold()
    assert "seeding.manage_users create" in NO_USERS_MESSAGE

    dialog.close()
    if created:
        app.quit()


def test_login_dialog_shows_error_on_invalid_credentials(tmp_path, monkeypatch):
    """Проверяет показ ошибки и отсутствие закрытия при неверном пароле."""

    _prepare_test_database(tmp_path, monkeypatch)
    create_user("operator", "secret12")
    app, created = _ensure_offscreen_qt()
    dialog = LoginDialog()
    dialog.login_input.setText("operator")
    dialog.password_input.setText("bad")

    dialog._attempt_login()

    assert dialog.result() != QDialog.Accepted
    assert dialog.authenticated_user is None
    assert not dialog.error_label.isHidden()
    assert "Неверный логин" in dialog.error_label.text()

    dialog.close()
    if created:
        app.quit()
