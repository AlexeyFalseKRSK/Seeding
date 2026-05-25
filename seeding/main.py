"""Точка входа настольного клиента Seeding."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMessageBox

from seeding.auth import AuthUser
from seeding.config import (
    APP_FONT_FAMILY,
    APP_FONT_SIZE,
    DEFAULT_CLASSIFY_WEIGHTS_PATH,
    DEFAULT_WEIGHTS_PATH,
    PROJECT_ROOT,
)
from seeding.ui.login_dialog import LoginDialog
from seeding.ui.main_window import ImageEditor
from seeding.ui.styles import build_main_stylesheet
from seeding.user_service import (
    StorageUnavailableError,
    initialize_user_storage,
    record_user_action,
)
from seeding.utils import resolve_weights_path

logger = logging.getLogger(__name__)


def _resolve_model_path(path_value: str, *, default_path: Path) -> str | None:
    """Разрешает путь к модели с резервным переходом к пути проекта по умолчанию."""

    resolved = resolve_weights_path(path_value, base_dirs=(PROJECT_ROOT, Path.cwd()))
    if resolved is not None:
        return str(resolved)
    fallback = resolve_weights_path(str(default_path), base_dirs=(PROJECT_ROOT, Path.cwd()))
    if fallback is not None:
        return str(fallback)
    return None


def _show_startup_error(message: str) -> None:
    """Показывает критическое сообщение при ошибке запуска приложения."""

    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication(sys.argv)
        created = True
    QMessageBox.critical(None, "Ошибка запуска", message)
    if created:
        app.quit()


class SessionController:
    """Управляет входом, выходом и повторным открытием главного окна приложения."""

    def __init__(
        self,
        app: QApplication,
        *,
        weights_path: str,
        classify_weights_path: str,
    ) -> None:
        self.app = app
        self.weights_path = weights_path
        self.classify_weights_path = classify_weights_path
        self.window: ImageEditor | None = None
        self.current_user: AuthUser | None = None

    def start(self) -> bool:
        """Запускает первую пользовательскую сессию."""

        if not self._show_login_dialog():
            return False
        self._show_main_window()
        return True

    def _record_action(
        self,
        user_id: int,
        action: str,
        details: str | None = None,
    ) -> None:
        """Записывает действие пользователя, не прерывая работу сессии."""

        try:
            record_user_action(user_id, action, details)
        except Exception:
            logger.exception("Не удалось записать лог действия '%s'", action)

    def _show_login_dialog(self) -> bool:
        """Показывает форму входа и сохраняет аутентифицированного пользователя."""

        login_dialog = LoginDialog()
        if login_dialog.exec_() != LoginDialog.Accepted:
            return False

        user = login_dialog.authenticated_user
        if user is None:
            return False

        self.current_user = user
        self._record_action(
            user.id,
            "login",
            "Пользователь выполнил вход в настольное приложение.",
        )
        return True

    def _show_main_window(self) -> None:
        """Создаёт и показывает главное окно для текущей сессии."""

        window = ImageEditor(
            weights_path=self.weights_path,
            classify_weights_path=self.classify_weights_path,
            current_user=self.current_user,
        )
        window.logout_requested.connect(self._handle_logout_requested)
        window.showMaximized()
        window.preload_models()
        self.window = window

    def _handle_logout_requested(self) -> None:
        """Завершает текущую сессию и возвращает пользователя к форме входа."""

        if self.window is None:
            return

        previous_window = self.window
        previous_window.hide()

        if self.current_user is not None:
            self._record_action(
                self.current_user.id,
                "logout",
                "Пользователь вышел из настольного приложения.",
            )

        if self._show_login_dialog():
            self._show_main_window()
            previous_window.close()
            previous_window.deleteLater()
            return

        previous_window.close()
        previous_window.deleteLater()
        self.window = None
        self.current_user = None
        self.app.quit()


def main() -> None:
    """Настраивает и запускает графическое приложение."""

    parser = argparse.ArgumentParser(description="Seeding")
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS_PATH),
        help="Path to the detector .pt weights",
    )
    parser.add_argument(
        "--classify-weights",
        default=str(DEFAULT_CLASSIFY_WEIGHTS_PATH),
        help="Path to the classifier .pt weights",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s - %(message)s",
    )

    try:
        initialize_user_storage()
    except StorageUnavailableError as error:
        _show_startup_error(
            "Не удалось инициализировать локальную базу данных.\n\n"
            f"{error}"
        )
        sys.exit(1)

    weights_path = _resolve_model_path(args.weights, default_path=DEFAULT_WEIGHTS_PATH)
    classify_weights_path = _resolve_model_path(
        args.classify_weights,
        default_path=DEFAULT_CLASSIFY_WEIGHTS_PATH,
    )
    if weights_path is None or classify_weights_path is None:
        _show_startup_error("Не удалось найти один или оба файла моделей.")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setFont(QFont(APP_FONT_FAMILY, APP_FONT_SIZE))
    app.setStyleSheet(build_main_stylesheet("dark"))

    session_controller = SessionController(
        app,
        weights_path=weights_path,
        classify_weights_path=classify_weights_path,
    )
    if not session_controller.start():
        sys.exit(0)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
