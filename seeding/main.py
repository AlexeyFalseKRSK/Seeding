"""Точка входа для запуска графического приложения Seeding."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMessageBox

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
from seeding.utils import resolve_weights_path


def _resolve_model_path(path_value: str, *, default_path: Path) -> str | None:
    """Пытается найти путь к модели и при необходимости использует резервный."""
    resolved = resolve_weights_path(path_value, base_dirs=(PROJECT_ROOT, Path.cwd()))
    if resolved is not None:
        return str(resolved)
    fallback = resolve_weights_path(str(default_path), base_dirs=(PROJECT_ROOT, Path.cwd()))
    if fallback is not None:
        return str(fallback)
    return None


class SessionController:
    """Управляет входом пользователя и повторным открытием главного окна."""

    def __init__(
        self,
        app: QApplication,
        *,
        weights_path: str,
        classify_weights_path: str,
    ) -> None:
        """Сохраняет параметры запуска и текущее активное окно приложения."""

        self.app = app
        self.weights_path = weights_path
        self.classify_weights_path = classify_weights_path
        self.window: ImageEditor | None = None

    def start(self) -> bool:
        """Запускает первую пользовательскую сессию и сообщает, входить ли в event loop."""

        if not self._show_login_dialog():
            return False
        self._show_main_window()
        return True

    def _show_login_dialog(self) -> bool:
        """Показывает экран входа и возвращает признак успешной авторизации."""

        login_dialog = LoginDialog()
        return login_dialog.exec_() == LoginDialog.Accepted

    def _show_main_window(self) -> None:
        """Создаёт и показывает главное окно для новой сессии."""

        window = ImageEditor(
            weights_path=self.weights_path,
            classify_weights_path=self.classify_weights_path,
        )
        window.logout_requested.connect(self._handle_logout_requested)
        window.showMaximized()
        self.window = window

    def _handle_logout_requested(self) -> None:
        """Завершает текущую сессию и возвращает пользователя на экран входа."""

        if self.window is None:
            return

        previous_window = self.window
        previous_window.hide()

        if self._show_login_dialog():
            self._show_main_window()
            previous_window.close()
            previous_window.deleteLater()
            return

        previous_window.close()
        previous_window.deleteLater()
        self.window = None
        self.app.quit()


def main() -> None:
    """Настраивает приложение, проверяет пути к моделям и запускает главное окно."""
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

    weights_path = _resolve_model_path(args.weights, default_path=DEFAULT_WEIGHTS_PATH)
    classify_weights_path = _resolve_model_path(
        args.classify_weights,
        default_path=DEFAULT_CLASSIFY_WEIGHTS_PATH,
    )
    if weights_path is None or classify_weights_path is None:
        app = QApplication(sys.argv)
        QMessageBox.critical(
            None,
            "Ошибка моделей",
            "Не удалось найти один или оба файла моделей.",
        )
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
