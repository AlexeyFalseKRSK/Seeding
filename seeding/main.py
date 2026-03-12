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

    window = ImageEditor(
        weights_path=weights_path,
        classify_weights_path=classify_weights_path,
    )
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
