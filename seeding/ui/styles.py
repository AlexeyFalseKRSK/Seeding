"""Загрузка QSS-тем из папки ресурсов."""

from __future__ import annotations

from pathlib import Path

from seeding.config import PROJECT_ROOT

_STYLES_DIR = PROJECT_ROOT / "seeding" / "resources" / "styles"


def _style_path(theme: str) -> Path:
    """Возвращает путь до QSS-файла темы."""
    return _STYLES_DIR / f"{theme}.qss"


def build_main_stylesheet(theme: str = "dark") -> str:
    """Возвращает QSS основной темы приложения."""
    path = _style_path(theme)
    if not path.is_file():
        path = _style_path("dark")
    return path.read_text(encoding="utf-8")
