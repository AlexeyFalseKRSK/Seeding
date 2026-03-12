"""Менеджер иконок с единым доступом к ресурсам."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QStyle, QWidget

from seeding.config import PROJECT_ROOT


class IconManager:
    """Загружает иконки из ресурсов и кэширует их."""

    def __init__(self, widget: QWidget) -> None:
        """Инициализирует менеджер иконок для конкретного виджета.

        Виджет используется как источник fallback-иконок из ``QStyle``.
        """
        self._widget = widget
        self._icons_dir = PROJECT_ROOT / "seeding" / "resources" / "icons"
        self._cache: dict[str, QIcon] = {}

    def get_icon(
        self,
        name: str,
        *,
        fallback_standard_icon: QStyle.StandardPixmap | None = None,
    ) -> QIcon:
        """Возвращает иконку по имени ресурса."""
        if name in self._cache:
            return self._cache[name]

        path = self._icons_dir / name
        if path.exists():
            icon = QIcon(str(path))
            if not icon.isNull():
                self._cache[name] = icon
                return icon

        if fallback_standard_icon is not None:
            return self._widget.style().standardIcon(fallback_standard_icon)
        return QIcon()

    @staticmethod
    def has_icon_resource(name: str) -> bool:
        """Проверяет наличие файла иконки в ресурсах."""
        path = PROJECT_ROOT / "seeding" / "resources" / "icons" / name
        return Path(path).exists()
