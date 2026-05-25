"""Диалог выбора класса при ручном добавлении бокса."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

_CLASS_LABELS: list[tuple[str, str]] = [
    ("seeding",       "🌱 Сеянец"),
    ("root",          "🫚 Корень"),
    ("stem",          "🌿 Стебель"),
    ("inflorescence", "🌸 Соцветие"),
]


class AddBoxDialog(QDialog):
    """Позволяет оператору выбрать класс перед добавлением ручного бокса."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Добавить объект")
        self.setModal(True)
        self.setMinimumWidth(260)
        self.setObjectName("panelCard")
        self._chosen: str | None = None
        self._class_buttons: list[QPushButton] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Выбери тип объекта", self)
        title.setObjectName("panelCardTitle")
        layout.addWidget(title)

        for class_name, label in _CLASS_LABELS:
            btn = QPushButton(label, self)
            btn.setProperty("class_name", class_name)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, cn=class_name: self._select_class(cn))
            layout.addWidget(btn)
            self._class_buttons.append(btn)

        actions = QHBoxLayout()
        cancel_btn = QPushButton("Отмена", self)
        cancel_btn.setProperty("variant", "secondary")
        cancel_btn.clicked.connect(self.reject)
        self._confirm_btn = QPushButton("Добавить", self)
        self._confirm_btn.setEnabled(False)
        self._confirm_btn.clicked.connect(self.accept)
        actions.addWidget(cancel_btn)
        actions.addWidget(self._confirm_btn)
        layout.addLayout(actions)

    def _select_class(self, class_name: str) -> None:
        """Выбирает класс и подсвечивает кнопку."""
        self._chosen = class_name
        for btn in self._class_buttons:
            btn.setChecked(btn.property("class_name") == class_name)
        self._confirm_btn.setEnabled(True)

    def selected_class(self) -> str | None:
        """Возвращает выбранный класс или None если диалог отменён."""
        return self._chosen if self.result() == QDialog.Accepted else None
