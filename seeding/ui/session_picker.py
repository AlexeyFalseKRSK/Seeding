"""Диалог выбора сохранённой сессии анализа."""
from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from seeding import database


class SessionPickerDialog(QDialog):
    """Показывает список сохранённых сессий пользователя для выбора и восстановления."""

    def __init__(self, user_id: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Открыть сессию")
        self.setMinimumSize(640, 380)
        self.setModal(True)

        self._session_ids: list[int] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        layout.addWidget(QLabel("Выберите сессию для восстановления:"))

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Дата", "Файл", "Стр.", "Статус"])
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.doubleClicked.connect(self.accept)
        layout.addWidget(self._table)

        buttons = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_sessions(user_id)

    def _load_sessions(self, user_id: int) -> None:
        rows = database.fetch_sessions_by_user(user_id, limit=50)
        self._table.setRowCount(len(rows))
        self._session_ids = []
        for i, row in enumerate(rows):
            self._session_ids.append(row["id"])
            source_path = row["source_path"]
            file_ok = Path(source_path).exists()

            date_item = QTableWidgetItem(row["created_at"][:16].replace("T", " "))
            date_item.setTextAlignment(Qt.AlignCenter)

            path_item = QTableWidgetItem(source_path)
            path_item.setToolTip(source_path)

            pages_item = QTableWidgetItem(str(row["page_count"] or "—"))
            pages_item.setTextAlignment(Qt.AlignCenter)

            status_item = QTableWidgetItem("✓ Найден" if file_ok else "✗ Не найден")
            status_item.setTextAlignment(Qt.AlignCenter)
            status_item.setForeground(Qt.darkGreen if file_ok else Qt.red)

            self._table.setItem(i, 0, date_item)
            self._table.setItem(i, 1, path_item)
            self._table.setItem(i, 2, pages_item)
            self._table.setItem(i, 3, status_item)

        if rows:
            self._table.selectRow(0)

    def selected_session_id(self) -> int | None:
        """Возвращает id выбранной сессии или None если ничего не выбрано."""
        indexes = self._table.selectedIndexes()
        if not indexes:
            return None
        row = indexes[0].row()
        if row < 0 or row >= len(self._session_ids):
            return None
        return self._session_ids[row]
