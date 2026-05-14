"""Dialog for opening and managing saved analysis sessions."""
from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from seeding import database


class SessionPickerDialog(QDialog):
    """Shows saved sessions for the current user and allows opening or deleting them."""

    def __init__(self, user_id: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Сохраненные сессии")
        self.setMinimumSize(820, 480)
        self.setModal(True)

        self._user_id = user_id
        self._session_ids: list[int] = []
        self._rows: list = []

        root = QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(18, 18, 18, 18)

        header = QFrame(self)
        header.setObjectName("sessionPickerHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        title = QLabel("Сохраненные сессии", header)
        title.setObjectName("dialogTitle")
        subtitle = QLabel(
            "Выберите проект для восстановления или удалите ненужные записи.",
            header,
        )
        subtitle.setObjectName("panelHint")
        subtitle.setWordWrap(True)
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        root.addWidget(header)

        self._empty_label = QLabel("Сохраненных сессий пока нет.", self)
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setObjectName("emptyStateLabel")
        self._empty_label.setMinimumHeight(80)
        root.addWidget(self._empty_label)

        self._table = QTableWidget(0, 5, self)
        self._table.setHorizontalHeaderLabels(
            ["Дата", "Файл", "Страниц", "Отчет", "Статус"]
        )
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(34)
        self._table.horizontalHeader().setHighlightSections(False)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._update_actions)
        self._table.doubleClicked.connect(self.accept)
        root.addWidget(self._table, 1)

        footer = QHBoxLayout()
        self._summary_label = QLabel("", self)
        self._summary_label.setObjectName("panelHint")
        footer.addWidget(self._summary_label, 1)

        self._refresh_button = QPushButton("Обновить", self)
        self._refresh_button.clicked.connect(self._load_sessions)
        footer.addWidget(self._refresh_button)

        self._delete_button = QPushButton("Удалить", self)
        self._delete_button.setObjectName("dangerButton")
        self._delete_button.clicked.connect(self._delete_selected_session)
        footer.addWidget(self._delete_button)

        buttons = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        footer.addWidget(buttons)
        root.addLayout(footer)

        self._open_button = buttons.button(QDialogButtonBox.Open)
        if self._open_button is not None:
            self._open_button.setText("Открыть")
        cancel_button = buttons.button(QDialogButtonBox.Cancel)
        if cancel_button is not None:
            cancel_button.setText("Отмена")

        self._load_sessions()

    def _load_sessions(self) -> None:
        rows = database.fetch_sessions_by_user(self._user_id, limit=100)
        self._rows = rows
        self._session_ids = []
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(rows))

        for index, row in enumerate(rows):
            self._session_ids.append(row["id"])
            source_path = row["source_path"] or ""
            report_path = row["report_path"] or ""
            file_ok = Path(source_path).exists()
            report_ok = bool(report_path and Path(report_path).exists())

            date_item = self._set_item(
                index,
                0,
                self._format_date(row["updated_at"] or row["created_at"]),
                center=True,
            )
            date_item.setData(Qt.UserRole, row["id"])
            self._set_item(index, 1, self._compact_path(source_path), tooltip=source_path)
            self._set_item(index, 2, str(row["page_count"] or "-"), center=True)
            self._set_item(
                index,
                3,
                "есть" if report_ok else "-",
                center=True,
                color=QColor("#2f9e44") if report_ok else QColor("#8b949e"),
                tooltip=report_path,
            )
            self._set_item(
                index,
                4,
                "файл найден" if file_ok else "файл потерян",
                center=True,
                color=QColor("#2f9e44") if file_ok else QColor("#d9480f"),
            )

        self._table.setSortingEnabled(True)
        if rows:
            self._table.selectRow(0)
        self._empty_label.setVisible(not rows)
        self._table.setVisible(bool(rows))
        self._summary_label.setText(f"Найдено сессий: {len(rows)}")
        self._update_actions()

    def _set_item(
        self,
        row: int,
        column: int,
        text: str,
        *,
        center: bool = False,
        color: QColor | None = None,
        tooltip: str | None = None,
    ) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        if center:
            item.setTextAlignment(Qt.AlignCenter)
        if color is not None:
            item.setForeground(color)
        if tooltip:
            item.setToolTip(tooltip)
        self._table.setItem(row, column, item)
        return item

    def _format_date(self, value: str) -> str:
        if not value:
            return "-"
        return value[:16].replace("T", " ")

    def _compact_path(self, source_path: str) -> str:
        if not source_path:
            return "-"
        path = Path(source_path)
        parent = path.parent.name
        return f"{parent}/{path.name}" if parent else path.name

    def _update_actions(self) -> None:
        has_selection = self.selected_session_id() is not None
        self._delete_button.setEnabled(has_selection)
        if self._open_button is not None:
            self._open_button.setEnabled(has_selection)

    def _delete_selected_session(self) -> None:
        session_id = self.selected_session_id()
        if session_id is None:
            return

        source_path = ""
        row_index = self._selected_row_index()
        if row_index is not None:
            path_item = self._table.item(row_index, 1)
            if path_item is not None:
                source_path = path_item.toolTip()
        message = "Удалить выбранную сессию?"
        if source_path:
            message += f"\n\n{source_path}"

        answer = QMessageBox.question(
            self,
            "Удалить сессию",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        try:
            deleted = database.delete_analysis_session(session_id, user_id=self._user_id)
        except Exception as error:
            QMessageBox.warning(self, "Ошибка удаления", str(error))
            return

        if not deleted:
            QMessageBox.information(
                self,
                "Сессия не найдена",
                "Запись уже удалена или принадлежит другому пользователю.",
            )
        self._load_sessions()

    def _selected_row_index(self) -> int | None:
        indexes = self._table.selectedIndexes()
        if not indexes:
            return None
        row = indexes[0].row()
        if row < 0:
            return None
        return row

    def selected_session_id(self) -> int | None:
        """Returns selected session id, or None when nothing is selected."""
        row = self._selected_row_index()
        if row is None:
            return None
        item = self._table.item(row, 0)
        if item is None:
            return None
        value = item.data(Qt.UserRole)
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
