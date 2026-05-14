# Session Restore UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить диалог "Открыть сессию" в меню Файл — список сохранённых сессий пользователя с возможностью восстановить состояние без повторного инференса.

**Architecture:** Создаём отдельный виджет `SessionPickerDialog` в `seeding/ui/session_picker.py`. `ImageEditor` получает метод `_open_session_dialog()` и пункт меню. При выборе сессии `load_session()` восстанавливает `AppState`, окно перерисовывает дерево и холст. Если исходный файл не найден — предупреждение, но данные отображаются.

**Tech Stack:** PyQt5, SQLite (через `database.fetch_sessions_by_user` + `session_service.load_session`), существующий `AppState`.

---

## Файловая карта

| Файл | Действие | Роль |
|------|----------|------|
| `seeding/ui/session_picker.py` | Создать | Диалог со списком сессий |
| `seeding/ui/main_window.py` | Изменить | Пункт меню + метод открытия сессии |
| `tests/test_session_picker.py` | Создать | Тесты диалога (offscreen) |

---

## Task 1: SessionPickerDialog

**Files:**
- Create: `seeding/ui/session_picker.py`
- Create: `tests/test_session_picker.py`

- [ ] **Шаг 1: Написать падающий тест**

```python
# tests/test_session_picker.py
import os
import sys
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication
from seeding import database
from seeding.ui.session_picker import SessionPickerDialog

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app

@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.sqlite3"
    os.environ["SEEDING_DB_PATH"] = str(path)
    database.initialize_database()
    yield path
    del os.environ["SEEDING_DB_PATH"]

def test_dialog_shows_sessions(qapp, db_path):
    user_id = database.insert_user("u1", "hash")
    database.insert_analysis_session(user_id=user_id, source_path="/a.pdf", page_count=2)
    database.insert_analysis_session(user_id=user_id, source_path="/b.pdf", page_count=5)
    dlg = SessionPickerDialog(user_id=user_id, parent=None)
    assert dlg._table.rowCount() == 2

def test_dialog_empty_for_unknown_user(qapp, db_path):
    dlg = SessionPickerDialog(user_id=999, parent=None)
    assert dlg._table.rowCount() == 0

def test_selected_session_id(qapp, db_path):
    user_id = database.insert_user("u2", "hash")
    sid = database.insert_analysis_session(user_id=user_id, source_path="/c.pdf")
    dlg = SessionPickerDialog(user_id=user_id, parent=None)
    dlg._table.selectRow(0)
    assert dlg.selected_session_id() == sid
```

- [ ] **Шаг 2: Запустить тест — убедиться что падает (ImportError)**

```
cd e:\_JOB_\_Python\Seeding && set QT_QPA_PLATFORM=offscreen && .venv\Scripts\python -m pytest tests/test_session_picker.py -v
```

- [ ] **Шаг 3: Создать seeding/ui/session_picker.py**

```python
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
            status_item.setForeground(
                Qt.darkGreen if file_ok else Qt.red
            )

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
```

- [ ] **Шаг 4: Запустить тесты диалога**

```
cd e:\_JOB_\_Python\Seeding && set QT_QPA_PLATFORM=offscreen && .venv\Scripts\python -m pytest tests/test_session_picker.py -v
```
Ожидаем: 3 PASS.

- [ ] **Шаг 5: Запустить все тесты**

```
cd e:\_JOB_\_Python\Seeding && set QT_QPA_PLATFORM=offscreen && .venv\Scripts\python -m pytest tests/ -v
```
Ожидаем: все PASS.

- [ ] **Шаг 6: Коммит**

```bash
git add seeding/ui/session_picker.py tests/test_session_picker.py
git commit -m "feat: SessionPickerDialog — диалог выбора сохранённой сессии"
```

---

## Task 2: Интеграция в ImageEditor

**Files:**
- Modify: `seeding/ui/main_window.py`

Нужно:
1. Добавить импорт `SessionPickerDialog` и `load_session`
2. Добавить `action_open_session` в `_build_toolbar()`
3. Добавить пункт "Открыть сессию" в меню Файл (после "Открыть файл")
4. Реализовать `_open_session_dialog()` — показать диалог, загрузить сессию, обновить UI
5. Передавать `current_user_id` в `app_state` при инициализации окна

> **Ориентиры в файле:**
> - Импорты — блок вокруг строки 88–115
> - `_build_toolbar()` — строка ~1030, список action-ов строки ~1047–1136
> - Меню Файл — строки ~1166–1172
> - `__init__` — строка ~257, `self.current_user` устанавливается на строке ~269
> - `self.app_state = AppState(...)` — строка ~272

- [ ] **Шаг 1: Добавить импорты**

Найти строку `from seeding.session_service import save_session` и изменить на:

```python
from seeding.session_service import load_session, save_session
from seeding.ui.session_picker import SessionPickerDialog
```

- [ ] **Шаг 2: Передать current_user_id в app_state при инициализации**

Найти строку `self.app_state = AppState(image_storage=OriginalImage())` и заменить на:

```python
self.app_state = AppState(
    image_storage=OriginalImage(),
    current_user_id=current_user.id if current_user else None,
)
```

- [ ] **Шаг 3: Добавить action_open_session в _build_toolbar()**

Найти блок где создаётся `self.action_open` (строка ~1047) и добавить после него:

```python
self.action_open_session = self._create_action(
    "action_open.svg",
    "Открыть сессию",
    self._open_session_dialog,
    shortcut="Ctrl+Shift+O",
    fallback_standard_icon=QStyle.SP_FileDialogDetailedView,
)
```

- [ ] **Шаг 4: Добавить пункт в меню Файл**

Найти в `_build_menu()` строки:
```python
file_menu.addAction(self.action_open)
file_menu.addAction(self.action_add)
file_menu.addSeparator()
```
И заменить на:
```python
file_menu.addAction(self.action_open)
file_menu.addAction(self.action_add)
file_menu.addAction(self.action_open_session)
file_menu.addSeparator()
```

- [ ] **Шаг 5: Реализовать _open_session_dialog()**

Добавить метод рядом с `_on_save()`:

```python
def _open_session_dialog(self) -> None:
    """Открывает диалог выбора сохранённой сессии и восстанавливает её."""
    user_id = self.app_state.current_user_id
    if user_id is None:
        self._show_info("Нет пользователя", "Войдите в систему чтобы открыть сессию.")
        return

    dlg = SessionPickerDialog(user_id=user_id, parent=self)
    if dlg.exec_() != SessionPickerDialog.Accepted:
        return

    session_id = dlg.selected_session_id()
    if session_id is None:
        return

    try:
        state = load_session(session_id)
    except Exception as exc:
        self._show_error("Ошибка загрузки сессии", str(exc))
        return

    if state is None:
        self._show_error("Сессия не найдена", f"Сессия {session_id} не найдена в БД.")
        return

    missing = getattr(state, "_missing_source", None)

    self.app_state = state
    self.image_storage = state.image_storage
    self.pixels_per_mm = state.pixels_per_mm

    self._refresh_tree()
    self._refresh_statistics_panel()
    self._restore_display(preserve_view=False)

    if missing:
        self._show_info(
            "Файл не найден",
            f"Исходный файл не найден:\n{missing}\n\n"
            "Данные анализа восстановлены, изображения недоступны.",
        )
    else:
        self.statusBar().showMessage(
            f"Сессия восстановлена: {state.image_storage.file_path}", 4000
        )
```

- [ ] **Шаг 6: Запустить все тесты**

```
cd e:\_JOB_\_Python\Seeding && set QT_QPA_PLATFORM=offscreen && .venv\Scripts\python -m pytest tests/ -v
```
Ожидаем: все PASS.

- [ ] **Шаг 7: Коммит**

```bash
git add seeding/ui/main_window.py
git commit -m "feat: меню Файл → Открыть сессию с восстановлением AppState"
```

---

## Самопроверка

**Покрытие требований:**
- ✅ Диалог со списком сессий пользователя — `SessionPickerDialog`
- ✅ Дата, путь, кол-во страниц, статус файла — 4 колонки таблицы
- ✅ Двойной клик = открыть сессию — `doubleClicked.connect(accept)`
- ✅ Восстановление без инференса — `load_session()` из БД
- ✅ Предупреждение если файл не найден — `_missing_source` + `_show_info`
- ✅ Пункт меню Файл → Открыть сессию + Ctrl+Shift+O

**Типовая согласованность:**
- `selected_session_id() -> int | None` — используется в Task 2 шаг 5
- `load_session(session_id: int) -> AppState | None` — уже существует в `session_service.py`
- `fetch_sessions_by_user(user_id, limit)` — уже существует в `database.py`
