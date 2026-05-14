import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMessageBox

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
    assert not dlg._delete_button.isEnabled()


def test_selected_session_id(qapp, db_path):
    user_id = database.insert_user("u2", "hash")
    sid = database.insert_analysis_session(user_id=user_id, source_path="/c.pdf")
    dlg = SessionPickerDialog(user_id=user_id, parent=None)
    dlg._table.selectRow(0)
    assert dlg.selected_session_id() == sid


def test_selected_session_id_survives_table_sort(qapp, db_path):
    user_id = database.insert_user("u_sort", "hash")
    first = database.insert_analysis_session(user_id=user_id, source_path="/z.pdf")
    second = database.insert_analysis_session(user_id=user_id, source_path="/a.pdf")
    dlg = SessionPickerDialog(user_id=user_id, parent=None)

    dlg._table.sortItems(1, 0)
    dlg._table.selectRow(0)

    assert dlg.selected_session_id() in {first, second}
    assert dlg.selected_session_id() == second


def test_delete_selected_session(qapp, db_path, monkeypatch):
    user_id = database.insert_user("u3", "hash")
    sid = database.insert_analysis_session(user_id=user_id, source_path="/delete.pdf")
    dlg = SessionPickerDialog(user_id=user_id, parent=None)
    dlg._table.selectRow(0)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    dlg._delete_selected_session()

    assert database.fetch_session_by_id(sid) is None
    assert dlg._table.rowCount() == 0
    assert dlg.selected_session_id() is None
