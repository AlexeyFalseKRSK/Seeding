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
