import os
import sqlite3

import pytest

from seeding import database
from seeding.database import (
    delete_analysis_session,
    fetch_detections_by_session,
    fetch_parts_by_detection,
    fetch_session_by_id,
    fetch_session_sources,
    fetch_sessions_by_user,
    insert_analysis_session,
    insert_detection,
    insert_edit_history,
    insert_plant_part,
    replace_session_sources,
    update_detection_bbox,
    update_plant_part,
    update_session,
)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.sqlite3"
    os.environ["SEEDING_DB_PATH"] = str(path)
    database.initialize_database()
    yield path
    del os.environ["SEEDING_DB_PATH"]


def test_new_tables_exist(db_path):
    with database.get_connection() as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert "analysis_session" in tables
    assert "detection" in tables
    assert "plant_part" in tables
    assert "edit_history" in tables
    assert "session_source" in tables


def test_initialize_database_migrates_legacy_session_source(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    os.environ["SEEDING_DB_PATH"] = str(path)
    try:
        with sqlite3.connect(path) as conn:
            conn.execute(
                """
                CREATE TABLE session_source (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    page_index INTEGER NOT NULL,
                    source_path TEXT NOT NULL
                )
                """
            )

        database.initialize_database()

        with database.get_connection() as conn:
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(session_source)").fetchall()
            }
        assert "rotation_deg" in columns
    finally:
        del os.environ["SEEDING_DB_PATH"]


def test_insert_and_fetch_session(db_path):
    user_id = database.insert_user("op1", "hash")
    sid = insert_analysis_session(
        user_id=user_id,
        source_path="/data/scan.pdf",
        page_count=3,
        calibration_ppm=10.5,
    )
    assert sid > 0
    row = fetch_session_by_id(sid)
    assert row["source_path"] == "/data/scan.pdf"
    assert row["page_count"] == 3
    assert row["status"] == "active"


def test_fetch_sessions_by_user(db_path):
    user_id = database.insert_user("op_list", "hash")
    insert_analysis_session(user_id=user_id, source_path="/a.pdf")
    insert_analysis_session(user_id=user_id, source_path="/b.pdf")
    rows = fetch_sessions_by_user(user_id)
    assert len(rows) == 2


def test_replace_and_fetch_session_sources(db_path):
    user_id = database.insert_user("op_sources", "hash")
    sid = insert_analysis_session(user_id=user_id, source_path="/a.pdf")

    replace_session_sources(sid, ["/a.pdf", "/b.png"])

    rows = fetch_session_sources(sid)
    assert [row["source_path"] for row in rows] == ["/a.pdf", "/b.png"]
    assert [row["page_index"] for row in rows] == [0, 1]


def test_update_session(db_path):
    user_id = database.insert_user("op_upd", "hash")
    sid = insert_analysis_session(user_id=user_id, source_path="/x.pdf")
    update_session(sid, report_path="/reports/x.pdf", status="archived")
    row = fetch_session_by_id(sid)
    assert row["report_path"] == "/reports/x.pdf"
    assert row["status"] == "archived"


def test_insert_detection_and_parts(db_path):
    user_id = database.insert_user("op2", "hash")
    sid = insert_analysis_session(user_id=user_id, source_path="/img.jpg")
    det_id = insert_detection(
        session_id=sid, page_index=0, object_index=0,
        bbox=(10, 20, 100, 200), confidence=0.91,
        rotation_deg=0.0, orientation_uncertain=False,
    )
    assert det_id > 0
    part_id = insert_plant_part(
        detection_id=det_id, class_name="root", confidence=0.85,
        bbox=(10, 20, 50, 80), polygon_json="[[10,20],[50,20]]",
    )
    assert part_id > 0
    dets = fetch_detections_by_session(sid)
    assert len(dets) == 1
    parts = fetch_parts_by_detection(det_id)
    assert len(parts) == 1
    assert parts[0]["class_name"] == "root"


def test_delete_analysis_session_cascades_detections_and_parts(db_path):
    user_id = database.insert_user("op_delete", "hash")
    sid = insert_analysis_session(user_id=user_id, source_path="/img.jpg")
    replace_session_sources(sid, ["/img.jpg"])
    det_id = insert_detection(session_id=sid, page_index=0, object_index=0)
    insert_plant_part(detection_id=det_id, class_name="root")

    assert delete_analysis_session(sid, user_id=user_id) is True

    assert fetch_session_by_id(sid) is None
    assert fetch_session_sources(sid) == []
    assert fetch_detections_by_session(sid) == []
    assert fetch_parts_by_detection(det_id) == []


def test_delete_analysis_session_respects_user_id(db_path):
    owner_id = database.insert_user("op_owner", "hash")
    other_id = database.insert_user("op_other", "hash")
    sid = insert_analysis_session(user_id=owner_id, source_path="/img.jpg")

    assert delete_analysis_session(sid, user_id=other_id) is False
    assert fetch_session_by_id(sid) is not None


def test_update_detection_bbox(db_path):
    user_id = database.insert_user("op_bbox", "hash")
    sid = insert_analysis_session(user_id=user_id, source_path="/img.jpg")
    det_id = insert_detection(session_id=sid, page_index=0, object_index=0,
                               bbox=(10, 20, 100, 200), confidence=0.9)
    update_detection_bbox(det_id, bbox=(15, 25, 110, 210))
    dets = fetch_detections_by_session(sid)
    assert dets[0]["bbox_x"] == 15
    assert dets[0]["is_manual"] == 1


def test_update_plant_part(db_path):
    user_id = database.insert_user("op_part", "hash")
    sid = insert_analysis_session(user_id=user_id, source_path="/img.jpg")
    det_id = insert_detection(session_id=sid, page_index=0, object_index=0,
                               bbox=(0, 0, 100, 100), confidence=0.9)
    part_id = insert_plant_part(detection_id=det_id, class_name="stem",
                                 confidence=0.8, bbox=(5, 5, 30, 60))
    update_plant_part(part_id, polygon_json="[[1,2],[3,4]]")
    parts = fetch_parts_by_detection(det_id)
    assert parts[0]["polygon_json"] == "[[1,2],[3,4]]"
    assert parts[0]["is_manual"] == 1


def test_insert_edit_history(db_path):
    user_id = database.insert_user("op3", "hash")
    sid = insert_analysis_session(user_id=user_id, source_path="/img.jpg")
    det_id = insert_detection(session_id=sid, page_index=0, object_index=0,
                               bbox=(10, 20, 100, 200), confidence=0.9)
    eid = insert_edit_history(
        user_id=user_id, target_type="detection", target_id=det_id,
        field="bbox", value_before="[10,20,100,200]", value_after="[15,25,110,210]",
    )
    assert eid > 0
