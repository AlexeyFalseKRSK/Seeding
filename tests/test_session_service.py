import os

import numpy as np
import pytest

from seeding import database
from seeding.image_loader import restore_session_source_images
from seeding.models import AllClassImage, AppState, ObjectImage, OriginalImage
from seeding.services import rotate_page
from seeding.session_service import load_session, record_edit, save_session


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.sqlite3"
    os.environ["SEEDING_DB_PATH"] = str(path)
    database.initialize_database()
    yield path
    del os.environ["SEEDING_DB_PATH"]


def _make_state(source_path: str, user_id: int) -> AppState:
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    part = AllClassImage(
        class_name="root",
        confidence=0.88,
        image=dummy_img,
        bbox=(5, 5, 30, 40),
        mask_polygon=np.array([[5, 5], [35, 5], [35, 45], [5, 45]]),
    )
    obj = ObjectImage(
        class_name="seeding",
        confidence=0.92,
        image=[dummy_img],
        image_all_class=[part],
        bbox=(10, 20, 100, 150),
        rotation_k=0,
    )
    orig = OriginalImage(
        file_path=source_path,
        source_files=[source_path],
        images=[dummy_img],
        class_object_image=[[obj]],
    )
    state = AppState(image_storage=orig, pixels_per_mm=11.0)
    state.session_id = None
    state.current_user_id = user_id
    return state


def test_save_session_creates_records(db_path):
    user_id = database.insert_user("tester", "hash")
    state = _make_state("/data/scan.pdf", user_id)
    session_id = save_session(state)
    assert session_id > 0
    assert state.session_id == session_id
    dets = database.fetch_detections_by_session(session_id)
    assert len(dets) == 1
    parts = database.fetch_parts_by_detection(dets[0]["id"])
    assert len(parts) == 1
    assert parts[0]["class_name"] == "root"


def test_save_session_idempotent(db_path):
    user_id = database.insert_user("tester_idem", "hash")
    state = _make_state("/data/scan.pdf", user_id)
    sid1 = save_session(state)
    state.last_report_path = "/reports/out.pdf"
    sid2 = save_session(state)
    assert sid1 == sid2
    row = database.fetch_session_by_id(sid1)
    assert row["report_path"] == "/reports/out.pdf"


def test_save_session_update_replaces_detections(db_path):
    user_id = database.insert_user("tester_replace", "hash")
    state = _make_state("/data/scan.pdf", user_id)
    session_id = save_session(state)

    state.image_storage.class_object_image = [[]]
    save_session(state)

    assert database.fetch_detections_by_session(session_id) == []


def test_save_session_requires_authorized_user(db_path):
    user_id = database.insert_user("tester_auth", "hash")
    state = _make_state("/data/scan.pdf", user_id)
    state.current_user_id = None

    with pytest.raises(ValueError, match="авториз|authorized|user"):
        save_session(state)


def test_load_session_restores_state(db_path, tmp_path):
    src = str(tmp_path / "scan.pdf")
    open(src, "w").close()
    user_id = database.insert_user("tester2", "hash")
    state = _make_state(src, user_id)
    session_id = save_session(state)
    restored = load_session(session_id)
    assert restored is not None
    assert restored.image_storage.file_path == src
    assert len(restored.image_storage.class_object_image) == 1
    obj = restored.image_storage.class_object_image[0][0]
    assert obj.confidence == pytest.approx(0.92)
    assert obj.image_all_class[0].class_name == "root"
    assert obj.image_all_class[0].mask_polygon is not None


def test_load_session_missing_file(db_path):
    user_id = database.insert_user("tester3", "hash")
    state = _make_state("/nonexistent/file.pdf", user_id)
    session_id = save_session(state)
    restored = load_session(session_id)
    assert restored is not None
    assert hasattr(restored, "_missing_source")
    assert restored._missing_source == "/nonexistent/file.pdf"
    assert restored.image_storage.source_files == ["/nonexistent/file.pdf"]


def test_save_and_load_session_preserves_page_source_files(db_path):
    user_id = database.insert_user("tester_sources", "hash")
    state = _make_state("/data/first.png", user_id)
    state.image_storage.images = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((12, 12, 3), dtype=np.uint8),
    ]
    state.image_storage.source_files = ["/data/first.png", "/data/second.png"]
    state.image_storage.class_object_image = [[], []]

    session_id = save_session(state)
    restored = load_session(session_id)

    assert restored is not None
    assert restored.image_storage.file_path == "/data/first.png"
    assert restored.image_storage.source_files == ["/data/first.png", "/data/second.png"]


def test_save_load_and_restore_preserves_page_rotation(db_path, tmp_path):
    cv2 = pytest.importorskip("cv2")
    user_id = database.insert_user("tester_rotation", "hash")
    source = tmp_path / "rotated-source.png"
    original = np.zeros((4, 6, 3), dtype=np.uint8)
    original[1:3, 2:5] = (10, 80, 200)
    assert cv2.imwrite(str(source), original)

    obj = ObjectImage(
        class_name="seeding",
        confidence=0.95,
        image=[original[1:3, 2:5].copy()],
        bbox=(2, 1, 5, 3),
    )
    state = AppState(
        image_storage=OriginalImage(
            file_path=str(source),
            source_files=[str(source)],
            images=[original.copy()],
            class_object_image=[[obj]],
        ),
        pixels_per_mm=11.0,
        current_user_id=user_id,
    )
    rotated = rotate_page(state.image_storage, 0, angle=90, rotate_k=-1)
    rotated_bbox = obj.bbox

    session_id = save_session(state)
    restored = load_session(session_id)
    assert restored is not None
    assert restored.image_storage.page_rotation_k == [1]

    missing = restore_session_source_images(restored)

    assert missing is None
    assert np.array_equal(restored.image_storage.images[0], rotated)
    restored_obj = restored.image_storage.class_object_image[0][0]
    assert restored_obj.bbox == rotated_bbox
    x1, y1, x2, y2 = rotated_bbox
    assert np.array_equal(restored_obj.image[0], rotated[y1:y2, x1:x2])


def test_load_session_not_found(db_path):
    result = load_session(99999)
    assert result is None


def test_record_edit(db_path):
    user_id = database.insert_user("editor", "hash")
    state = _make_state("/img.pdf", user_id)
    save_session(state)
    dets = database.fetch_detections_by_session(state.session_id)
    det_id = dets[0]["id"]
    record_edit(
        state=state,
        target_type="detection",
        target_id=det_id,
        field="bbox",
        value_before=(10, 20, 100, 150),
        value_after=(15, 25, 110, 160),
    )
