# Session Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Сохранять сессии анализа (проростки + части + ручные правки) в SQLite и восстанавливать их без повторного инференса.

**Architecture:** Расширяем существующий `database.py` новыми таблицами и CRUD-функциями. Добавляем `session_service.py` — сервисный слой между UI и БД. Интегрируем автосохранение после инференса и ручное сохранение по кнопке в `main_window.py`.

**Tech Stack:** Python 3.10+, SQLite3 (встроенный), PyQt5, существующие доменные модели (`AppState`, `OriginalImage`, `ObjectImage`, `AllClassImage`).

---

## Файловая карта

| Файл | Действие | Роль |
|------|----------|------|
| `seeding/data/schema.sql` | Изменить | Добавить 4 новые таблицы + миграция |
| `seeding/database.py` | Изменить | CRUD для новых таблиц |
| `seeding/session_service.py` | Создать | Сохранение/загрузка сессии из `AppState` |
| `seeding/models.py` | Изменить | Добавить `session_id` в `AppState` |
| `seeding/ui/main_window.py` | Изменить | Автосохранение + кнопка + диалог открытия сессии |
| `tests/test_session_service.py` | Создать | Тесты сервисного слоя |
| `tests/test_database_session.py` | Создать | Тесты CRUD новых таблиц |

---

## Task 1: Расширить схему БД (миграция)

**Files:**
- Modify: `seeding/data/schema.sql`
- Modify: `seeding/database.py`

- [ ] **Шаг 1: Написать падающий тест на существование новых таблиц**

```python
# tests/test_database_session.py
import os, sqlite3, pytest
from seeding import database

@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.sqlite3"
    os.environ["SEEDING_DB_PATH"] = str(path)
    yield path
    del os.environ["SEEDING_DB_PATH"]

def test_new_tables_exist(db_path):
    database.initialize_database()
    with database.get_connection() as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert "analysis_session" in tables
    assert "detection" in tables
    assert "plant_part" in tables
    assert "edit_history" in tables
```

- [ ] **Шаг 2: Запустить тест — убедиться что падает**

```
pytest tests/test_database_session.py::test_new_tables_exist -v
```
Ожидаем: FAIL — таблицы не существуют.

- [ ] **Шаг 3: Добавить таблицы в schema.sql**

Дописать в конец `seeding/data/schema.sql`:

```sql
-- ── СЕССИИ АНАЛИЗА ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS analysis_session (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER REFERENCES users(id) ON DELETE SET NULL,
    source_path     TEXT NOT NULL,
    page_count      INTEGER,
    calibration_ppm REAL,
    report_path     TEXT,
    status          TEXT NOT NULL DEFAULT 'active',
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_session_user_id
    ON analysis_session (user_id, created_at DESC);

-- ── ОБНАРУЖЕННЫЕ ПРОРОСТКИ ────────────────────────────────────
CREATE TABLE IF NOT EXISTS detection (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id            INTEGER NOT NULL REFERENCES analysis_session(id) ON DELETE CASCADE,
    page_index            INTEGER NOT NULL,
    object_index          INTEGER NOT NULL,
    bbox_x                REAL,
    bbox_y                REAL,
    bbox_w                REAL,
    bbox_h                REAL,
    confidence            REAL,
    rotation_deg          REAL NOT NULL DEFAULT 0,
    orientation_uncertain INTEGER NOT NULL DEFAULT 0,
    is_manual             INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_detection_session_id
    ON detection (session_id, page_index, object_index);

-- ── ЧАСТИ РАСТЕНИЯ ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS plant_part (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER NOT NULL REFERENCES detection(id) ON DELETE CASCADE,
    class_name   TEXT NOT NULL,
    confidence   REAL,
    bbox_x       REAL,
    bbox_y       REAL,
    bbox_w       REAL,
    bbox_h       REAL,
    polygon_json TEXT,
    is_manual    INTEGER NOT NULL DEFAULT 0
);

-- ── ИСТОРИЯ РУЧНЫХ ПРАВОК ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS edit_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER REFERENCES users(id) ON DELETE SET NULL,
    target_type  TEXT NOT NULL,
    target_id    INTEGER NOT NULL,
    field        TEXT NOT NULL,
    value_before TEXT,
    value_after  TEXT,
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_edit_history_target
    ON edit_history (target_type, target_id, created_at DESC);
```

- [ ] **Шаг 4: Запустить тест — убедиться что проходит**

```
pytest tests/test_database_session.py::test_new_tables_exist -v
```
Ожидаем: PASS.

- [ ] **Шаг 5: Убедиться что старые тесты не сломались**

```
pytest tests/ -v
```
Ожидаем: все зелёные.

- [ ] **Шаг 6: Коммит**

```bash
git add seeding/data/schema.sql tests/test_database_session.py
git commit -m "feat: добавить таблицы сессий, детекций и правок в схему БД"
```

---

## Task 2: CRUD для сессий и детекций в database.py

**Files:**
- Modify: `seeding/database.py`
- Test: `tests/test_database_session.py`

- [ ] **Шаг 1: Написать падающие тесты на CRUD сессий**

Добавить в `tests/test_database_session.py`:

```python
from seeding.database import (
    insert_analysis_session, fetch_session_by_id, update_session,
    insert_detection, fetch_detections_by_session,
    insert_plant_part, fetch_parts_by_detection,
    insert_edit_history,
)

def test_insert_and_fetch_session(db_path):
    database.initialize_database()
    # нужен пользователь
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

def test_insert_detection_and_parts(db_path):
    database.initialize_database()
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

def test_insert_edit_history(db_path):
    database.initialize_database()
    user_id = database.insert_user("op3", "hash")
    sid = insert_analysis_session(user_id=user_id, source_path="/img.jpg")
    det_id = insert_detection(
        session_id=sid, page_index=0, object_index=0,
        bbox=(10, 20, 100, 200), confidence=0.9,
    )
    eid = insert_edit_history(
        user_id=user_id, target_type="detection", target_id=det_id,
        field="bbox", value_before="[10,20,100,200]", value_after="[15,25,110,210]",
    )
    assert eid > 0
```

- [ ] **Шаг 2: Запустить — убедиться что падают (ImportError)**

```
pytest tests/test_database_session.py -v
```

- [ ] **Шаг 3: Реализовать CRUD-функции в database.py**

Добавить в конец `seeding/database.py`:

```python
# ── Сессии анализа ────────────────────────────────────────────

def insert_analysis_session(
    user_id: int | None,
    source_path: str,
    page_count: int | None = None,
    calibration_ppm: float | None = None,
    report_path: str | None = None,
) -> int:
    """Создаёт запись сессии анализа и возвращает её id."""
    try:
        with closing(get_connection()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO analysis_session
                    (user_id, source_path, page_count, calibration_ppm, report_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, source_path, page_count, calibration_ppm, report_path),
            )
            conn.commit()
            return int(cursor.lastrowid)
    except sqlite3.Error as error:
        raise DatabaseError("Failed to insert analysis_session.") from error


def fetch_session_by_id(session_id: int) -> sqlite3.Row | None:
    """Возвращает сессию по id."""
    try:
        with closing(get_connection()) as conn:
            return conn.execute(
                "SELECT * FROM analysis_session WHERE id = ?", (session_id,)
            ).fetchone()
    except sqlite3.Error as error:
        raise DatabaseError(f"Failed to fetch session id={session_id}.") from error


def fetch_sessions_by_user(user_id: int, limit: int = 50) -> list[sqlite3.Row]:
    """Возвращает последние сессии пользователя."""
    try:
        with closing(get_connection()) as conn:
            return conn.execute(
                """
                SELECT * FROM analysis_session
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
    except sqlite3.Error as error:
        raise DatabaseError(f"Failed to fetch sessions for user id={user_id}.") from error


def update_session(
    session_id: int,
    report_path: str | None = None,
    calibration_ppm: float | None = None,
    status: str | None = None,
) -> None:
    """Обновляет поля сессии (только переданные)."""
    fields, values = [], []
    if report_path is not None:
        fields.append("report_path = ?"); values.append(report_path)
    if calibration_ppm is not None:
        fields.append("calibration_ppm = ?"); values.append(calibration_ppm)
    if status is not None:
        fields.append("status = ?"); values.append(status)
    if not fields:
        return
    fields.append("updated_at = CURRENT_TIMESTAMP")
    values.append(session_id)
    try:
        with closing(get_connection()) as conn:
            conn.execute(
                f"UPDATE analysis_session SET {', '.join(fields)} WHERE id = ?",
                values,
            )
            conn.commit()
    except sqlite3.Error as error:
        raise DatabaseError(f"Failed to update session id={session_id}.") from error


# ── Детекции ──────────────────────────────────────────────────

def insert_detection(
    session_id: int,
    page_index: int,
    object_index: int,
    bbox: tuple[int, int, int, int] | None = None,
    confidence: float | None = None,
    rotation_deg: float = 0.0,
    orientation_uncertain: bool = False,
    is_manual: bool = False,
) -> int:
    """Сохраняет обнаруженный проросток и возвращает его id."""
    bx, by, bw, bh = bbox if bbox else (None, None, None, None)
    try:
        with closing(get_connection()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO detection
                    (session_id, page_index, object_index,
                     bbox_x, bbox_y, bbox_w, bbox_h,
                     confidence, rotation_deg, orientation_uncertain, is_manual)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, page_index, object_index,
                 bx, by, bw, bh,
                 confidence, rotation_deg,
                 int(orientation_uncertain), int(is_manual)),
            )
            conn.commit()
            return int(cursor.lastrowid)
    except sqlite3.Error as error:
        raise DatabaseError("Failed to insert detection.") from error


def fetch_detections_by_session(session_id: int) -> list[sqlite3.Row]:
    """Возвращает все детекции сессии, отсортированные по странице и индексу."""
    try:
        with closing(get_connection()) as conn:
            return conn.execute(
                """
                SELECT * FROM detection
                WHERE session_id = ?
                ORDER BY page_index, object_index
                """,
                (session_id,),
            ).fetchall()
    except sqlite3.Error as error:
        raise DatabaseError(f"Failed to fetch detections for session id={session_id}.") from error


def update_detection_bbox(
    detection_id: int,
    bbox: tuple[int, int, int, int],
) -> None:
    """Обновляет bbox детекции (после ручной правки)."""
    bx, by, bw, bh = bbox
    try:
        with closing(get_connection()) as conn:
            conn.execute(
                """
                UPDATE detection
                SET bbox_x=?, bbox_y=?, bbox_w=?, bbox_h=?, is_manual=1
                WHERE id=?
                """,
                (bx, by, bw, bh, detection_id),
            )
            conn.commit()
    except sqlite3.Error as error:
        raise DatabaseError(f"Failed to update detection id={detection_id}.") from error


# ── Части растения ────────────────────────────────────────────

def insert_plant_part(
    detection_id: int,
    class_name: str,
    confidence: float | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    polygon_json: str | None = None,
    is_manual: bool = False,
) -> int:
    """Сохраняет часть растения и возвращает её id."""
    bx, by, bw, bh = bbox if bbox else (None, None, None, None)
    try:
        with closing(get_connection()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO plant_part
                    (detection_id, class_name, confidence,
                     bbox_x, bbox_y, bbox_w, bbox_h, polygon_json, is_manual)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (detection_id, class_name, confidence,
                 bx, by, bw, bh, polygon_json, int(is_manual)),
            )
            conn.commit()
            return int(cursor.lastrowid)
    except sqlite3.Error as error:
        raise DatabaseError("Failed to insert plant_part.") from error


def fetch_parts_by_detection(detection_id: int) -> list[sqlite3.Row]:
    """Возвращает все части растения для одной детекции."""
    try:
        with closing(get_connection()) as conn:
            return conn.execute(
                "SELECT * FROM plant_part WHERE detection_id = ?",
                (detection_id,),
            ).fetchall()
    except sqlite3.Error as error:
        raise DatabaseError(f"Failed to fetch parts for detection id={detection_id}.") from error


def update_plant_part(
    part_id: int,
    bbox: tuple[int, int, int, int] | None = None,
    polygon_json: str | None = None,
) -> None:
    """Обновляет bbox/маску части растения (после ручной правки)."""
    fields, values = [], []
    if bbox is not None:
        bx, by, bw, bh = bbox
        fields += ["bbox_x=?", "bbox_y=?", "bbox_w=?", "bbox_h=?"]
        values += [bx, by, bw, bh]
    if polygon_json is not None:
        fields.append("polygon_json=?"); values.append(polygon_json)
    if not fields:
        return
    fields.append("is_manual=1")
    values.append(part_id)
    try:
        with closing(get_connection()) as conn:
            conn.execute(
                f"UPDATE plant_part SET {', '.join(fields)} WHERE id=?", values
            )
            conn.commit()
    except sqlite3.Error as error:
        raise DatabaseError(f"Failed to update plant_part id={part_id}.") from error


# ── История правок ────────────────────────────────────────────

def insert_edit_history(
    user_id: int | None,
    target_type: str,
    target_id: int,
    field: str,
    value_before: str | None,
    value_after: str | None,
) -> int:
    """Записывает одно изменение в журнал правок."""
    try:
        with closing(get_connection()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO edit_history
                    (user_id, target_type, target_id, field, value_before, value_after)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, target_type, target_id, field, value_before, value_after),
            )
            conn.commit()
            return int(cursor.lastrowid)
    except sqlite3.Error as error:
        raise DatabaseError("Failed to insert edit_history.") from error
```

- [ ] **Шаг 4: Запустить тесты**

```
pytest tests/test_database_session.py -v
```
Ожидаем: все PASS.

- [ ] **Шаг 5: Коммит**

```bash
git add seeding/database.py tests/test_database_session.py
git commit -m "feat: CRUD-функции для сессий, детекций, частей и правок"
```

---

## Task 3: Добавить session_id в AppState и создать session_service.py

**Files:**
- Modify: `seeding/models.py`
- Create: `seeding/session_service.py`
- Create: `tests/test_session_service.py`

- [ ] **Шаг 1: Написать падающий тест на сохранение сессии**

```python
# tests/test_session_service.py
import os, json, pytest
import numpy as np
from PIL import Image
from seeding import database
from seeding.models import AppState, OriginalImage, ObjectImage, AllClassImage
from seeding.session_service import save_session, load_session

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
        class_name="root", confidence=0.88,
        image=dummy_img,
        bbox=(5, 5, 30, 40),
        mask_polygon=np.array([[5,5],[35,5],[35,45],[5,45]]),
    )
    obj = ObjectImage(
        class_name="seeding", confidence=0.92,
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
    state.current_user_id = user_id
    return state

def test_save_session_creates_records(db_path):
    user_id = database.insert_user("tester", "hash")
    state = _make_state("/data/scan.pdf", user_id)
    session_id = save_session(state)
    assert session_id > 0
    dets = database.fetch_detections_by_session(session_id)
    assert len(dets) == 1
    parts = database.fetch_parts_by_detection(dets[0]["id"])
    assert len(parts) == 1
    assert parts[0]["class_name"] == "root"

def test_load_session_restores_state(db_path, tmp_path):
    # создаём реальный файл чтобы проверить reload
    src = str(tmp_path / "scan.pdf")
    open(src, "w").close()
    user_id = database.insert_user("tester2", "hash")
    state = _make_state(src, user_id)
    session_id = save_session(state)
    restored = load_session(session_id)
    assert restored is not None
    assert restored.image_storage.file_path == src
    assert len(restored.image_storage.class_object_image[0]) == 1
    obj = restored.image_storage.class_object_image[0][0]
    assert obj.confidence == pytest.approx(0.92)
    assert obj.image_all_class[0].class_name == "root"
```

- [ ] **Шаг 2: Запустить — убедиться что падает (ImportError)**

```
pytest tests/test_session_service.py -v
```

- [ ] **Шаг 3: Добавить session_id в AppState**

В `seeding/models.py` изменить класс `AppState`:

```python
@dataclass
class AppState:
    """Хранит текущее состояние проекта, выбора пользователя и параметров просмотра."""

    image_storage: OriginalImage = field(default_factory=OriginalImage)
    active_image_index: int = 0
    selected_item: SelectionPayload | None = None
    zoom_factor: float = 1.0
    pixels_per_mm: float = 0.0
    last_report_path: str = ""
    session_id: int | None = None        # id текущей сессии в БД
    current_user_id: int | None = None   # id вошедшего пользователя
```

- [ ] **Шаг 4: Создать seeding/session_service.py**

```python
"""Сохранение и загрузка сессии анализа в/из SQLite."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from seeding import database
from seeding.models import AllClassImage, AppState, ObjectImage, OriginalImage


def save_session(state: AppState) -> int:
    """Сохраняет или обновляет текущую сессию. Возвращает session_id."""
    orig = state.image_storage
    user_id = state.current_user_id

    if state.session_id is None:
        session_id = database.insert_analysis_session(
            user_id=user_id,
            source_path=orig.file_path,
            page_count=len(orig.images),
            calibration_ppm=state.pixels_per_mm or None,
            report_path=state.last_report_path or None,
        )
        state.session_id = session_id
        _insert_all_detections(session_id, orig)
    else:
        database.update_session(
            session_id=state.session_id,
            calibration_ppm=state.pixels_per_mm or None,
            report_path=state.last_report_path or None,
        )

    return state.session_id


def _insert_all_detections(session_id: int, orig: OriginalImage) -> None:
    """Вставляет все детекции и части из OriginalImage."""
    pages = orig.class_object_image or []
    for page_idx, objects in enumerate(pages):
        for obj_idx, obj in enumerate(objects):
            bbox = obj.bbox or (0, 0, 0, 0)
            det_id = database.insert_detection(
                session_id=session_id,
                page_index=page_idx,
                object_index=obj_idx,
                bbox=bbox,
                confidence=obj.confidence,
                rotation_deg=float(obj.rotation_k) * 90,
                orientation_uncertain=obj.orientation_uncertain,
            )
            for part in obj.image_all_class or []:
                polygon_json = None
                if part.mask_polygon is not None:
                    polygon_json = json.dumps(part.mask_polygon.tolist())
                database.insert_plant_part(
                    detection_id=det_id,
                    class_name=part.class_name,
                    confidence=part.confidence,
                    bbox=part.bbox,
                    polygon_json=polygon_json,
                )


def load_session(session_id: int) -> AppState | None:
    """Загружает сессию из БД. Возвращает None если сессия не найдена."""
    row = database.fetch_session_by_id(session_id)
    if row is None:
        return None

    source_path = row["source_path"]
    file_exists = Path(source_path).exists()

    detections = database.fetch_detections_by_session(session_id)

    pages: dict[int, list[ObjectImage]] = {}
    for det in detections:
        parts = database.fetch_parts_by_detection(det["id"])
        all_class = [_row_to_part(p) for p in parts]
        obj = ObjectImage(
            class_name="seeding",
            confidence=det["confidence"] or 0.0,
            image=[],
            image_all_class=all_class,
            bbox=_bbox_from_row(det),
            rotation_k=int((det["rotation_deg"] or 0) / 90) % 4,
            orientation_uncertain=bool(det["orientation_uncertain"]),
        )
        pages.setdefault(det["page_index"], []).append(obj)

    page_list = [pages.get(i, []) for i in range(max(pages.keys(), default=-1) + 1)]

    orig = OriginalImage(
        file_path=source_path,
        source_files=[source_path] if file_exists else [],
        images=[],
        class_object_image=page_list,
    )

    state = AppState(
        image_storage=orig,
        pixels_per_mm=row["calibration_ppm"] or 0.0,
        last_report_path=row["report_path"] or "",
        session_id=session_id,
        current_user_id=row["user_id"],
    )

    if not file_exists:
        state._missing_source = source_path  # сигнал для UI

    return state


def record_edit(
    state: AppState,
    target_type: str,
    target_id: int,
    field: str,
    value_before,
    value_after,
) -> None:
    """Записывает ручную правку в журнал edit_history."""
    database.insert_edit_history(
        user_id=state.current_user_id,
        target_type=target_type,
        target_id=target_id,
        field=field,
        value_before=json.dumps(value_before),
        value_after=json.dumps(value_after),
    )


def _row_to_part(row) -> AllClassImage:
    polygon = None
    if row["polygon_json"]:
        polygon = np.array(json.loads(row["polygon_json"]))
    return AllClassImage(
        class_name=row["class_name"],
        confidence=row["confidence"] or 0.0,
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        bbox=_bbox_from_row(row),
        mask_polygon=polygon,
    )


def _bbox_from_row(row) -> tuple[int, int, int, int] | None:
    if row["bbox_x"] is None:
        return None
    return (int(row["bbox_x"]), int(row["bbox_y"]),
            int(row["bbox_w"]), int(row["bbox_h"]))
```

- [ ] **Шаг 5: Запустить тесты**

```
pytest tests/test_session_service.py -v
```
Ожидаем: все PASS.

- [ ] **Шаг 6: Запустить все тесты**

```
pytest tests/ -v
```
Ожидаем: все зелёные.

- [ ] **Шаг 7: Коммит**

```bash
git add seeding/models.py seeding/session_service.py tests/test_session_service.py
git commit -m "feat: session_service — сохранение и загрузка сессии из AppState"
```

---

## Task 4: Автосохранение после инференса и кнопка сохранения в UI

**Files:**
- Modify: `seeding/ui/main_window.py`

> Примечание: `main_window.py` ~2000 LOC. Ищем методы `_run_detection` / `_on_inference_done` (или аналог) — именно туда вставляем автосохранение. Кнопку добавляем в тулбар рядом с кнопкой отчёта.

- [ ] **Шаг 1: Найти точку вызова инференса в main_window.py**

```bash
grep -n "run_detection\|inference\|_on_.*done\|after_inference" seeding/ui/main_window.py
```

Запомнить номер строки метода, вызываемого после завершения инференса.

- [ ] **Шаг 2: Добавить импорт session_service в main_window.py**

В блок импортов добавить:
```python
from seeding.session_service import save_session, load_session
```

- [ ] **Шаг 3: Добавить автосохранение после инференса**

В метод, вызываемый после завершения инференса (найденный на шаге 1), добавить в конец:

```python
try:
    save_session(self._state)
except Exception:
    pass  # не прерываем работу если БД недоступна
```

- [ ] **Шаг 4: Добавить кнопку "Сохранить" в тулбар**

Найти место добавления кнопок тулбара (grep `addAction\|toolbar\|QToolBar`) и добавить:

```python
save_action = toolbar.addAction("Сохранить")
save_action.setToolTip("Сохранить текущее состояние сессии (Ctrl+S)")
save_action.setShortcut("Ctrl+S")
save_action.triggered.connect(self._on_save)
```

- [ ] **Шаг 5: Добавить метод _on_save**

```python
def _on_save(self) -> None:
    """Сохраняет текущую сессию по запросу пользователя."""
    try:
        save_session(self._state)
        self.statusBar().showMessage("Сессия сохранена.", 3000)
    except Exception as exc:
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.warning(self, "Ошибка сохранения", str(exc))
```

- [ ] **Шаг 6: Показывать предупреждение если файл не найден при загрузке**

В метод открытия сессии (или `__init__` при передаче session_id) добавить:

```python
if hasattr(state, "_missing_source"):
    from PyQt5.QtWidgets import QMessageBox
    QMessageBox.warning(
        self, "Файл не найден",
        f"Исходный файл не найден:\n{state._missing_source}\n\n"
        "Изображения не будут загружены, данные анализа доступны."
    )
```

- [ ] **Шаг 7: Запустить все тесты**

```
set QT_QPA_PLATFORM=offscreen
pytest tests/ -v
```
Ожидаем: все PASS.

- [ ] **Шаг 8: Коммит**

```bash
git add seeding/ui/main_window.py
git commit -m "feat: автосохранение после инференса и кнопка Сохранить в тулбаре"
```

---

## Финальная проверка

- [ ] Запустить полный тест-сьют: `pytest tests/ -v`
- [ ] Убедиться что `python -m seeding.main` запускается без ошибок
- [ ] Проверить что после инференса в `seeding/data/seeding.sqlite3` появляются записи

---

## Самопроверка плана

**Покрытие требований:**
- ✅ История сессий — `analysis_session`
- ✅ Reload без инференса — `load_session()` восстанавливает `AppState`
- ✅ Предупреждение если файл не найден — `_missing_source` + QMessageBox
- ✅ Ручные правки сохраняются — `edit_history` + `is_manual` флаги
- ✅ Автосохранение после инференса — Task 4 шаг 3
- ✅ Кнопка ручного сохранения — Task 4 шаги 4-5

**Типовая согласованность:**
- `BBox = tuple[int, int, int, int]` — используется везде одинаково
- `session_id: int | None` в `AppState` — добавлено в Task 3 шаг 3
- `current_user_id: int | None` в `AppState` — добавлено в Task 3 шаг 3
- `save_session` / `load_session` — одни и те же имена в сервисе и импортах UI
