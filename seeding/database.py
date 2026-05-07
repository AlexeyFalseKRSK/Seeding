"""Вспомогательные функции SQLite для пользователей и простых логов действий."""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

from seeding.config import PROJECT_ROOT

DATA_DIR_NAME = "data"
DATABASE_FILE_NAME = "seeding.sqlite3"
SCHEMA_FILE_NAME = "schema.sql"


class DatabaseError(RuntimeError):
    """Выбрасывается, когда локальное SQLite-хранилище недоступно."""


def get_data_dir() -> Path:
    """Возвращает каталог по умолчанию для SQL-файлов и SQLite-базы."""

    return PROJECT_ROOT / "seeding" / DATA_DIR_NAME


def get_schema_path() -> Path:
    """Возвращает путь к файлу схемы для инициализации SQLite-хранилища."""

    return get_data_dir() / SCHEMA_FILE_NAME


def get_legacy_database_path() -> Path:
    """Возвращает прежнее расположение базы данных в корне проекта."""

    return PROJECT_ROOT / DATABASE_FILE_NAME


def get_database_path() -> Path:
    """Возвращает путь к SQLite-базе данных с учётом конфигурации."""

    raw_path = os.getenv("SEEDING_DB_PATH")
    if raw_path:
        return Path(raw_path).expanduser()

    database_path = get_data_dir() / DATABASE_FILE_NAME
    _migrate_legacy_database(database_path)
    return database_path


def _migrate_legacy_database(database_path: Path) -> None:
    """Один раз переносит старый файл базы из корня проекта в `seeding/data`."""

    legacy_database_path = get_legacy_database_path()
    if database_path.exists() or not legacy_database_path.exists():
        return

    database_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        legacy_database_path.replace(database_path)
    except OSError as error:
        raise DatabaseError(
            "Failed to move the legacy database into the new data directory."
        ) from error


def _load_schema_sql() -> str:
    """Считывает SQL-схему с диска."""

    schema_path = get_schema_path()
    try:
        return schema_path.read_text(encoding="utf-8")
    except OSError as error:
        raise DatabaseError(
            f"Failed to read the schema file: {schema_path}"
        ) from error


def get_connection() -> sqlite3.Connection:
    """Открывает SQLite-соединение с доступом к строкам по именам колонок."""

    database_path = get_database_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        connection = sqlite3.connect(database_path, timeout=5)
    except sqlite3.Error as error:
        raise DatabaseError(
            f"Failed to open the SQLite database: {database_path}"
        ) from error

    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database() -> None:
    """Создаёт минимальную схему, если она ещё не существует."""

    schema_sql = _load_schema_sql()
    try:
        with closing(get_connection()) as connection:
            connection.executescript(schema_sql)
            connection.commit()
    except sqlite3.Error as error:
        raise DatabaseError("Failed to initialize the SQLite schema.") from error


def fetch_user_by_login(login: str) -> sqlite3.Row | None:
    """Получает одну запись пользователя по логину."""

    try:
        with closing(get_connection()) as connection:
            return connection.execute(
                """
                SELECT id, login, password_hash, created_at, updated_at
                FROM users
                WHERE login = ?
                """,
                (login,),
            ).fetchone()
    except sqlite3.Error as error:
        raise DatabaseError(f"Failed to load user by login: {login}") from error


def fetch_all_users() -> list[sqlite3.Row]:
    """Возвращает всех пользователей, отсортированных по логину."""

    try:
        with closing(get_connection()) as connection:
            rows = connection.execute(
                """
                SELECT id, login, password_hash, created_at, updated_at
                FROM users
                ORDER BY login ASC
                """
            ).fetchall()
    except sqlite3.Error as error:
        raise DatabaseError("Failed to load users list.") from error
    return list(rows)


def count_users() -> int:
    """Возвращает количество настроенных пользователей."""

    try:
        with closing(get_connection()) as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM users"
            ).fetchone()
    except sqlite3.Error as error:
        raise DatabaseError("Failed to count users.") from error
    if row is None:
        return 0
    return int(row["total"])


def insert_user(login: str, password_hash: str) -> int:
    """Добавляет пользователя и возвращает его созданный идентификатор."""

    try:
        with closing(get_connection()) as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (login, password_hash)
                VALUES (?, ?)
                """,
                (login, password_hash),
            )
            connection.commit()
            return int(cursor.lastrowid)
    except sqlite3.Error as error:
        raise DatabaseError(f"Failed to create user: {login}") from error


def update_user_password_hash(user_id: int, password_hash: str) -> bool:
    """Обновляет сохранённый хэш пароля пользователя."""

    try:
        with closing(get_connection()) as connection:
            cursor = connection.execute(
                """
                UPDATE users
                SET password_hash = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (password_hash, user_id),
            )
            connection.commit()
            return int(cursor.rowcount) > 0
    except sqlite3.Error as error:
        raise DatabaseError(
            f"Failed to update password for user id={user_id}"
        ) from error


def delete_user(user_id: int) -> bool:
    """Удаляет пользователя и сообщает, была ли удалена хотя бы одна строка."""

    try:
        with closing(get_connection()) as connection:
            cursor = connection.execute(
                "DELETE FROM users WHERE id = ?",
                (user_id,),
            )
            connection.commit()
            return int(cursor.rowcount) > 0
    except sqlite3.Error as error:
        raise DatabaseError(f"Failed to delete user id={user_id}") from error


def insert_user_log(user_id: int, action: str, details: str | None = None) -> int:
    """Добавляет пользовательское действие в таблицу логов."""

    try:
        with closing(get_connection()) as connection:
            cursor = connection.execute(
                """
                INSERT INTO user_logs (user_id, action, details)
                VALUES (?, ?, ?)
                """,
                (user_id, action, details),
            )
            connection.commit()
            return int(cursor.lastrowid)
    except sqlite3.Error as error:
        raise DatabaseError(
            f"Failed to write a log record for user id={user_id}"
        ) from error


def fetch_user_logs(user_id: int, limit: int = 100) -> list[sqlite3.Row]:
    """Возвращает последние записи лога для одного пользователя."""

    try:
        with closing(get_connection()) as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, action, details, created_at
                FROM user_logs
                WHERE user_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
    except sqlite3.Error as error:
        raise DatabaseError(
            f"Failed to load logs for user id={user_id}"
        ) from error
    return list(rows)


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
    fields: list[str] = []
    values: list = []
    if report_path is not None:
        fields.append("report_path = ?")
        values.append(report_path)
    if calibration_ppm is not None:
        fields.append("calibration_ppm = ?")
        values.append(calibration_ppm)
    if status is not None:
        fields.append("status = ?")
        values.append(status)
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
        raise DatabaseError(
            f"Failed to fetch detections for session id={session_id}."
        ) from error


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
        raise DatabaseError(
            f"Failed to fetch parts for detection id={detection_id}."
        ) from error


def update_plant_part(
    part_id: int,
    bbox: tuple[int, int, int, int] | None = None,
    polygon_json: str | None = None,
) -> None:
    """Обновляет bbox/маску части растения (после ручной правки)."""
    fields: list[str] = []
    values: list = []
    if bbox is not None:
        bx, by, bw, bh = bbox
        fields += ["bbox_x=?", "bbox_y=?", "bbox_w=?", "bbox_h=?"]
        values += [bx, by, bw, bh]
    if polygon_json is not None:
        fields.append("polygon_json=?")
        values.append(polygon_json)
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
