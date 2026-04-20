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
