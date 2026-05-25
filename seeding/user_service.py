"""Сервисный слой для управления пользователями и простого логирования действий."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from seeding.auth import AuthUser, hash_password, verify_password
from seeding.database import (
    DatabaseError,
    count_users,
    fetch_all_users,
    fetch_user_by_login,
    initialize_database,
    insert_user,
    insert_user_log,
    update_user_password_hash,
)
from seeding.database import (
    delete_user as delete_user_row,
)
from seeding.database import (
    fetch_user_logs as fetch_user_logs_rows,
)

LOGIN_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,49}$")
MIN_PASSWORD_LENGTH = 6
MAX_PASSWORD_LENGTH = 128


class UserServiceError(RuntimeError):
    """Базовая ошибка сервисного слоя пользовательских операций."""


class StorageUnavailableError(UserServiceError):
    """Выбрасывается, когда SQLite-хранилище недоступно для работы."""


class UserValidationError(UserServiceError):
    """Выбрасывается, когда пользовательский ввод не проходит валидацию."""


class DuplicateUserError(UserServiceError):
    """Выбрасывается, когда логин уже существует."""


class UserNotFoundError(UserServiceError):
    """Выбрасывается, когда ожидаемая запись пользователя не найдена."""


@dataclass(frozen=True, slots=True)
class UserSummary:
    """Минимальное сериализуемое представление сохранённого пользователя."""

    id: int
    login: str
    created_at: str
    updated_at: str


def _normalize_login(login: str) -> str:
    """Нормализует логин для хранения и поиска."""

    return login.strip().casefold()


def _handle_storage_error(error: DatabaseError) -> StorageUnavailableError:
    """Преобразует низкоуровневые ошибки БД в ошибки сервисного слоя."""

    return StorageUnavailableError(str(error))


def _row_to_auth_user(row: Any) -> AuthUser:
    """Преобразует строку БД в объект аутентифицированного пользователя."""

    return AuthUser(
        id=int(row["id"]),
        login=str(row["login"]),
    )


def _row_to_user_summary(row: Any) -> UserSummary:
    """Преобразует строку БД в облегчённое описание пользователя."""

    return UserSummary(
        id=int(row["id"]),
        login=str(row["login"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def validate_login(login: str) -> str:
    """Проверяет и нормализует логин."""

    normalized_login = _normalize_login(login)
    if not normalized_login:
        raise UserValidationError("Логин не должен быть пустым.")
    if not LOGIN_PATTERN.fullmatch(normalized_login):
        raise UserValidationError(
            "Логин должен содержать 3-50 символов: латиницу, цифры, '.', '_' или '-'."
        )
    return normalized_login


def validate_password(password: str) -> str:
    """Проверяет пароль в открытом виде."""

    if not password:
        raise UserValidationError("Пароль не должен быть пустым.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise UserValidationError(
            f"Пароль должен содержать не менее {MIN_PASSWORD_LENGTH} символов."
        )
    if len(password) > MAX_PASSWORD_LENGTH:
        raise UserValidationError(
            f"Пароль не должен превышать {MAX_PASSWORD_LENGTH} символов."
        )
    return password


def initialize_user_storage() -> None:
    """Инициализирует SQLite-схему, используемую приложением."""

    try:
        initialize_database()
    except DatabaseError as error:
        raise _handle_storage_error(error) from error


def has_users() -> bool:
    """Возвращает признак того, что в системе есть хотя бы один пользователь."""

    initialize_user_storage()
    try:
        return count_users() > 0
    except DatabaseError as error:
        raise _handle_storage_error(error) from error


def list_users() -> list[UserSummary]:
    """Возвращает список всех настроенных пользователей."""

    initialize_user_storage()
    try:
        return [_row_to_user_summary(row) for row in fetch_all_users()]
    except DatabaseError as error:
        raise _handle_storage_error(error) from error


def create_user(login: str, password: str) -> AuthUser:
    """Создаёт нового пользователя в локальной SQLite-базе."""

    initialize_user_storage()
    normalized_login = validate_login(login)
    plain_password = validate_password(password)

    try:
        existing_user = fetch_user_by_login(normalized_login)
    except DatabaseError as error:
        raise _handle_storage_error(error) from error

    if existing_user is not None:
        raise DuplicateUserError(f"Пользователь '{normalized_login}' уже существует.")

    try:
        insert_user(normalized_login, hash_password(plain_password))
        user_row = fetch_user_by_login(normalized_login)
    except DatabaseError as error:
        raise _handle_storage_error(error) from error

    if user_row is None:
        raise StorageUnavailableError("Не удалось загрузить созданного пользователя.")
    return _row_to_auth_user(user_row)


def authenticate_user(login: str, password: str) -> AuthUser | None:
    """Проверяет учётные данные и возвращает найденного пользователя."""

    initialize_user_storage()
    normalized_login = _normalize_login(login)
    if not normalized_login or not password:
        return None

    try:
        user_row = fetch_user_by_login(normalized_login)
    except DatabaseError as error:
        raise _handle_storage_error(error) from error

    if user_row is None:
        return None
    if not verify_password(password, str(user_row["password_hash"])):
        return None
    return _row_to_auth_user(user_row)


def get_user_by_login(login: str) -> AuthUser | None:
    """Загружает пользователя по логину, если он существует."""

    initialize_user_storage()
    normalized_login = _normalize_login(login)
    if not normalized_login:
        return None

    try:
        user_row = fetch_user_by_login(normalized_login)
    except DatabaseError as error:
        raise _handle_storage_error(error) from error

    if user_row is None:
        return None
    return _row_to_auth_user(user_row)


def update_user_password(user_id: int, new_password: str) -> None:
    """Обновляет пароль существующего пользователя."""

    validate_password(new_password)
    initialize_user_storage()
    try:
        updated = update_user_password_hash(user_id, hash_password(new_password))
    except DatabaseError as error:
        raise _handle_storage_error(error) from error

    if not updated:
        raise UserNotFoundError(f"Пользователь с id={user_id} не найден.")


def update_user_password_by_login(login: str, new_password: str) -> None:
    """Обновляет пароль пользователя, найденного по логину."""

    user = get_user_by_login(login)
    if user is None:
        raise UserNotFoundError(f"Пользователь '{_normalize_login(login)}' не найден.")
    update_user_password(user.id, new_password)


def delete_user_by_login(login: str) -> None:
    """Удаляет пользователя по логину."""

    user = get_user_by_login(login)
    if user is None:
        raise UserNotFoundError(f"Пользователь '{_normalize_login(login)}' не найден.")

    try:
        deleted = delete_user_row(user.id)
    except DatabaseError as error:
        raise _handle_storage_error(error) from error

    if not deleted:
        raise UserNotFoundError(f"Пользователь '{user.login}' не найден.")


def record_user_action(user_id: int, action: str, details: str | None = None) -> None:
    """Записывает минимальную запись о действии пользователя в БД."""

    if user_id <= 0:
        raise UserValidationError("Идентификатор пользователя должен быть положительным.")
    if not action or not action.strip():
        raise UserValidationError("Действие не должно быть пустым.")

    initialize_user_storage()
    normalized_details = details.strip() if details and details.strip() else None
    try:
        insert_user_log(user_id, action.strip(), normalized_details)
    except DatabaseError as error:
        raise _handle_storage_error(error) from error


def fetch_user_logs(user_id: int, limit: int = 100) -> list[dict[str, Any]]:
    """Возвращает последние пользовательские логи в сериализуемом формате."""

    if user_id <= 0:
        raise UserValidationError("Идентификатор пользователя должен быть положительным.")
    if limit <= 0:
        raise UserValidationError("Лимит логов должен быть положительным.")

    initialize_user_storage()
    try:
        rows = fetch_user_logs_rows(user_id, limit=limit)
    except DatabaseError as error:
        raise _handle_storage_error(error) from error
    return [dict(row) for row in rows]

