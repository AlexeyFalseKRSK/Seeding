"""CLI-утилиты для управления локальными пользователями SQLite."""

from __future__ import annotations

import argparse
import getpass
import sys

from seeding.database import get_database_path
from seeding.user_service import (
    DuplicateUserError,
    StorageUnavailableError,
    UserNotFoundError,
    UserValidationError,
    create_user,
    delete_user_by_login,
    list_users,
    update_user_password_by_login,
)


def _read_password(confirm: bool = True) -> str:
    """Считывает пароль из терминала без отображения на экране."""

    password = getpass.getpass("Пароль: ")
    if not confirm:
        return password

    repeated_password = getpass.getpass("Повторите пароль: ")
    if password != repeated_password:
        raise UserValidationError("Пароли не совпадают.")
    return password


def _build_parser() -> argparse.ArgumentParser:
    """Создаёт CLI-парсер для локального управления пользователями."""

    parser = argparse.ArgumentParser(description="Управление пользователями Seeding")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Создать пользователя")
    create_parser.add_argument("login", help="Логин пользователя")
    create_parser.add_argument(
        "--password",
        help="Пароль пользователя. Если не указан, будет запрошен интерактивно.",
    )

    subparsers.add_parser("list", help="Показать список пользователей")

    password_parser = subparsers.add_parser(
        "set-password",
        help="Изменить пароль пользователя",
    )
    password_parser.add_argument("login", help="Логин пользователя")
    password_parser.add_argument(
        "--password",
        help="Новый пароль. Если не указан, будет запрошен интерактивно.",
    )

    delete_parser = subparsers.add_parser("delete", help="Удалить пользователя")
    delete_parser.add_argument("login", help="Логин пользователя")
    delete_parser.add_argument(
        "--yes",
        action="store_true",
        help="Подтвердить удаление без дополнительного вопроса.",
    )
    return parser


def _handle_create(login: str, password: str | None) -> int:
    """Создаёт пользователя по данным из CLI."""

    plain_password = password or _read_password(confirm=True)
    user = create_user(login, plain_password)
    print(f"Пользователь создан: id={user.id}, login={user.login}")
    print(f"База данных: {get_database_path()}")
    return 0


def _handle_list() -> int:
    """Печатает список настроенных пользователей."""

    users = list_users()
    print(f"База данных: {get_database_path()}")
    if not users:
        print("Пользователи отсутствуют.")
        return 0

    for user in users:
        print(
            f"id={user.id} login={user.login} "
            f"created_at={user.created_at} updated_at={user.updated_at}"
        )
    return 0


def _handle_set_password(login: str, password: str | None) -> int:
    """Обновляет пароль пользователя по данным из CLI."""

    plain_password = password or _read_password(confirm=True)
    update_user_password_by_login(login, plain_password)
    print(f"Пароль пользователя '{login.strip().casefold()}' обновлён.")
    return 0


def _handle_delete(login: str, confirmed: bool) -> int:
    """Удаляет пользователя после явного подтверждения."""

    normalized_login = login.strip().casefold()
    if not confirmed:
        answer = input(
            f"Удалить пользователя '{normalized_login}'? Введите 'yes' для подтверждения: "
        ).strip()
        if answer.lower() != "yes":
            print("Удаление отменено.")
            return 1

    delete_user_by_login(normalized_login)
    print(f"Пользователь '{normalized_login}' удалён.")
    return 0


def main() -> None:
    """Запускает CLI управления пользователями."""

    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "create":
            raise SystemExit(_handle_create(args.login, args.password))
        if args.command == "list":
            raise SystemExit(_handle_list())
        if args.command == "set-password":
            raise SystemExit(_handle_set_password(args.login, args.password))
        if args.command == "delete":
            raise SystemExit(_handle_delete(args.login, args.yes))
        raise SystemExit(2)
    except (
        DuplicateUserError,
        StorageUnavailableError,
        UserNotFoundError,
        UserValidationError,
    ) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
