"""Простейшая временная авторизация приложения."""

from __future__ import annotations

from dataclasses import dataclass

TEST_USER_EMAIL = "user@user.com"
TEST_USER_PASSWORD = "user"


@dataclass(frozen=True, slots=True)
class AuthUser:
    """Описывает пользователя, допущенного к работе с приложением."""

    display_name: str
    email: str


TEST_USER = AuthUser(display_name="User", email=TEST_USER_EMAIL)


def authenticate(email: str, password: str) -> AuthUser | None:
    """Проверяет тестовые учётные данные и возвращает пользователя при успехе."""

    normalized_email = email.strip().casefold()
    if normalized_email != TEST_USER.email.casefold():
        return None
    if password != TEST_USER_PASSWORD:
        return None
    return TEST_USER
