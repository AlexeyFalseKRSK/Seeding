"""Доменные объекты авторизации и функции для работы с паролями."""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass

PBKDF2_ALGORITHM = "sha256"
PBKDF2_ITERATIONS = 120_000
PBKDF2_SALT_SIZE = 16


@dataclass(frozen=True, slots=True)
class AuthUser:
    """Аутентифицированный пользователь приложения."""

    id: int
    login: str


def hash_password(password: str) -> str:
    """Хэширует пароль с помощью PBKDF2-HMAC-SHA256."""

    if not password:
        raise ValueError("Password must not be empty.")

    salt = os.urandom(PBKDF2_SALT_SIZE)
    password_bytes = password.encode("utf-8")
    derived_key = hashlib.pbkdf2_hmac(
        PBKDF2_ALGORITHM,
        password_bytes,
        salt,
        PBKDF2_ITERATIONS,
    )
    return (
        f"pbkdf2_{PBKDF2_ALGORITHM}"
        f"${PBKDF2_ITERATIONS}"
        f"${salt.hex()}"
        f"${derived_key.hex()}"
    )


def verify_password(password: str, password_hash: str) -> bool:
    """Проверяет обычный пароль по сохранённому PBKDF2-хэшу."""

    try:
        algorithm, iterations, salt_hex, expected_hash_hex = password_hash.split("$", 3)
    except ValueError:
        return False

    if algorithm != f"pbkdf2_{PBKDF2_ALGORITHM}":
        return False

    try:
        iterations_value = int(iterations)
        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(expected_hash_hex)
    except ValueError:
        return False

    actual_hash = hashlib.pbkdf2_hmac(
        PBKDF2_ALGORITHM,
        password.encode("utf-8"),
        salt,
        iterations_value,
    )
    return hmac.compare_digest(actual_hash, expected_hash)
