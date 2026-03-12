"""Утилиты работы с путями для Windows/macOS/Linux."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Iterable


def _normalize_path_separators(path_value: str) -> str:
    """Нормализует разделители путей под текущую ОС."""
    normalized = path_value
    if os.sep != "/":
        normalized = normalized.replace("/", os.sep)
    if os.sep != "\\":
        normalized = normalized.replace("\\", os.sep)
    return normalized


def _iter_candidates(path_obj: Path, base_dirs: Iterable[Path]) -> list[Path]:
    """Строит кандидаты путей для абсолютного/относительного ввода."""
    if path_obj.is_absolute():
        return [path_obj]

    candidates: list[Path] = []
    for base_dir in base_dirs:
        candidates.append(base_dir / path_obj)
    candidates.append(path_obj)
    return candidates


def resolve_weights_path(
    path_value: str,
    *,
    base_dirs: Iterable[Path] | None = None,
) -> Path | None:
    """Разрешает путь к весам модели с учётом правил разных ОС.

    Логика:
    - абсолютный и относительный путь валидируются как файл;
    - имя без расширения ищется как ``<name>.pt`` в ``base_dirs``;
    - имя с ``.pt`` при отсутствии локального файла считается алиасом модели.
    """
    if not path_value:
        return None

    raw_value = path_value.strip()
    if not raw_value:
        return None

    normalized = _normalize_path_separators(os.path.expandvars(raw_value))
    expanded = Path(normalized).expanduser()
    if base_dirs is None:
        base_dirs = (Path.cwd(),)
    else:
        base_dirs = tuple(base_dirs)

    has_path_separators = any(sep in raw_value for sep in ("/", "\\"))
    looks_like_path = (
        expanded.is_absolute()
        or has_path_separators
        or expanded.parent != Path(".")
    )

    if looks_like_path:
        for candidate in _iter_candidates(expanded, base_dirs):
            if candidate.is_file():
                return candidate.resolve()
        if expanded.suffix == "":
            pt_path = expanded.with_suffix(".pt")
            pt_candidates = _iter_candidates(pt_path, base_dirs)
            for candidate in pt_candidates:
                if candidate.is_file():
                    return candidate.resolve()
        return None

    if expanded.suffix == "":
        for base_dir in base_dirs:
            candidate = (base_dir / expanded.name).with_suffix(".pt")
            if candidate.is_file():
                return candidate.resolve()
        return None

    if expanded.suffix.lower() == ".pt":
        for base_dir in base_dirs:
            candidate = base_dir / expanded.name
            if candidate.is_file():
                return candidate.resolve()
        return Path(expanded.name)

    return None
