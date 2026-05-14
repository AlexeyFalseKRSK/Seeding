"""Session actions shared by the main window and UI helpers."""

from __future__ import annotations

import logging

from seeding.auth import AuthUser
from seeding.image_loader import restore_session_source_images
from seeding.models import AppState
from seeding.session_service import save_session


def sync_current_user_to_state(state: AppState, current_user: AuthUser | None) -> None:
    """Copy the authenticated user id into the state before DB operations."""
    if current_user is not None:
        state.current_user_id = current_user.id


def save_current_session(state: AppState, current_user: AuthUser | None) -> int:
    """Synchronize the current user and save the session."""
    sync_current_user_to_state(state, current_user)
    return save_session(state)


def auto_save_current_session(
    state: AppState,
    current_user: AuthUser | None,
    *,
    logger: logging.Logger,
) -> bool:
    """Save the session for background UI flows and log failures."""
    try:
        save_current_session(state, current_user)
    except Exception:
        logger.exception("Auto-save session failed")
        return False
    return True


def restore_session_images(state: AppState, *, logger: logging.Logger) -> str | None:
    """Load source images for a restored session and rebuild saved crops."""
    try:
        return restore_session_source_images(state)
    except Exception:
        logger.exception("Failed to load session source images")
        return state.image_storage.file_path or None
