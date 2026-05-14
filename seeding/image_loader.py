"""Image/PDF loading helpers shared by UI and session restore code."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np

from seeding.config import PDF_RENDER_SCALE_DEFAULT
from seeding.models import AppState
from seeding.utils import clip_bbox_to_image


class ImageLoadError(RuntimeError):
    """Raised when a source file cannot be rendered as project pages."""


@dataclass(frozen=True)
class LoadedPage:
    """One loaded project page and the source file it came from."""

    image: np.ndarray
    source_path: str


ProgressCallback = Callable[[int, int], None]
CancelCallback = Callable[[], bool]


def load_source_file_images(source_path: str) -> list[np.ndarray]:
    """Load all renderable images/pages from one image or PDF source."""
    path = Path(source_path)
    if path.suffix.lower() == ".pdf":
        return _load_pdf_images(source_path)

    image = cv2.imread(source_path)
    if image is None:
        raise ImageLoadError(f"Cannot open image: {source_path}")
    return [image]


def load_pages_from_path(
    file_path: str,
    *,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> list[LoadedPage]:
    """Load a file into project pages, optionally reporting per-page PDF progress."""
    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        return [LoadedPage(image=image, source_path=file_path) for image in load_source_file_images(file_path)]

    pages: list[LoadedPage] = []
    doc = fitz.open(file_path)
    try:
        total = int(doc.page_count)
        for page_index in range(total):
            if should_cancel is not None and should_cancel():
                break
            pages.append(
                LoadedPage(
                    image=_render_pdf_page(doc, page_index),
                    source_path=file_path,
                )
            )
            if on_progress is not None:
                on_progress(page_index + 1, total)
    finally:
        doc.close()
    return pages


def restore_session_source_images(state: AppState) -> str | None:
    """Load session source files into state and rebuild object/part crops."""
    source_path = state.image_storage.file_path
    source_files = list(state.image_storage.source_files or [])
    if not source_files and source_path:
        source_files = [source_path]

    missing = [path for path in source_files if not path or not Path(path).exists()]
    if missing:
        return "\n".join(dict.fromkeys(missing))

    loaded: list[np.ndarray] = []
    cache: dict[str, list[np.ndarray]] = {}
    cursor_by_source: dict[str, int] = {}
    for page_source in source_files:
        if page_source not in cache:
            cache[page_source] = load_source_file_images(page_source)
            cursor_by_source[page_source] = 0
        page_images = cache[page_source]
        page_cursor = cursor_by_source[page_source]
        if page_cursor >= len(page_images):
            return page_source
        loaded.append(page_images[page_cursor])
        cursor_by_source[page_source] = page_cursor + 1

    if not loaded:
        return source_path or None

    state.image_storage.images = loaded
    state.image_storage.source_files = source_files[: len(loaded)]
    if state.image_storage.class_object_image is None:
        state.image_storage.class_object_image = []
    while len(state.image_storage.class_object_image) < len(loaded):
        state.image_storage.class_object_image.append([])
    rebuild_session_crops(state)
    return None


def rebuild_session_crops(state: AppState) -> None:
    """Restore object and part crop images from loaded session pages."""
    pages = state.image_storage.class_object_image or []
    for page_index, objects in enumerate(pages):
        if page_index >= len(state.image_storage.images):
            continue
        page_image = state.image_storage.images[page_index]
        height, width = page_image.shape[:2]
        for obj in objects:
            if obj.bbox is None:
                continue
            clipped = clip_bbox_to_image(obj.bbox, width, height)
            if clipped is None:
                obj.image = []
                continue
            x1, y1, x2, y2 = clipped
            obj.bbox = clipped
            crop = page_image[y1:y2, x1:x2].copy()
            if obj.rotation_k:
                crop = np.rot90(crop, k=obj.rotation_k)
            obj.image = [crop]
            for part in obj.image_all_class or []:
                if part.bbox is None:
                    continue
                crop_height, crop_width = crop.shape[:2]
                part_box = clip_bbox_to_image(part.bbox, crop_width, crop_height)
                if part_box is None:
                    continue
                part.bbox = part_box
                px1, py1, px2, py2 = part_box
                part.image = crop[py1:py2, px1:px2].copy()


def _load_pdf_images(source_path: str) -> list[np.ndarray]:
    doc = fitz.open(source_path)
    try:
        return [_render_pdf_page(doc, page_index) for page_index in range(int(doc.page_count))]
    finally:
        doc.close()


def _render_pdf_page(doc, page_index: int) -> np.ndarray:
    page = doc.load_page(page_index)
    pix = page.get_pixmap(
        matrix=fitz.Matrix(
            PDF_RENDER_SCALE_DEFAULT,
            PDF_RENDER_SCALE_DEFAULT,
        )
    )
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height,
        pix.width,
        pix.n,
    )
    if pix.n == 4:
        image = image[:, :, :3].copy()
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
