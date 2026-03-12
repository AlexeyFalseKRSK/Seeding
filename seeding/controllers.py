"""Контроллер приложения, оркестрирующий сервисы и состояние.

Пакет объединён в один модуль для уменьшения количества файлов.
"""

from __future__ import annotations

from seeding.config import DETECTION_CLASS_NAME, NMS_IOU_THRESHOLD, ROTATE_K
from seeding.models import (
    AllClassImage,
    AppState,
    ObjectImage,
    RotateSelectionResult,
    SelectionPayload,
)
from seeding.services import (
    ClassificationService,
    DetectionService,
    ImageService,
    ReportService,
)


class AppController:
    """Оркестрирует пользовательские сценарии приложения."""

    def __init__(
        self,
        image_service: ImageService | None = None,
        report_service: ReportService | None = None,
        detection_service: DetectionService | None = None,
        classification_service: ClassificationService | None = None,
    ) -> None:
        """Инициализирует контроллер и подключает сервисы доменной логики.

        Если сервис не передан извне (например, в тестах), создаётся
        стандартная реализация по умолчанию.
        """
        self.image_service = image_service or ImageService()
        self.report_service = report_service or ReportService()
        self.detection_service = detection_service or DetectionService()
        self.classification_service = (
            classification_service or ClassificationService()
        )

    def rotate_selection(
        self,
        state: AppState,
        item_data: SelectionPayload,
        *,
        angle: float,
        rotate_k: int,
    ) -> RotateSelectionResult | None:
        """Поворачивает выбранный элемент и возвращает данные для отрисовки."""
        item_type = item_data.get("type")
        if item_type in ("original", "pdf"):
            page_index = int(item_data["index"])
            rotated = self.image_service.rotate_page(
                state.image_storage,
                page_index,
                angle=angle,
                rotate_k=rotate_k,
            )
            state.active_image_index = page_index
            return RotateSelectionResult(
                target="page",
                page_index=page_index,
                image=rotated,
            )

        if item_type == "seeding":
            page_index = int(item_data["parent_index"])
            crop_index = int(item_data["index"])
            rotated = self.image_service.rotate_crop(
                state.image_storage,
                page_index,
                crop_index,
                angle=angle,
                rotate_k=rotate_k,
            )
            state.active_image_index = page_index
            return RotateSelectionResult(
                target="crop",
                page_index=page_index,
                crop_index=crop_index,
                image=rotated,
            )

        return None

    def rotate_current(
        self,
        state: AppState,
        selection: SelectionPayload,
        *,
        angle: float,
        rotate_k: int = ROTATE_K,
    ) -> RotateSelectionResult | None:
        """Синоним сценария поворота текущего выбранного элемента."""
        return self.rotate_selection(
            state,
            selection,
            angle=angle,
            rotate_k=rotate_k,
        )

    def run_detection(
        self,
        state: AppState,
        page_index: int,
        results,
        *,
        detection_class_name: str = DETECTION_CLASS_NAME,
        iou_threshold: float = NMS_IOU_THRESHOLD,
        rotate_k: int = ROTATE_K,
    ) -> list[ObjectImage]:
        """Парсит предсказания детекции и записывает объекты в ``AppState``."""
        image = state.image_storage.images[page_index]
        objects = self.detection_service.build_objects(
            image,
            results,
            detection_class_name=detection_class_name,
            iou_threshold=iou_threshold,
            rotate_k=rotate_k,
        )

        if state.image_storage.class_object_image is None:
            state.image_storage.class_object_image = [
                [] for _ in state.image_storage.images
            ]
        state.image_storage.class_object_image[page_index] = objects
        return objects

    def run_classification_for_selection(
        self,
        state: AppState,
        page_index: int,
        crop_index: int,
        results,
    ) -> list[AllClassImage]:
        """Парсит предсказания классификации для выбранного кропа."""
        page_objects = state.image_storage.class_object_image or []
        if page_index >= len(page_objects):
            return []
        objects = page_objects[page_index]
        if crop_index >= len(objects):
            return []
        selected_object = objects[crop_index]
        if not selected_object.image:
            return []

        parts = self.classification_service.build_parts(
            selected_object.image[0],
            results,
        )
        selected_object.image_all_class = parts
        return parts

    def generate_report(self, state: AppState, output_path: str) -> str:
        """Генерирует отчёт и сохраняет путь в состоянии приложения."""
        report_path = self.report_service.generate_report(
            state.image_storage,
            output_path,
        )
        state.last_report_path = report_path
        return report_path


__all__ = ["AppController"]
