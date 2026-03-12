"""Core services for image processing, detection, classification and reports."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from seeding.models import AllClassImage, ObjectImage, OriginalImage
from seeding.report import create_pdf_report
from seeding.utils import (
    clip_bbox_to_image,
    rotate_bbox,
    rotate_image_and_boxes,
    rotate_polygon_points,
    simple_nms,
)


class ImageService:
    @staticmethod
    def _iter_page_objects(
        image_storage: OriginalImage,
        page_index: int,
    ) -> list[ObjectImage]:
        if (
            not image_storage.class_object_image
            or page_index >= len(image_storage.class_object_image)
        ):
            return []
        return image_storage.class_object_image[page_index]

    def refresh_page_crops(
        self,
        image_storage: OriginalImage,
        page_index: int,
        *,
        rotate_k: int,
        clear_classification: bool,
    ) -> None:
        page_objects = self._iter_page_objects(image_storage, page_index)
        if not page_objects:
            return

        base_img = image_storage.images[page_index]
        if base_img is None:
            return
        height, width = base_img.shape[:2]

        for obj in page_objects:
            if not obj.bbox:
                continue

            clipped = clip_bbox_to_image(obj.bbox, width, height)
            if clipped is None:
                obj.bbox = None
                obj.image = []
                if clear_classification:
                    obj.image_all_class = None
                continue

            x1, y1, x2, y2 = clipped
            obj.bbox = clipped
            crop = base_img[y1:y2, x1:x2].copy()

            rotation_k = 0
            if crop.shape[1] > crop.shape[0]:
                crop = np.rot90(crop, k=rotate_k)
                rotation_k = rotate_k

            obj.rotation_k = rotation_k
            obj.image = [crop]
            if clear_classification:
                obj.image_all_class = None

    def rotate_page(
        self,
        image_storage: OriginalImage,
        page_index: int,
        *,
        angle: float,
        rotate_k: int,
    ) -> np.ndarray:
        page_image = image_storage.images[page_index]
        if page_image is None:
            raise ValueError("Изображение страницы пустое")

        page_objects = self._iter_page_objects(image_storage, page_index)
        object_map: list[int] = []
        boxes_to_rotate: list[tuple[int, int, int, int]] = []
        for obj_idx, obj in enumerate(page_objects):
            if obj.bbox:
                boxes_to_rotate.append(obj.bbox)
                object_map.append(obj_idx)

        rotated_image, rotated_boxes = rotate_image_and_boxes(
            page_image,
            boxes_to_rotate,
            angle,
        )
        image_storage.images[page_index] = rotated_image

        for mapped_idx, obj_idx in enumerate(object_map):
            page_objects[obj_idx].bbox = rotated_boxes[mapped_idx]

        self.refresh_page_crops(
            image_storage,
            page_index,
            rotate_k=rotate_k,
            clear_classification=True,
        )
        return rotated_image

    def rotate_crop(
        self,
        image_storage: OriginalImage,
        page_index: int,
        crop_index: int,
        *,
        angle: float,
        rotate_k: int,
    ) -> np.ndarray:
        page_objects = self._iter_page_objects(image_storage, page_index)
        obj = page_objects[crop_index]
        if not obj.image or obj.image[0] is None:
            raise ValueError("Изображение кропа пустое")

        crop = obj.image[0]
        crop_height, crop_width = crop.shape[:2]
        class_map: list[int] = []
        class_boxes: list[tuple[int, int, int, int]] = []
        if obj.image_all_class:
            for class_idx, cls_obj in enumerate(obj.image_all_class):
                if cls_obj.bbox:
                    class_boxes.append(cls_obj.bbox)
                    class_map.append(class_idx)

        rotated_crop, rotated_class_boxes = rotate_image_and_boxes(
            crop,
            class_boxes,
            angle,
        )
        obj.image[0] = rotated_crop
        obj.rotation_k = (obj.rotation_k + rotate_k) % 4

        for mapped_idx, class_idx in enumerate(class_map):
            obj.image_all_class[class_idx].bbox = rotated_class_boxes[mapped_idx]
        if obj.image_all_class:
            for cls_obj in obj.image_all_class:
                if cls_obj.mask_polygon is None:
                    continue
                cls_obj.mask_polygon = rotate_polygon_points(
                    cls_obj.mask_polygon,
                    crop_width,
                    crop_height,
                    angle,
                )
        return rotated_crop

    @staticmethod
    def clip_local_bbox(
        crop_image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        height, width = crop_image.shape[:2]
        return clip_bbox_to_image(bbox, width, height)

    @staticmethod
    def crop_region(
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]

    @staticmethod
    def clip_many(
        image: np.ndarray,
        boxes: Iterable[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int] | None]:
        height, width = image.shape[:2]
        return [clip_bbox_to_image(box, width, height) for box in boxes]

    def sync_crops_and_parts(self, image_storage: OriginalImage) -> None:
        if not image_storage.images or not image_storage.class_object_image:
            return

        for page_index, objects in enumerate(image_storage.class_object_image):
            if page_index >= len(image_storage.images):
                continue
            base_img = image_storage.images[page_index]
            height, width = base_img.shape[:2]
            for obj in objects:
                if obj.bbox:
                    clipped_obj = clip_bbox_to_image(obj.bbox, width, height)
                    if clipped_obj is None:
                        obj.image = []
                        obj.bbox = None
                        continue
                    x1, y1, x2, y2 = clipped_obj
                    obj.bbox = clipped_obj
                    crop = base_img[y1:y2, x1:x2].copy()
                    if getattr(obj, "rotation_k", 0):
                        crop = np.rot90(crop, k=obj.rotation_k)
                    obj.image = [crop]

                if not obj.image_all_class:
                    continue

                rotation_k = getattr(obj, "rotation_k", 0) % 4
                crop_height, crop_width = (
                    obj.image[0].shape[:2] if obj.image else (0, 0)
                )
                for cls_obj in obj.image_all_class:
                    if not cls_obj.bbox:
                        continue

                    lx1, ly1, lx2, ly2 = cls_obj.bbox
                    if rotation_k and crop_height and crop_width:
                        ux1, uy1, ux2, uy2 = rotate_bbox(
                            lx1,
                            ly1,
                            lx2,
                            ly2,
                            crop_width,
                            crop_height,
                            (-rotation_k) % 4,
                        )
                    else:
                        ux1, uy1, ux2, uy2 = lx1, ly1, lx2, ly2

                    if obj.bbox:
                        gx1 = obj.bbox[0] + ux1
                        gy1 = obj.bbox[1] + uy1
                        gx2 = obj.bbox[0] + ux2
                        gy2 = obj.bbox[1] + uy2
                    else:
                        gx1, gy1, gx2, gy2 = ux1, uy1, ux2, uy2

                    clipped_part = clip_bbox_to_image(
                        (gx1, gy1, gx2, gy2),
                        width,
                        height,
                    )
                    if clipped_part is None:
                        cls_obj.image = np.empty((0, 0, 3), dtype=base_img.dtype)
                        continue
                    gx1, gy1, gx2, gy2 = clipped_part
                    part = base_img[gy1:gy2, gx1:gx2].copy()
                    if rotation_k:
                        part = np.rot90(part, k=rotation_k)
                    cls_obj.image = part


class DetectionService:
    @staticmethod
    def build_objects(
        image: np.ndarray,
        results,
        *,
        detection_class_name: str,
        iou_threshold: float,
        rotate_k: int,
    ) -> list[ObjectImage]:
        if image is None or results is None or not results:
            return []

        parsed: list[dict] = []
        boxes: list[list[int]] = []
        scores: list[float] = []
        height, width = image.shape[:2]

        for box in results[0].boxes:
            class_id = int(box.cls)
            class_name = str(results[0].names[class_id]).lower()
            if class_name != detection_class_name.lower():
                continue

            score = float(box.conf)
            x_center, y_center, box_width, box_height = box.xywh[0].cpu().numpy()
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            clipped = clip_bbox_to_image((x1, y1, x2, y2), width, height)
            if clipped is None:
                continue

            cx1, cy1, cx2, cy2 = clipped
            parsed.append(
                {
                    "class_name": class_name,
                    "score": score,
                    "bbox": (cx1, cy1, cx2, cy2),
                }
            )
            boxes.append([cx1, cy1, cx2, cy2])
            scores.append(score)

        if not boxes:
            return []

        kept_indices = simple_nms(boxes, scores, iou_threshold=iou_threshold)
        objects: list[ObjectImage] = []
        for idx in kept_indices:
            item = parsed[idx]
            x1, y1, x2, y2 = item["bbox"]
            crop = image[y1:y2, x1:x2].copy()

            rotation_k = 0
            if crop.shape[1] > crop.shape[0]:
                crop = np.rot90(crop, k=rotate_k)
                rotation_k = rotate_k

            objects.append(
                ObjectImage(
                    class_name=item["class_name"],
                    confidence=float(item["score"]),
                    image=[crop],
                    bbox=(x1, y1, x2, y2),
                    rotation_k=rotation_k,
                )
            )
        return objects


class ClassificationService:
    @staticmethod
    def _clip_mask_polygon(
        mask_polygon: np.ndarray | None,
        width: int,
        height: int,
    ) -> np.ndarray | None:
        if mask_polygon is None or width <= 0 or height <= 0:
            return None
        polygon = np.asarray(mask_polygon, dtype=np.float32)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
            return None
        clipped = polygon.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(width - 1))
        clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(height - 1))
        return np.ascontiguousarray(clipped)

    @staticmethod
    def build_parts(crop_image: np.ndarray, results) -> list[AllClassImage]:
        if crop_image is None or results is None or not results:
            return []

        crop_height, crop_width = crop_image.shape[:2]
        parts: list[AllClassImage] = []
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf)
                class_id = int(box.cls)
                class_name = result.names[class_id]
                coords = box.xyxy[0].cpu().numpy().astype(int)
                local_bbox = clip_bbox_to_image(
                    (
                        int(coords[0]),
                        int(coords[1]),
                        int(coords[2]),
                        int(coords[3]),
                    ),
                    crop_width,
                    crop_height,
                )
                if local_bbox is None:
                    continue

                lx1, ly1, lx2, ly2 = local_bbox
                part_image = crop_image[ly1:ly2, lx1:lx2].copy()
                mask_polygon = ClassificationService._clip_mask_polygon(
                    getattr(box, "mask_polygon", None),
                    crop_width,
                    crop_height,
                )
                parts.append(
                    AllClassImage(
                        class_name=class_name,
                        confidence=confidence,
                        image=part_image,
                        bbox=local_bbox,
                        mask_polygon=mask_polygon,
                    )
                )
        return parts


class ReportService:
    @staticmethod
    def generate_report(data: OriginalImage, output_path: str) -> str:
        create_pdf_report(data, output_path)
        return output_path


__all__ = [
    "ImageService",
    "ReportService",
    "DetectionService",
    "ClassificationService",
]
