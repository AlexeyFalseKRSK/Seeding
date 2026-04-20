"""Build a multi-class YOLO detection dataset from a single-class source dataset.

The source dataset is expected to have a standard YOLO detect layout:

    source/
      train/
        images/
        labels/
      valid/
        images/
        labels/
      test/
        images/
        labels/

This utility is designed for the case where every image contains objects of only
one species. The user groups representative images into per-species folders,
and the script rewrites the class id of every box in the source label files.

Example:

    python tools/build_species_detection_dataset.py ^
      --source dataset/datasetV1 ^
      --species-root dataset/species_split ^
      --classes cedr pinus ^
      --output dataset/datasetV1_species

Expected `species-root` layout:

    dataset/species_split/
      cedr/
        some_source_image.jpg
      pinus/
        another_source_image.jpg

Matching is performed by a normalized image key. For Roboflow-style names such
as `image.rf.<hash>.jpg`, the `.rf.<hash>` suffix is ignored, so one reference
image is enough to classify all augmented variants of the same original image.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a YOLO detect dataset with per-species classes by rewriting "
            "class ids in label files."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the source single-class YOLO detect dataset.",
    )
    parser.add_argument(
        "--species-root",
        type=Path,
        required=True,
        help=(
            "Directory containing one subdirectory per class "
            "(for example `cedr/` and `pinus/`)."
        ),
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Ordered class names. Each name must match a subdirectory in species-root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset directory to create.",
    )
    parser.add_argument(
        "--allow-unassigned",
        action="store_true",
        help="Skip source images that are not assigned to any class.",
    )
    return parser.parse_args()


def normalized_image_key(filename: str) -> str:
    path = Path(filename)
    stem = path.stem
    if ".rf." in stem:
        stem = stem.split(".rf.", 1)[0]
    return stem.lower()


def collect_reference_keys(species_root: Path, class_names: list[str]) -> dict[str, int]:
    key_to_class_id: dict[str, int] = {}
    for class_id, class_name in enumerate(class_names):
        class_dir = species_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}"
            )
        for item in class_dir.rglob("*"):
            if not item.is_file():
                continue
            key = normalized_image_key(item.name)
            previous = key_to_class_id.get(key)
            if previous is not None and previous != class_id:
                raise ValueError(
                    f"Image key '{key}' is assigned to multiple classes: "
                    f"'{class_names[previous]}' and '{class_name}'."
                )
            key_to_class_id[key] = class_id
    if not key_to_class_id:
        raise ValueError(f"No reference files were found in {species_root}.")
    return key_to_class_id


def find_source_splits(source_root: Path) -> list[str]:
    splits: list[str] = []
    for child in sorted(source_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "images").is_dir() and (child / "labels").is_dir():
            splits.append(child.name)
    if not splits:
        raise ValueError(
            f"No YOLO splits with images/labels were found in {source_root}."
        )
    return splits


def rewrite_label_lines(lines: list[str], class_id: int) -> list[str]:
    rewritten: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO label line: {raw_line!r}")
        parts[0] = str(class_id)
        rewritten.append(" ".join(parts))
    return rewritten


def ensure_empty_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def write_data_yaml(output_root: Path, splits: list[str], class_names: list[str]) -> None:
    lines = [f"path: {output_root.resolve().as_posix()}"]
    if "train" in splits:
        lines.append("train: train/images")
    if "valid" in splits:
        lines.append("val: valid/images")
    elif "val" in splits:
        lines.append("val: val/images")
    if "test" in splits:
        lines.append("test: test/images")
    lines.append(f"nc: {len(class_names)}")
    names = ", ".join(f"'{name}'" for name in class_names)
    lines.append(f"names: [{names}]")
    (output_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(
    source_root: Path,
    species_root: Path,
    class_names: list[str],
    output_root: Path,
    *,
    allow_unassigned: bool,
) -> None:
    splits = find_source_splits(source_root)
    ensure_empty_directory(output_root)
    key_to_class_id = collect_reference_keys(species_root, class_names)

    image_counts: Counter[str] = Counter()
    box_counts: Counter[str] = Counter()
    unassigned_images: list[Path] = []
    matched_keys: set[str] = set()
    per_split_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for split in splits:
        source_images_dir = source_root / split / "images"
        source_labels_dir = source_root / split / "labels"
        target_images_dir = output_root / split / "images"
        target_labels_dir = output_root / split / "labels"
        target_images_dir.mkdir(parents=True, exist_ok=True)
        target_labels_dir.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(source_images_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            key = normalized_image_key(image_path.name)
            class_id = key_to_class_id.get(key)
            if class_id is None:
                unassigned_images.append(image_path)
                if allow_unassigned:
                    continue
                continue

            matched_keys.add(key)
            class_name = class_names[class_id]
            label_path = source_labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label file for image: {image_path}")

            label_lines = label_path.read_text(encoding="utf-8").splitlines()
            rewritten = rewrite_label_lines(label_lines, class_id)

            shutil.copy2(image_path, target_images_dir / image_path.name)
            (target_labels_dir / label_path.name).write_text(
                "\n".join(rewritten) + ("\n" if rewritten else ""),
                encoding="utf-8",
            )

            image_counts[class_name] += 1
            per_split_counts[split][class_name] += 1
            box_counts[class_name] += len(rewritten)

    if unassigned_images and not allow_unassigned:
        preview = "\n".join(str(path) for path in unassigned_images[:10])
        raise ValueError(
            "Some source images were not assigned to any class. "
            "Place representative files into the species folders or use "
            "--allow-unassigned.\n"
            f"First unmatched images:\n{preview}"
        )

    unmatched_reference_keys = sorted(set(key_to_class_id) - matched_keys)
    if unmatched_reference_keys:
        sample = ", ".join(unmatched_reference_keys[:10])
        print(
            "Warning: some reference files did not match any source image keys:",
            sample,
        )

    write_data_yaml(output_root, splits, class_names)

    print("Dataset created:", output_root)
    for split in splits:
        parts = ", ".join(
            f"{class_name}={per_split_counts[split][class_name]}"
            for class_name in class_names
            if per_split_counts[split][class_name] > 0
        )
        print(f"  {split}: {parts or 'no images copied'}")
    print("Totals:")
    for class_name in class_names:
        print(
            f"  {class_name}: images={image_counts[class_name]}, "
            f"boxes={box_counts[class_name]}"
        )
    print("data.yaml:", output_root / "data.yaml")


def main() -> None:
    args = parse_args()
    build_dataset(
        source_root=args.source,
        species_root=args.species_root,
        class_names=list(args.classes),
        output_root=args.output,
        allow_unassigned=bool(args.allow_unassigned),
    )


if __name__ == "__main__":
    main()
