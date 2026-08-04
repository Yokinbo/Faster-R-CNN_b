"""Adapter from a standard YOLO detection dataset to this repository's loader.

The Faster R-CNN model and its training losses are untouched.  This module only
converts normalized YOLO ``class cx cy width height`` labels to the pixel-space
``xmin,ymin,xmax,ymax,class`` records already consumed by ``FRCNNDataset`` and
``EvalCallback``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class YoloDatasetSummary:
    split: str
    images: int
    objects: int
    negative_images: int

    def __str__(self) -> str:
        return (
            f"{self.split}: images={self.images}, objects={self.objects}, "
            f"negative_images={self.negative_images}"
        )


def _read_classes(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing class file: {path}")
    classes = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    classes = [name for name in classes if name]
    if not classes:
        raise ValueError(f"Class file is empty: {path}")
    if len(set(classes)) != len(classes):
        raise ValueError(f"Duplicate class names found in: {path}")
    return classes


def validate_yolo_class_files(dataset_root: Path) -> Path:
    """Ensure train/val use exactly the same ordered class list."""
    dataset_root = Path(dataset_root).resolve()
    train_path = dataset_root / "train" / "classes.txt"
    val_path = dataset_root / "val" / "classes.txt"
    train_classes = _read_classes(train_path)
    val_classes = _read_classes(val_path)
    if train_classes != val_classes:
        raise ValueError(
            "train/classes.txt and val/classes.txt must contain the same "
            "classes in the same order."
        )
    return train_path


def _files_by_stem(directory: Path, extensions=None) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")

    result: Dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file():
            continue
        if extensions is not None and path.suffix.lower() not in extensions:
            continue
        key = path.stem.casefold()
        if key in result:
            raise ValueError(
                f"Duplicate file stem '{path.stem}' in {directory}: "
                f"{result[key].name}, {path.name}"
            )
        result[key] = path.resolve()
    return result


def _yolo_box_to_pixels(
    values: List[str], image_width: int, image_height: int,
    num_classes: int, label_path: Path, line_number: int,
) -> Tuple[int, int, int, int, int]:
    if len(values) != 5:
        raise ValueError(
            f"{label_path}:{line_number}: expected 5 YOLO fields, got {len(values)}"
        )
    try:
        class_value, cx, cy, box_width, box_height = map(float, values)
    except ValueError as exc:
        raise ValueError(
            f"{label_path}:{line_number}: label contains a non-numeric value"
        ) from exc

    class_id = int(class_value)
    if class_value != class_id:
        raise ValueError(f"{label_path}:{line_number}: class id must be an integer")
    if not 0 <= class_id < num_classes:
        raise ValueError(
            f"{label_path}:{line_number}: class id {class_id} is outside "
            f"[0, {num_classes - 1}]"
        )
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        raise ValueError(
            f"{label_path}:{line_number}: normalized center must be within [0, 1]"
        )
    if not (0.0 < box_width <= 1.0 and 0.0 < box_height <= 1.0):
        raise ValueError(
            f"{label_path}:{line_number}: normalized width/height must be in (0, 1]"
        )

    left_f = (cx - box_width / 2.0) * image_width
    top_f = (cy - box_height / 2.0) * image_height
    right_f = (cx + box_width / 2.0) * image_width
    bottom_f = (cy + box_height / 2.0) * image_height

    left = max(0, min(image_width - 1, math.floor(left_f)))
    top = max(0, min(image_height - 1, math.floor(top_f)))
    right = max(left + 1, min(image_width, math.ceil(right_f)))
    bottom = max(top + 1, min(image_height, math.ceil(bottom_f)))
    return left, top, right, bottom, class_id


def build_yolo_annotation_lines(
    dataset_root: Path, split: str, num_classes: int,
) -> Tuple[List[str], YoloDatasetSummary]:
    """Read one YOLO split and create in-memory legacy annotation records."""
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split!r}; expected 'train' or 'val'")

    split_root = Path(dataset_root).resolve() / split
    images = _files_by_stem(split_root / "images", SUPPORTED_IMAGE_EXTENSIONS)
    labels = _files_by_stem(split_root / "labels", {".txt"})
    if not images:
        raise ValueError(f"No supported images found in: {split_root / 'images'}")

    missing_labels = sorted(images.keys() - labels.keys())
    orphan_labels = sorted(labels.keys() - images.keys())
    if missing_labels or orphan_labels:
        details = []
        if missing_labels:
            details.append(f"images without labels: {len(missing_labels)}")
        if orphan_labels:
            details.append(f"labels without images: {len(orphan_labels)}")
        raise ValueError(f"{split} image/label mismatch ({', '.join(details)})")

    annotation_lines: List[str] = []
    object_count = 0
    negative_count = 0

    for key, image_path in images.items():
        if any(char.isspace() for char in str(image_path)):
            raise ValueError(
                "This repository's legacy annotation parser cannot handle "
                f"whitespace in image paths: {image_path}"
            )
        label_path = labels[key]
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Invalid image dimensions: {image_path}")

        boxes = []
        label_text = label_path.read_text(encoding="utf-8-sig")
        for line_number, raw_line in enumerate(label_text.splitlines(), start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            box = _yolo_box_to_pixels(
                raw_line.split(), image_width, image_height,
                num_classes, label_path, line_number,
            )
            boxes.append(",".join(map(str, box)))

        if not boxes:
            negative_count += 1
        object_count += len(boxes)
        annotation_lines.append(" ".join([str(image_path), *boxes]) + "\n")

    summary = YoloDatasetSummary(
        split=split,
        images=len(annotation_lines),
        objects=object_count,
        negative_images=negative_count,
    )
    return annotation_lines, summary
