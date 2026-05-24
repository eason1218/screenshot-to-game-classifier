"""
Export the Hugging Face gameplay baseline dataset to local ImageFolder format.

The exported structure is:
    dataset_hf/
      Among Us/*.png
      Apex Legends/*.png
      ...
      hf_manifest.csv

This is useful when the data group wants a concrete local handoff artifact
instead of relying on the training code to download from Hugging Face at run time.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


DATASET_NAME = "Bingsu/Gameplay_Images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/export Bingsu/Gameplay_Images into ImageFolder format."
    )
    parser.add_argument("--output-dir", default="dataset_hf", help="Export folder.")
    parser.add_argument("--split", default="train", help="Hugging Face split name.")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional class-name filter, e.g. --classes Fortnite Minecraft.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap per class for a smaller local copy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing exported images and manifest.",
    )
    parser.add_argument(
        "--image-format",
        choices=("png", "jpg"),
        default="png",
        help="Image format to save locally.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality if --image-format jpg is used.",
    )
    return parser.parse_args()


def safe_folder_name(name: str) -> str:
    """Keep class names readable while avoiding path-hostile characters."""
    cleaned = re.sub(r"[\\/:\0]", "_", name).strip()
    return cleaned or "unknown"


def selected_classes(classes: Iterable[str] | None) -> set[str] | None:
    if classes is None:
        return None
    return {name.strip().lower() for name in classes if name.strip()}


def save_image(image: Image.Image, path: Path, image_format: str, jpeg_quality: int) -> None:
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image_format == "jpg":
        image.save(path, quality=jpeg_quality, optimize=True)
    else:
        image.save(path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET_NAME} split={args.split}")
    dataset = load_dataset(DATASET_NAME, split=args.split)
    label_feature = dataset.features["label"]
    class_names = list(label_feature.names)
    class_filter = selected_classes(args.classes)

    print("Classes:")
    for idx, name in enumerate(class_names):
        marker = ""
        if class_filter is not None and name.lower() not in class_filter:
            marker = " (skipped)"
        print(f"  {idx}: {name}{marker}")

    manifest_path = output_dir / "hf_manifest.csv"
    mode = "w" if args.overwrite or not manifest_path.exists() else "a"
    write_header = mode == "w"

    counts: Counter[str] = Counter()
    next_index: defaultdict[str, int] = defaultdict(int)
    if not args.overwrite:
        for class_name in class_names:
            class_dir = output_dir / safe_folder_name(class_name)
            if class_dir.exists():
                existing = sorted(class_dir.glob(f"hf_*.{args.image_format}"))
                counts[class_name] = len(existing)
                next_index[class_name] = len(existing)

    with manifest_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "class_name",
                "label",
                "source_dataset",
                "source_split",
                "source_index",
                "width",
                "height",
            ],
        )
        if write_header:
            writer.writeheader()

        for source_index, item in enumerate(tqdm(dataset, desc="Exporting")):
            label = int(item["label"])
            class_name = class_names[label]
            if class_filter is not None and class_name.lower() not in class_filter:
                continue
            if args.max_per_class is not None and counts[class_name] >= args.max_per_class:
                continue

            class_dir = output_dir / safe_folder_name(class_name)
            class_dir.mkdir(parents=True, exist_ok=True)
            local_index = next_index[class_name]
            image_path = class_dir / f"hf_{local_index:05d}.{args.image_format}"
            next_index[class_name] += 1

            if image_path.exists() and not args.overwrite:
                continue

            image: Image.Image = item["image"]
            width, height = image.size
            save_image(image, image_path, args.image_format, args.jpeg_quality)
            counts[class_name] += 1
            writer.writerow(
                {
                    "path": str(image_path),
                    "class_name": class_name,
                    "label": label,
                    "source_dataset": DATASET_NAME,
                    "source_split": args.split,
                    "source_index": source_index,
                    "width": width,
                    "height": height,
                }
            )

    print("\nExport complete:")
    for class_name in class_names:
        if class_filter is None or class_name.lower() in class_filter:
            print(f"  {class_name}: {counts[class_name]}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
