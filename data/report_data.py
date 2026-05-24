"""
report_data.py - Summarize a local ImageFolder gameplay dataset.

This gives the data owner quick evidence for the final report: class balance,
image sizes, and the expected train/val/test split counts.

Run from the project root:
    python data/report_data.py --dataset-dir dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_image_paths(dataset_dir: Path) -> list[Path]:
    return sorted(
        path for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_counts(labels: list[str], val_split: float, seed: int) -> dict[str, dict[str, int]]:
    indices = list(range(len(labels)))
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 3:
        return {}

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=val_split * 2,
        stratify=labels,
        random_state=seed,
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=temp_labels,
        random_state=seed,
    )
    output = {"train": {}, "val": {}, "test": {}}
    for name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_counter = Counter(labels[i] for i in idxs)
        output[name] = dict(sorted(split_counter.items()))
    return output


def build_report(dataset_dir: Path, val_split: float, seed: int) -> dict:
    paths = iter_image_paths(dataset_dir)
    if not paths:
        raise RuntimeError(f"No image files found under {dataset_dir}")

    class_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    bad_files: list[str] = []
    labels: list[str] = []
    per_class_sizes: dict[str, Counter[str]] = defaultdict(Counter)

    for path in paths:
        class_name = path.relative_to(dataset_dir).parts[0]
        try:
            with Image.open(path) as img:
                size = f"{img.width}x{img.height}"
        except OSError:
            bad_files.append(str(path))
            continue
        class_counts[class_name] += 1
        size_counts[size] += 1
        per_class_sizes[class_name][size] += 1
        labels.append(class_name)

    min_count = min(class_counts.values()) if class_counts else 0
    max_count = max(class_counts.values()) if class_counts else 0
    imbalance_ratio = round(max_count / min_count, 3) if min_count else None

    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "total_images": sum(class_counts.values()),
        "num_classes": len(class_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "smallest_class_count": min_count,
        "largest_class_count": max_count,
        "imbalance_ratio_largest_to_smallest": imbalance_ratio,
        "recommended_balanced_cap_per_class": min_count,
        "image_size_counts": dict(size_counts.most_common()),
        "per_class_image_sizes": {
            cls: dict(counter.most_common())
            for cls, counter in sorted(per_class_sizes.items())
        },
        "planned_split_counts": split_counts(labels, val_split, seed),
        "bad_files": bad_files,
    }


def print_report(report: dict) -> None:
    print(f"Dataset: {report['dataset_dir']}")
    print(f"Images : {report['total_images']}")
    print(f"Classes: {report['num_classes']}")
    print("\nClass counts:")
    for cls, count in report["class_counts"].items():
        print(f"  {cls:<24} {count:>5}")
    print(
        "\nBalance: "
        f"smallest={report['smallest_class_count']}, "
        f"largest={report['largest_class_count']}, "
        f"ratio={report['imbalance_ratio_largest_to_smallest']}"
    )
    print("\nMost common image sizes:")
    for size, count in list(report["image_size_counts"].items())[:8]:
        print(f"  {size:<12} {count:>5}")
    if report["planned_split_counts"]:
        print("\nPlanned 80/10/10 split counts:")
        for split_name, counts in report["planned_split_counts"].items():
            total = sum(counts.values())
            print(f"  {split_name:<5} total={total}")
            for cls, count in counts.items():
                print(f"    {cls:<22} {count:>5}")
    else:
        print("\nPlanned split counts unavailable: need at least 2 classes and 3 images per class.")
    if report["bad_files"]:
        print(f"\nBad files: {len(report['bad_files'])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize local gameplay dataset.")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output", default="dataset_report.json")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_report(Path(args.dataset_dir), args.val_split, args.seed)
    print_report(report)
    output = Path(args.output)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved JSON report to {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
