"""
clean_data.py - Quality control for locally collected gameplay frames.

The script scans an ImageFolder-style dataset, flags broken/low-quality images,
and removes near-duplicates with perceptual hashing. By default it is a dry run.
Use --apply to move rejected files into a quarantine folder.

Run from the project root:
    python data/clean_data.py --dataset-dir dataset
    python data/clean_data.py --dataset-dir dataset --apply
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageStat, UnidentifiedImageError
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageRecord:
    path: Path
    class_name: str
    width: int = 0
    height: int = 0
    brightness: float = 0.0
    variance: float = 0.0
    hash_value: imagehash.ImageHash | None = None


def iter_image_paths(dataset_dir: Path) -> list[Path]:
    return sorted(
        path for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_image(path: Path, dataset_dir: Path) -> tuple[ImageRecord | None, str | None]:
    class_name = path.relative_to(dataset_dir).parts[0]
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            gray = rgb.convert("L")
            stat = ImageStat.Stat(gray)
            arr = np.asarray(gray, dtype=np.float32)
            return ImageRecord(
                path=path,
                class_name=class_name,
                width=rgb.width,
                height=rgb.height,
                brightness=float(stat.mean[0]),
                variance=float(arr.var()),
                hash_value=imagehash.phash(rgb),
            ), None
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return None, f"invalid_image:{exc}"


def quality_reason(
    record: ImageRecord,
    min_width: int,
    min_height: int,
    min_variance: float,
    min_brightness: float,
    max_brightness: float,
) -> str | None:
    if record.width < min_width or record.height < min_height:
        return f"too_small:{record.width}x{record.height}"
    if record.variance < min_variance:
        return f"low_detail:variance={record.variance:.2f}"
    if record.brightness < min_brightness:
        return f"too_dark:brightness={record.brightness:.1f}"
    if record.brightness > max_brightness:
        return f"too_bright:brightness={record.brightness:.1f}"
    return None


def unique_quarantine_path(path: Path, dataset_dir: Path, quarantine_dir: Path) -> Path:
    rel = path.relative_to(dataset_dir)
    target = quarantine_dir / rel
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def move_to_quarantine(path: Path, dataset_dir: Path, quarantine_dir: Path) -> Path:
    target = unique_quarantine_path(path, dataset_dir, quarantine_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    return target


def write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "class",
        "status",
        "reason",
        "width",
        "height",
        "brightness",
        "variance",
        "hash",
        "quarantine_path",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean locally collected gameplay frames.",
    )
    parser.add_argument("--dataset-dir", default="dataset",
                        help="ImageFolder dataset root (default: dataset)")
    parser.add_argument("--quarantine-dir", default="dataset_rejected",
                        help="Where rejected files are moved when --apply is set")
    parser.add_argument("--manifest", default="dataset_cleaning_report.csv",
                        help="CSV report path")
    parser.add_argument("--min-width", type=int, default=320)
    parser.add_argument("--min-height", type=int, default=180)
    parser.add_argument("--min-variance", type=float, default=80.0,
                        help="Reject nearly blank/static frames below this grayscale variance")
    parser.add_argument("--min-brightness", type=float, default=8.0)
    parser.add_argument("--max-brightness", type=float, default=247.0)
    parser.add_argument("--hash-dist", type=int, default=6,
                        help="Reject as duplicate if pHash distance is below this value")
    parser.add_argument("--cross-class-duplicates", action="store_true",
                        help="Also reject near-duplicates across different classes")
    parser.add_argument("--apply", action="store_true",
                        help="Move rejected files to quarantine. Omit for dry run.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = Path(args.dataset_dir)
    quarantine_dir = Path(args.quarantine_dir)
    manifest_path = Path(args.manifest)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    paths = iter_image_paths(dataset_dir)
    if not paths:
        raise RuntimeError(f"No image files found under {dataset_dir}")

    accepted_hashes: dict[str, list[tuple[Path, imagehash.ImageHash]]] = {}
    all_hashes: list[tuple[Path, str, imagehash.ImageHash]] = []
    rows: list[dict[str, str]] = []
    rejected = 0

    for path in tqdm(paths, desc="Inspecting images", unit="img"):
        record, invalid_reason = inspect_image(path, dataset_dir)
        quarantine_path = ""
        status = "keep"
        reason = ""

        if record is None:
            class_name = path.relative_to(dataset_dir).parts[0]
            status = "reject"
            reason = invalid_reason or "invalid_image"
            width = height = brightness = variance = ""
            hash_text = ""
        else:
            class_name = record.class_name
            width = str(record.width)
            height = str(record.height)
            brightness = f"{record.brightness:.2f}"
            variance = f"{record.variance:.2f}"
            hash_text = str(record.hash_value)
            reason = quality_reason(
                record,
                args.min_width,
                args.min_height,
                args.min_variance,
                args.min_brightness,
                args.max_brightness,
            ) or ""

            if not reason:
                duplicate_source = None
                class_hashes = accepted_hashes.setdefault(class_name, [])
                for kept_path, kept_hash in class_hashes:
                    if abs(record.hash_value - kept_hash) < args.hash_dist:
                        duplicate_source = kept_path
                        break
                if duplicate_source is None and args.cross_class_duplicates:
                    for kept_path, kept_class, kept_hash in all_hashes:
                        if abs(record.hash_value - kept_hash) < args.hash_dist:
                            duplicate_source = kept_path
                            break
                if duplicate_source is not None:
                    reason = f"near_duplicate:{duplicate_source}"

            if reason:
                status = "reject"
            else:
                class_hashes.append((path, record.hash_value))
                all_hashes.append((path, class_name, record.hash_value))

        if status == "reject":
            rejected += 1
            if args.apply:
                quarantine_path = str(move_to_quarantine(path, dataset_dir, quarantine_dir))

        rows.append({
            "path": str(path),
            "class": class_name,
            "status": status,
            "reason": reason,
            "width": width,
            "height": height,
            "brightness": brightness,
            "variance": variance,
            "hash": hash_text,
            "quarantine_path": quarantine_path,
        })

    write_manifest(rows, manifest_path)
    kept = len(rows) - rejected
    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"\nMode      : {mode}")
    print(f"Scanned   : {len(rows)} images")
    print(f"Kept      : {kept}")
    print(f"Rejected  : {rejected}")
    print(f"Manifest  : {manifest_path}")
    if not args.apply:
        print("No files were moved. Re-run with --apply to quarantine rejected images.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
