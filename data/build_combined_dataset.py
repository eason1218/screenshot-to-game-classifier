"""
build_combined_dataset.py — Merge the classic HF 10-class export (dataset_hf/)
and the self-collected YouTube 10-class set (dataset_youtube_hq/) into a single
ImageFolder dataset (dataset_combined/) for a unified ~17-class classifier.

Overlapping games are de-duplicated into one folder:
    Fortnite, Minecraft  -> identical names in both sources, merged as-is
    GenshinImpact        -> renamed to "Genshin Impact" (classic naming)

Each image is hard-linked by default (instant, no extra disk, original sets
untouched); pass --copy to physically copy instead. A manifest records every
image's provenance (source dataset + original path) for the final report.

Run from the project root:
    python data/build_combined_dataset.py
    python data/build_combined_dataset.py --copy --overwrite
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# source tag -> source folder (relative to project root)
SOURCES = {
    "hf": "dataset_hf",
    "yt": "dataset_youtube_hq",
}

# Raw ImageFolder name -> canonical class name. Only overlaps whose names
# differ between the two sources need an entry; everything else is identity.
CANONICAL = {
    "GenshinImpact": "Genshin Impact",
}


def canon(name: str) -> str:
    return CANONICAL.get(name, name)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Merge dataset_hf + dataset_youtube_hq into dataset_combined."
    )
    ap.add_argument("--output-dir", default="dataset_combined", help="Output folder.")
    ap.add_argument("--copy", action="store_true",
                    help="Copy files instead of hard-linking (uses extra disk).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Delete the output folder before rebuilding.")
    return ap.parse_args()


def link_or_copy(src: Path, dst: Path, do_copy: bool) -> None:
    if dst.exists():
        return
    if do_copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)            # NTFS hard link (same volume, instant)
    except OSError:
        shutil.copy2(src, dst)        # fall back to copy across volumes


def main() -> None:
    args = parse_args()
    # Resolve paths relative to the project root regardless of CWD.
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)

    out = Path(args.output_dir)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    per_source: dict[str, dict[str, int]] = {tag: {} for tag in SOURCES}
    rows: list[dict[str, str]] = []

    for tag, src_dir in SOURCES.items():
        sp = Path(src_dir)
        if not sp.exists():
            raise FileNotFoundError(f"Source dataset not found: {src_dir}")
        class_dirs = sorted(p for p in sp.iterdir() if p.is_dir())
        for class_dir in class_dirs:
            raw = class_dir.name
            cls = canon(raw)
            dst_dir = out / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            imgs = [p for p in sorted(class_dir.iterdir())
                    if p.suffix.lower() in IMG_EXT]
            for p in imgs:
                dst = dst_dir / f"{tag}__{p.name}"   # tag prefix avoids collisions
                link_or_copy(p, dst, args.copy)
                counts[cls] = counts.get(cls, 0) + 1
                per_source[tag][cls] = per_source[tag].get(cls, 0) + 1
                rows.append({
                    "combined_path": str(dst),
                    "canonical_class": cls,
                    "source_dataset": src_dir,
                    "source_tag": tag,
                    "original_class": raw,
                    "original_path": str(p),
                })

    manifest = out / "combined_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "combined_path", "canonical_class", "source_dataset",
            "source_tag", "original_class", "original_path",
        ])
        writer.writeheader()
        writer.writerows(rows)

    classes = sorted(counts)            # ImageFolder uses this alphabetical order
    print(f"\nCombined dataset written to: {out.resolve()}")
    print(f"{len(classes)} classes, {sum(counts.values())} images total\n")
    print(f"  {'idx':>3}  {'class':<18}{'total':>7}{'  (hf':>7}{' + yt)':>7}")
    for i, c in enumerate(classes):
        hf = per_source['hf'].get(c, 0)
        yt = per_source['yt'].get(c, 0)
        flag = "  <- merged" if hf and yt else ""
        print(f"  {i:>3}  {c:<18}{counts[c]:>7}{hf:>7}{yt:>7}{flag}")

    print("\n--- paste into config.py ---")
    print("CLASS_NAMES = [")
    for c in classes:
        print(f'    "{c}",')
    print("]")
    print(f"\nManifest: {manifest}")


if __name__ == "__main__":
    main()
