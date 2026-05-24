"""
Collect YouTube gameplay frames until each target class has 1,000 images.

Run from the project root:
    python data/collect_youtube_to_1000.py

This script is intentionally a thin orchestrator around data/collect_data.py so
the same frame extraction, resizing, pHash de-duplication, and manifest format
are used for every class.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

from collect_data import download_video, extract_frames, search_youtube


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TARGET_CLASSES: dict[str, list[str]] = {
    "Fortnite": [
        "Fortnite gameplay no commentary chapter 6 2026",
        "Fortnite battle royale gameplay no commentary 2026",
        "Fortnite ranked gameplay 2026 no commentary",
    ],
    "GenshinImpact": [
        "Genshin Impact gameplay no commentary 2026",
        "Genshin Impact exploration gameplay 2026 no commentary",
        "Genshin Impact combat gameplay no commentary 2026",
    ],
    "LeagueOfLegends": [
        "League of Legends gameplay no commentary 2026",
        "League of Legends ranked gameplay full game 2026",
        "League of Legends ARAM gameplay no commentary 2026",
    ],
    "Minecraft": [
        "Minecraft survival gameplay no commentary 2026",
        "Minecraft gameplay 2026 no commentary",
        "Minecraft building survival gameplay no commentary",
        "Minecraft hardcore gameplay no commentary 2026",
    ],
    "RocketLeague": [
        "Rocket League gameplay no commentary 2026",
        "Rocket League ranked gameplay 2026",
        "Rocket League competitive gameplay no commentary",
    ],
    "Valorant": [
        "Valorant gameplay no commentary 2026",
        "Valorant ranked gameplay 2026 no commentary",
        "Valorant full match gameplay no commentary",
    ],
    "CounterStrike2": [
        "Counter-Strike 2 gameplay no commentary 2026",
        "CS2 competitive gameplay no commentary 2026",
        "Counter Strike 2 premier gameplay no commentary",
    ],
    "MarvelRivals": [
        "Marvel Rivals gameplay no commentary 2026",
        "Marvel Rivals ranked gameplay no commentary",
        "Marvel Rivals full match gameplay 2026",
    ],
    "ARCRaiders": [
        "ARC Raiders gameplay no commentary 2026",
        "ARC Raiders extraction gameplay no commentary",
        "ARC Raiders full gameplay 2026",
    ],
    "Subnautica2": [
        "Subnautica 2 gameplay no commentary 2026",
        "Subnautica 2 early access gameplay no commentary",
        "Subnautica 2 survival gameplay 2026",
    ],
}


def image_count(class_dir: Path) -> int:
    if not class_dir.exists():
        return 0
    return sum(
        1 for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_paths(class_dir: Path) -> list[Path]:
    if not class_dir.exists():
        return []
    return sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def trim_to_target(class_dir: Path, target: int, reject_root: Path) -> None:
    paths = image_paths(class_dir)
    extra = len(paths) - target
    if extra <= 0:
        return
    reject_dir = reject_root / class_dir.name
    reject_dir.mkdir(parents=True, exist_ok=True)
    for path in paths[-extra:]:
        dest = reject_dir / path.name
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            i = 1
            while (reject_dir / f"{stem}_{i}{suffix}").exists():
                i += 1
            dest = reject_dir / f"{stem}_{i}{suffix}"
        shutil.move(str(path), str(dest))
    print(f"Trimmed {extra} extra image(s) from {class_dir.name} into {reject_dir}.", flush=True)


def next_frame_index(class_dir: Path) -> int:
    max_idx = -1
    for path in image_paths(class_dir):
        stem = path.stem
        if stem.startswith("frame_"):
            try:
                max_idx = max(max_idx, int(stem.split("_", 1)[1]))
            except ValueError:
                pass
    return max_idx + 1


def normalize_frame_names(class_dir: Path) -> None:
    """Make future collection append after the current highest frame number."""
    paths = image_paths(class_dir)
    tmp_paths: list[Path] = []
    for i, path in enumerate(paths):
        tmp = class_dir / f"__tmp_frame_{i:05d}{path.suffix.lower()}"
        path.rename(tmp)
        tmp_paths.append(tmp)
    for i, tmp in enumerate(sorted(tmp_paths)):
        tmp.rename(class_dir / f"frame_{i:05d}.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Top up YouTube dataset classes to a target size.")
    parser.add_argument("--dataset-dir", default="dataset_youtube_hq")
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--manifest", default="tobeclean/metadata/youtube_1000_expansion_manifest.csv")
    parser.add_argument("--overshoot-dir", default="tobeclean/rejected/overshoot_1000")
    parser.add_argument("--max-videos", type=int, default=8)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames-per-query", type=int, default=420)
    parser.add_argument("--start-time", default="0:30")
    parser.add_argument("--end-time", default="12:00")
    parser.add_argument("--max-duration", type=int, default=3600)
    parser.add_argument("--max-filesize", default="900M")
    parser.add_argument("--hash-dist", type=int, default=8)
    parser.add_argument("--classes", nargs="*", default=list(TARGET_CLASSES),
                        help="Optional subset of class names to collect.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = Path(args.dataset_dir)
    manifest = Path(args.manifest)
    overshoot_dir = Path(args.overshoot_dir)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    overshoot_dir.mkdir(parents=True, exist_ok=True)

    for class_name in args.classes:
        if class_name not in TARGET_CLASSES:
            raise ValueError(f"Unknown class '{class_name}'. Known: {', '.join(TARGET_CLASSES)}")

        class_dir = dataset_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        trim_to_target(class_dir, args.target, overshoot_dir)
        normalize_frame_names(class_dir)
        current = image_count(class_dir)
        print(f"\n=== {class_name}: {current}/{args.target} images ===", flush=True)
        if current >= args.target:
            print("Already at target; skipping.", flush=True)
            continue

        manifest_exists = manifest.exists()
        manifest_fields = [
            "path", "game", "source_url", "video_path", "video_time_sec",
            "width", "height", "phash",
        ]

        seen_urls: set[str] = set()
        with manifest.open("a", newline="", encoding="utf-8") as manifest_file:
            writer = csv.DictWriter(manifest_file, fieldnames=manifest_fields)
            if not manifest_exists:
                writer.writeheader()

            for query in TARGET_CLASSES[class_name]:
                current = image_count(class_dir)
                if current >= args.target:
                    break

                print(f"\nSearch query: {query}", flush=True)
                urls = search_youtube(query, args.max_videos)
                print(f"Found {len(urls)} candidate URL(s).", flush=True)

                for url in urls:
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    current = image_count(class_dir)
                    remaining = args.target - current
                    if remaining <= 0:
                        break

                    max_frames = min(args.max_frames_per_query, remaining)
                    print(f"\nURL: {url}", flush=True)
                    print(f"Need {remaining}; collecting up to {max_frames} frame(s).", flush=True)

                    import tempfile
                    with tempfile.TemporaryDirectory() as tmp:
                        video_path = download_video(
                            url,
                            tmp,
                            max_duration=args.max_duration,
                            max_filesize=args.max_filesize,
                        )
                        if video_path is None:
                            print("Skipped: video download failed.", flush=True)
                            continue

                        before = image_count(class_dir)
                        extract_frames(
                            video_path,
                            str(class_dir),
                            fps=args.fps,
                            max_frames=max_frames,
                            resize=(640, 360),
                            hash_dist=args.hash_dist,
                            start_time=parse_timestamp_arg(args.start_time),
                            end_time=parse_timestamp_arg(args.end_time),
                            source_url=url,
                            game_name=class_name,
                            manifest_writer=writer,
                        )
                        after = image_count(class_dir)
                        print(f"{class_name}: {after}/{args.target} (+{after - before}).", flush=True)

        final = image_count(class_dir)
        trim_to_target(class_dir, args.target, overshoot_dir)
        normalize_frame_names(class_dir)
        final = image_count(class_dir)
        if final < args.target:
            print(
                f"WARNING: {class_name} ended at {final}/{args.target}. "
                "Add more queries or explicit YouTube URLs and rerun.",
                flush=True,
            )


def parse_timestamp_arg(value: str | None) -> float | None:
    if value is None:
        return None
    parts = value.strip().split(":")
    nums = [float(part) for part in parts]
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    raise ValueError(f"Bad timestamp: {value}")


if __name__ == "__main__":
    main()
