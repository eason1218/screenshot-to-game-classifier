"""
collect_data.py — Download game videos from YouTube and extract screenshot frames.

Pipeline:
    1. Download video(s) via yt-dlp (up to 720p mp4)
    2. Sample frames at a given FPS
    3. Deduplicate with perceptual hashing (skip near-identical frames)
    4. Save as PNG into  dataset/<GameName>/frame_XXXXX.png
       → Compatible with torchvision.datasets.ImageFolder

Install extra deps (one-time):
    pip install yt-dlp opencv-python imagehash

Usage examples:
    # Single URL
    python collect_data.py --game "Minecraft" --url "https://youtu.be/XXX"

    # Multiple URLs from a text file (one URL per line)
    python collect_data.py --game "Fortnite" --url-file fortnite_urls.txt

    # YouTube search (downloads top-N results automatically)
    python collect_data.py --game "Genshin Impact" \\
        --search "Genshin Impact gameplay 2024 no commentary" --max-videos 3

    # Aggressive sampling: 2 fps, max 1000 frames per video, tight dedup
    python collect_data.py --game "Minecraft" --url "https://youtu.be/XXX" \\
        --fps 2 --max-frames 1000 --hash-dist 12

    # Only sample the 1:30 → 5:00 portion of each video
    python collect_data.py --game "Minecraft" --url "https://youtu.be/XXX" \\
        --start-time 1:30 --end-time 5:00
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import os
import shutil
import subprocess
import sys
import tempfile
from email import policy
from email.parser import BytesParser

import cv2
import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

# Recent YouTube web clients often expose only SABR formats to yt-dlp in this
# environment. Android VR/testsuite clients expose normal HTTPS video streams,
# including video-only MP4 files. That is enough for frame extraction and avoids
# requiring ffmpeg/audio merging.
YTDLP_COMMON: list[str] = [
    "--extractor-args", "youtube:player_client=android_vr,android",
]

_DENO_CHECKED = False


def ensure_deno_on_path() -> None:
    """
    Make sure a `deno` JS runtime is discoverable by the yt-dlp subprocess.

    yt-dlp locates JS runtimes via PATH. winget installs deno into a
    versioned package directory that is NOT on PATH by default, so we detect
    it and prepend its folder to this process's PATH (inherited by the
    yt-dlp subprocess). Idempotent — runs its search only once, and is a
    no-op if `deno` is already resolvable.
    """
    global _DENO_CHECKED
    if _DENO_CHECKED:
        return
    _DENO_CHECKED = True

    if shutil.which("deno"):
        return  # already on PATH — nothing to do

    candidates: list[str] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates += glob.glob(os.path.join(
            local, "Microsoft", "WinGet", "Packages",
            "DenoLand.Deno_*", "deno.exe"))
    candidates += [
        os.path.expanduser(r"~\scoop\apps\deno\current\deno.exe"),
        r"C:\Program Files\deno\deno.exe",
    ]

    for exe in candidates:
        if os.path.isfile(exe):
            deno_dir = os.path.dirname(exe)
            os.environ["PATH"] = deno_dir + os.pathsep + os.environ.get("PATH", "")
            print(f"  [deno] added to PATH: {deno_dir}")
            return

    print("  [deno] WARNING: no deno runtime found — YouTube downloads may "
          "fail with 'n challenge solving failed'.\n"
          "         Install with:  winget install denoland.deno")


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_timestamp(value: str) -> float:
    """
    Parse a timestamp into seconds. Accepts SS, MM:SS, or HH:MM:SS
    (fractional seconds allowed, e.g. "90", "1:30", "0:01:30.5").

    Used as an argparse `type=`, so a bad value raises ArgumentTypeError.
    """
    parts = value.strip().split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid timestamp '{value}' (use SS, MM:SS, or HH:MM:SS)"
        )
    if len(nums) == 1:
        seconds = nums[0]
    elif len(nums) == 2:
        seconds = nums[0] * 60 + nums[1]
    elif len(nums) == 3:
        seconds = nums[0] * 3600 + nums[1] * 60 + nums[2]
    else:
        raise argparse.ArgumentTypeError(
            f"invalid timestamp '{value}' (too many ':' parts)"
        )
    if seconds < 0:
        raise argparse.ArgumentTypeError(f"timestamp '{value}' is negative")
    return seconds


# ── Download ──────────────────────────────────────────────────────────────────

def search_youtube(query: str, max_videos: int) -> list[str]:
    """
    Use yt-dlp's built-in search to find YouTube video URLs.

    Returns canonical watch URLs built from video IDs. Note: `--get-url`
    would emit *direct stream* URLs and, for DASH videos, both a video
    and an audio URL per result — doubling/poisoning the list. Printing
    the id and rebuilding the watch URL avoids that and dedupes cleanly.

    Args:
        query:      Search string, e.g. "Minecraft survival gameplay".
        max_videos: How many results to return.

    Returns:
        List of unique https://www.youtube.com/watch?v=ID URLs.
    """
    ensure_deno_on_path()
    cmd = [
        sys.executable, "-m", "yt_dlp",
        *YTDLP_COMMON,
        f"ytsearch{max_videos}:{query}",
        "--flat-playlist", "--print", "%(id)s",
        "--skip-download", "--quiet", "--no-warnings",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    urls: list[str] = []
    seen: set[str] = set()
    for vid in (line.strip() for line in result.stdout.splitlines()):
        if vid and vid not in seen:
            seen.add(vid)
            urls.append(f"https://www.youtube.com/watch?v={vid}")
    return urls


def download_video(
    url: str,
    out_dir: str,
    max_duration: int = 600,
    max_filesize: str = "500M",
) -> str | None:
    """
    Download a YouTube video (≤720p, single-file, no ffmpeg needed).

    Args:
        url:          YouTube video URL.
        out_dir:      Temporary directory to write the file into.
        max_duration: Skip videos longer than this many seconds (default 600 = 10 min).

    Returns:
        Local file path on success, None on failure or if video too long.
    """
    ensure_deno_on_path()
    template = os.path.join(out_dir, "video.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        *YTDLP_COMMON,
        "--format",
        # 480p/360p true video frames are enough for 224px model input and
        # much more reliable than multi-GB 720p+ files for quick collection.
        "best[height<=480][ext=mp4]/best[height<=480][ext=webm]/best[height<=720][ext=mp4]/best[height<=720]/best",
        "--match-filter", f"duration <= {max_duration}",   # skip long videos
        "--max-filesize", max_filesize,                    # hard cap on file size
        "--output", template,
        "--no-playlist",
        "--quiet", "--no-progress",
        url,
    ]
    print(f"  Downloading (max {max_duration//60} min, <= {max_filesize}): {url}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"  Warning: yt-dlp exited with code {proc.returncode} "
              f"(video may exceed duration/size limit)")

    for fname in os.listdir(out_dir):
        if fname.startswith("video."):
            return os.path.join(out_dir, fname)
    return None


def download_storyboard(url: str, out_dir: str, max_duration: int = 600) -> str | None:
    """
    Download YouTube storyboard thumbnails as an MHTML file.

    This is a practical fallback when YouTube blocks direct MP4 downloads with
    HTTP 403 / SABR streaming restrictions. The storyboard still contains
    regularly sampled gameplay frames suitable for classification data.
    """
    template = os.path.join(out_dir, "storyboard.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "sb0",
        "--match-filter", f"duration <= {max_duration}",
        "--output", template,
        "--no-playlist",
        "--quiet", "--no-progress",
        url,
    ]
    print(f"  Downloading storyboard fallback (max {max_duration//60} min): {url}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"  Warning: storyboard download exited with code {proc.returncode}")
    for fname in os.listdir(out_dir):
        if fname.startswith("storyboard.") and fname.endswith(".mhtml"):
            return os.path.join(out_dir, fname)
    return None


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    out_dir: str,
    fps: float = 1.0,
    max_frames: int = 500,
    resize: tuple[int, int] = (640, 360),
    hash_dist: int = 8,
    start_time: float | None = None,
    end_time: float | None = None,
    source_url: str = "",
    game_name: str = "",
    manifest_writer: csv.DictWriter | None = None,
) -> int:
    """
    Sample frames from a video, deduplicate with perceptual hashing, save as PNG.

    Args:
        video_path: Path to the downloaded video file.
        out_dir:    Folder where PNGs are saved (created if absent).
        fps:        Target sampling rate in frames-per-second of video time.
        max_frames: Hard cap on frames written from this single video.
        resize:     Output image size (width, height).
        hash_dist:  Minimum pHash difference to accept a frame (higher = stricter).
        start_time: Skip everything before this many seconds (None = from start).
        end_time:   Stop sampling at this many seconds (None = until the end).

    Returns:
        Number of frames actually saved.
    """
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: cannot open video {video_path}")
        return 0

    video_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_raw   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval    = max(1, int(round(video_fps / fps)))
    duration_s  = total_raw / video_fps

    # Clamp the requested [start, end] window to the real video duration
    start_s = max(0.0, start_time or 0.0)
    end_s   = duration_s if end_time is None else min(end_time, duration_s)
    if start_s >= end_s:
        print(f"  Error: empty time range "
              f"(start {start_s:.1f}s ≥ end {end_s:.1f}s, "
              f"video is {duration_s:.0f}s)")
        cap.release()
        return 0

    start_frame = int(start_s * video_fps)
    end_frame   = int(end_s * video_fps)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Count existing frames so new ones get unique names
    existing = sum(1 for f in os.listdir(out_dir) if f.endswith(".png"))

    window = "" if (start_time is None and end_time is None) else \
        f"  |  window {start_s:.0f}s–{end_s:.0f}s"
    print(f"  Video: {duration_s:.0f}s @ {video_fps:.1f} fps  |  "
          f"sampling every {interval} frames  |  "
          f"target ≤{max_frames} new frames{window}")

    seen_hashes: list[imagehash.ImageHash] = []
    saved = 0
    raw_idx = start_frame

    pbar = tqdm(
        total=min(max_frames, max(0, (end_frame - start_frame) // interval)),
        desc="  Frames", unit="fr",
    )

    while cap.isOpened() and saved < max_frames and raw_idx < end_frame:
        ret, bgr = cap.read()
        if not ret:
            break

        if (raw_idx - start_frame) % interval == 0:
            # Convert BGR → RGB, resize
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize(resize, Image.LANCZOS)

            # Perceptual hash deduplication
            h = imagehash.phash(img)
            if any(abs(h - prev) < hash_dist for prev in seen_hashes):
                raw_idx += 1
                continue
            seen_hashes.append(h)

            saved_path = os.path.join(out_dir, f"frame_{existing + saved:05d}.png")
            img.save(saved_path)
            if manifest_writer is not None:
                manifest_writer.writerow({
                    "path": saved_path,
                    "game": game_name,
                    "source_url": source_url,
                    "video_path": video_path,
                    "video_time_sec": f"{raw_idx / video_fps:.3f}",
                    "width": resize[0],
                    "height": resize[1],
                    "phash": str(h),
                })
            saved += 1
            pbar.update(1)

        raw_idx += 1

    pbar.close()
    cap.release()
    return saved


def _iter_mhtml_images(mhtml_path: str):
    with open(mhtml_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    for part in msg.walk():
        if part.get_content_type().startswith("image/"):
            data = part.get_payload(decode=True)
            if data:
                try:
                    yield Image.open(io.BytesIO(data)).convert("RGB")
                    continue
                except OSError:
                    pass
                arr = np.frombuffer(data, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    yield Image.fromarray(rgb)


def extract_storyboard_frames(
    mhtml_path: str,
    out_dir: str,
    fps: float = 1.0,
    max_frames: int = 500,
    resize: tuple[int, int] = (640, 360),
    hash_dist: int = 8,
    source_url: str = "",
    game_name: str = "",
    manifest_writer: csv.DictWriter | None = None,
) -> int:
    """
    Extract individual 16:9 tiles from a YouTube storyboard MHTML file.

    The highest storyboard level usually stores a 3 x 3 grid of 320 x 180
    thumbnails per image. We crop the grid, upscale to the project's frame
    size, and apply the same pHash deduplication as normal video extraction.
    """
    os.makedirs(out_dir, exist_ok=True)
    existing = sum(1 for f in os.listdir(out_dir) if f.endswith(".png"))
    seen_hashes: list[imagehash.ImageHash] = []
    saved = 0
    tile_w, tile_h = 320, 180

    images = list(_iter_mhtml_images(mhtml_path))
    print(f"  Storyboard pages: {len(images)}")
    pbar = tqdm(total=max_frames, desc="  Storyboard frames", unit="fr")

    for page_idx, sheet in enumerate(images):
        if saved >= max_frames:
            break
        cols = max(1, sheet.width // tile_w)
        rows = max(1, sheet.height // tile_h)
        for row in range(rows):
            for col in range(cols):
                if saved >= max_frames:
                    break
                left, top = col * tile_w, row * tile_h
                if left + tile_w > sheet.width or top + tile_h > sheet.height:
                    continue
                tile = sheet.crop((left, top, left + tile_w, top + tile_h))
                img = tile.resize(resize, Image.LANCZOS)
                h = imagehash.phash(img)
                if any(abs(h - prev) < hash_dist for prev in seen_hashes):
                    continue
                seen_hashes.append(h)
                saved_path = os.path.join(out_dir, f"frame_{existing + saved:05d}.png")
                img.save(saved_path)
                if manifest_writer is not None:
                    manifest_writer.writerow({
                        "path": saved_path,
                        "game": game_name,
                        "source_url": source_url,
                        "video_path": mhtml_path,
                        "video_time_sec": f"storyboard:{page_idx}:{row}:{col}",
                        "width": resize[0],
                        "height": resize[1],
                        "phash": str(h),
                    })
                saved += 1
                pbar.update(1)

    pbar.close()
    return saved


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download game videos and extract screenshot frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_argument_group("source (at least one required)")
    src.add_argument("--url",        help="Single YouTube URL")
    src.add_argument("--url-file",   help="Text file with one URL per line")
    src.add_argument("--search",     help="YouTube search query string")
    src.add_argument("--video-file", help="Local gameplay video file to extract frames from")
    src.add_argument("--video-dir",  help="Local folder of gameplay videos to extract frames from")
    src.add_argument("--max-videos", type=int, default=3,
                     help="Videos to download when using --search (default: 3)")

    p.add_argument("--game",       required=True,
                   help="Game name → used as subfolder, e.g. 'Minecraft'")
    p.add_argument("--output-dir", default="dataset",
                   help="Root output directory (default: dataset/)")
    p.add_argument("--manifest", default=None,
                   help="CSV provenance log (default: <output-dir>/collection_manifest.csv)")
    p.add_argument("--allow-storyboard", action="store_true",
                   help="Allow low-resolution YouTube storyboard fallback when MP4 download fails")
    p.add_argument("--fps",        type=float, default=1.0,
                   help="Frames per second to sample (default: 1.0)")
    p.add_argument("--max-frames", type=int, default=500,
                   help="Max frames per video (default: 500)")
    p.add_argument("--width",      type=int, default=640,
                   help="Output frame width (default: 640)")
    p.add_argument("--height",     type=int, default=360,
                   help="Output frame height (default: 360)")
    p.add_argument("--start-time", type=parse_timestamp, default=None,
                   metavar="TS",
                   help="Start sampling at this point — SS, MM:SS, or "
                        "HH:MM:SS (default: video start)")
    p.add_argument("--end-time",   type=parse_timestamp, default=None,
                   metavar="TS",
                   help="Stop sampling at this point — SS, MM:SS, or "
                        "HH:MM:SS (default: video end)")
    p.add_argument("--hash-dist",     type=int, default=8,
                   help="pHash distance threshold for dedup — "
                        "higher = keep more frames (default: 8)")
    p.add_argument("--max-duration",  type=int, default=600,
                   help="Skip videos longer than this many seconds (default: 600 = 10 min)")
    p.add_argument("--max-filesize", default="500M",
                   help="Skip videos above this yt-dlp filesize limit (default: 500M)")
    return p


def _local_video_files(video_file: str | None, video_dir: str | None) -> list[str]:
    video_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
    paths: list[str] = []
    if video_file:
        paths.append(video_file)
    if video_dir:
        for path in sorted(glob.glob(os.path.join(video_dir, "**", "*"), recursive=True)):
            if os.path.isfile(path) and os.path.splitext(path)[1].lower() in video_exts:
                paths.append(path)
    return paths


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if (args.start_time is not None and args.end_time is not None
            and args.start_time >= args.end_time):
        parser.error(
            f"--start-time ({args.start_time:.0f}s) must be earlier than "
            f"--end-time ({args.end_time:.0f}s)"
        )

    # ── Collect all URLs ──────────────────────────────────────────────────────
    urls: list[str] = []
    local_videos = _local_video_files(args.video_file, args.video_dir)
    if args.url:
        urls.append(args.url)
    if args.url_file:
        with open(args.url_file) as f:
            urls += [ln.strip() for ln in f if ln.strip()]
    if args.search:
        print(f'Searching YouTube: "{args.search}" (top {args.max_videos}) …')
        found = search_youtube(args.search, args.max_videos)
        print(f"Found {len(found)} video(s).")
        if not found:
            print("  Search returned no usable YouTube IDs. Try --url with a "
                  "specific video link or a more specific search query.")
        urls += found
    if not urls and not local_videos:
        build_parser().error(
            "Provide at least one of: --url, --url-file, --search, --video-file, --video-dir"
        )

    # ── Process each video ────────────────────────────────────────────────────
    game_dir = os.path.join(args.output_dir, args.game)
    os.makedirs(game_dir, exist_ok=True)
    resize = (args.width, args.height)
    session_total = 0
    manifest_path = args.manifest or os.path.join(args.output_dir, "collection_manifest.csv")
    manifest_exists = os.path.exists(manifest_path)
    manifest_fields = [
        "path", "game", "source_url", "video_path", "video_time_sec",
        "width", "height", "phash",
    ]

    with open(manifest_path, "a", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=manifest_fields)
        if not manifest_exists:
            writer.writeheader()

        for i, video_path in enumerate(local_videos, 1):
            print(f"\n[local {i}/{len(local_videos)}] {video_path}")
            n = extract_frames(
                video_path, game_dir,
                fps=args.fps,
                max_frames=args.max_frames,
                resize=resize,
                hash_dist=args.hash_dist,
                start_time=args.start_time,
                end_time=args.end_time,
                source_url="local_file",
                game_name=args.game,
                manifest_writer=writer,
            )
            print(f"  → {n} frame(s) saved.")
            session_total += n

        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] {url}")
            with tempfile.TemporaryDirectory() as tmp:
                video_path = download_video(
                    url,
                    tmp,
                    max_duration=args.max_duration,
                    max_filesize=args.max_filesize,
                )
                if video_path is not None:
                    n = extract_frames(
                        video_path, game_dir,
                        fps=args.fps,
                        max_frames=args.max_frames,
                        resize=resize,
                        hash_dist=args.hash_dist,
                        start_time=args.start_time,
                        end_time=args.end_time,
                        source_url=url,
                        game_name=args.game,
                        manifest_writer=writer,
                    )
                else:
                    if not args.allow_storyboard:
                        print("  Skipped — video download failed. "
                              "Storyboard fallback is disabled for final-quality data.")
                        continue
                    print("  Video download failed; trying storyboard fallback.")
                    storyboard_path = download_storyboard(
                        url, tmp, max_duration=args.max_duration
                    )
                    if storyboard_path is None:
                        print("  Skipped — both video and storyboard failed.")
                        continue
                    n = extract_storyboard_frames(
                        storyboard_path, game_dir,
                        fps=args.fps,
                        max_frames=args.max_frames,
                        resize=resize,
                        hash_dist=args.hash_dist,
                        source_url=url,
                        game_name=args.game,
                        manifest_writer=writer,
                    )
                print(f"  → {n} frame(s) saved.")
                session_total += n

    # ── Summary ───────────────────────────────────────────────────────────────
    total_in_dir = sum(1 for f in os.listdir(game_dir) if f.endswith(".png"))
    print(f"\n{'─'*50}")
    print(f"Game      : {args.game}")
    print(f"Output    : {game_dir}")
    print(f"Manifest  : {manifest_path}")
    print(f"This run  : +{session_total} frames")
    print(f"Total now : {total_in_dir} frames")
    print(f"\nTo use this data with ImageFolder:")
    print(f"  from torchvision.datasets import ImageFolder")
    print(f"  ds = ImageFolder(root='{args.output_dir}', transform=...)")


if __name__ == "__main__":
    main()
