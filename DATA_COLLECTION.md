# Data Collection

The final **17-class** training set is built by **merging two sources**. All scripts live under `data/` and are run **from the project root**.

## Overview

| Source | Classes | Per class | Script | Output |
|--------|---------|-----------|--------|--------|
| Classic public dataset (HuggingFace) | 10 | 1000 | `export_hf_dataset.py` | `dataset_hf/` |
| Self-collected YouTube frames | 10 | 1000 | `collect_youtube_to_1000.py` | `dataset_youtube_hq/` |
| Merged & de-duplicated → **final training set** | **17** | 1000 | `build_combined_dataset.py` | `dataset_combined/` |

> The overlapping Fortnite / Minecraft / Genshin Impact exist in both sources (2,000 images each on disk); at training time `data.py` balances every class to 1,000 and applies a stratified 80/10/10 split (`seed=42`, reproducible).

---

## Source 1: Classic HuggingFace baseline

[`Bingsu/Gameplay_Images`](https://huggingface.co/datasets/Bingsu/Gameplay_Images) — 10 games × 1,000 images at 640×360. Download and export to a local ImageFolder in one line:

```bash
python data/export_hf_dataset.py --output-dir dataset_hf
python data/report_data.py --dataset-dir dataset_hf --output dataset_hf_report.json
```

Classes: Among Us · Apex Legends · Fortnite · Forza Horizon · Free Fire · Genshin Impact · God of War · Minecraft · Roblox · Terraria. A reproducible public baseline.

---

## Source 2: Self-collected YouTube frames (core)

The 10 newer games not in the public dataset (ARC Raiders, CS2, League of Legends, Marvel Rivals, Rocket League, Valorant, Subnautica 2, ...) are sampled from real YouTube gameplay videos.

### How it's collected (methodology)

Core principle: **use real video frames, ensure source diversity, drop duplicates and low quality.** Five steps:

**① Sourcing — search queries, not random picks**
`collect_youtube_to_1000.py` defines 3–4 search queries per game, deliberately favoring "no commentary / recent version / real gameplay" videos:

```
"Minecraft survival gameplay no commentary 2026"
"Valorant ranked gameplay 2026 no commentary"
"ARC Raiders extraction gameplay no commentary"
...
```

- **no commentary**: avoids webcam overlays, commentator faces, and stream decorations — the frame is just the game
- **multiple queries × top ~8 videos each**: the same game comes from different streamers / maps / characters / scenes, so the model learns the *game* rather than one video
- URLs are de-duplicated to avoid sampling the same video twice

**② Download — bypassing YouTube restrictions** (engine `collect_data.py`, via `yt-dlp`)
- `player_client=android_vr,android`: YouTube only serves SABR streams to the web client lately; the Android VR / testsuite clients return a normal HTTPS MP4, no ffmpeg muxing needed
- prefers **≤480p mp4**: enough for 224px model input, far faster and more reliable than multi-GB 720p+
- needs **`deno`** (a JS runtime) to solve YouTube's n-challenge, otherwise `n challenge solving failed`; the script auto-detects a winget/scoop deno and injects it into PATH

```bash
winget install denoland.deno   # one-time JS runtime install
```

**③ Frame sampling — controlled rate & window**
- samples at **2 fps** by default, only over the **0:30–12:00** window (skip intro / outro / loading)
- each frame is resized to **640×360**
- per-video / per-query frame caps prevent any single source from dominating

**④ Deduplication — perceptual hash (pHash)**
- each frame's pHash is compared to kept frames; Hamming distance **< 8** → near-duplicate, discarded
- games have many static / repeated shots, so this markedly improves the share of useful data

**⑤ Quota & provenance**
- stops at **1,000 images** per class; extras are moved to `tobeclean/rejected/`
- every image's origin is logged to a manifest (`tobeclean/metadata/youtube_1000_expansion_manifest.csv`): **source URL, video timestamp, pHash, size** — fully traceable and reproducible

### One-command self-collection

```bash
# Top up all 10 YouTube classes to 1,000 images each
python data/collect_youtube_to_1000.py

# Only some classes
python data/collect_youtube_to_1000.py --classes Valorant ARCRaiders
```

### Collect a single game (engine collect_data.py)

```bash
# Specific video + time window
python data/collect_data.py --game "Minecraft" --url "https://youtu.be/XXX" --start-time 1:30 --end-time 5:00

# Auto-download top N search results
python data/collect_data.py --game "Valorant" --search "Valorant gameplay 2024 no commentary" --max-videos 3

# Extract from a locally downloaded HD video (most reliable, no YouTube limits)
python data/collect_data.py --game "RocketLeague" --video-dir raw_videos/RocketLeague --fps 1 --max-frames 500
```

> By default **only real video frames are used**. Videos blocked by YouTube 403 / SABR are skipped; only with an explicit `--allow-storyboard` does it fall back to low-res storyboard thumbnails (debug only, never for final data). Output is `dataset/<Game>/frame_XXXXX.png` + `collection_manifest.csv` (also per-image provenance).

---

## Cleaning & reporting

```bash
# QC: corrupt / low-res / too-dark-or-bright / low-information / near-duplicate (dry run first)
python data/clean_data.py --dataset-dir dataset_youtube_hq
python data/clean_data.py --dataset-dir dataset_youtube_hq --apply   # move rejects out after review

# Stats: class counts, size distribution, 80/10/10 split plan
python data/report_data.py --dataset-dir dataset_youtube_hq
```

---

## Build the 17-class training set

```bash
python data/build_combined_dataset.py
```

Hard-links `dataset_hf/` (classic 10) + `dataset_youtube_hq/` (self-collected 10) into `dataset_combined/`, de-duplicating Fortnite / Minecraft / Genshin Impact → 17 classes, 20,000 images, and writes `combined_manifest.csv` recording each image's source.

---

## Full reproduction (from scratch to a trained model)

```bash
# 1. Classic 10 classes
python data/export_hf_dataset.py --output-dir dataset_hf

# 2. Self-collected 10 classes (first: winget install denoland.deno)
python data/collect_youtube_to_1000.py
python data/clean_data.py --dataset-dir dataset_youtube_hq --apply
python data/report_data.py --dataset-dir dataset_youtube_hq

# 3. Merge into 17 classes
python data/build_combined_dataset.py
```
```powershell
# 4. Train / evaluate (PowerShell env var; uses the local merged data)
$env:DATA_SOURCE='local'; python model/train.py
$env:DATA_SOURCE='local'; python model/eval.py
```

> The repo already ships a trained `best_model.pth` (17-class, test 99.53%); to just see it work, skip steps 1–4 and run `python demo/app.py`.
