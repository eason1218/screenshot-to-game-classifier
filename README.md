# 🎮 Screenshot to Game Classifier

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--50-EE4C2C?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Demo-Gradio-F97316?logo=gradio&logoColor=white)
![Accuracy](https://img.shields.io/badge/test%20accuracy-99.53%25-success)
![License](https://img.shields.io/badge/license-MIT-green)

Identify which of **17 popular games** a screenshot belongs to — on direct screenshots **and** on photos of a screen (the system detects the screen, corrects its perspective, then classifies).

Machine Learning 2 Final Project — by **Yizhuo Li**, **Elaine Wang**, **Cecilia Hua**, and **Cassie Li**.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Features

- 🖼️ **Screenshot classification** — upload a frame, get Top-3 predictions with confidence.
- 📷 **Photo-of-screen mode** — photograph a phone/monitor; the screen is auto-detected and perspective-corrected before classification (the webcam auto-starts in this mode).
- 🧹 **Reproducible data pipeline** — YouTube collection, perceptual-hash de-duplication, quality cleaning, reporting, and per-image provenance manifests.
- 🎯 **99.53% test accuracy** across 17 game classes.

## Architecture

Three sub-packages around a shared `config.py` contract, with two data flows — one for training, one for inference:

```
                        ┌──────────────────────────────────────────┐
                        │                config.py                   │
                        │   classes · paths · hyperparameters        │
                        └──────────────────────────────────────────┘
                            │               │                  │
            ┌───────────────▼───┐  ┌────────▼────────┐  ┌──────▼─────────────┐
            │      data/         │  │     model/       │  │      demo/          │
            │ collect · clean    │  │  ResNet-50       │  │  Gradio UI          │
            │ merge · DataLoader │  │  train · eval    │  │  screen detection   │
            └─────────┬──────────┘  └────────┬─────────┘  └─────────┬──────────┘
                      │                       │                      │
  Training flow:  dataset_combined ──► train.py ──► best_model.pth   │
                                                          │          │
  Inference flow:  photo / screenshot ─► screen_crop ─► letterbox ─► best_model.pth ─► Top-3
```

| Module | Responsibility | Key tech |
|--------|----------------|----------|
| `config.py` | Shared contract: class names, paths, hyperparameters | — |
| `data/` | Collection, cleaning, merging, DataLoaders | yt-dlp · OpenCV · imagehash · HF `datasets` |
| `model/` | ResNet-50 fine-tuning, training, evaluation | PyTorch · torchvision · scikit-learn |
| `demo/` | Gradio frontend, screen detection & perspective correction | Gradio · Mobile SAM · OpenCV |

**Tech stack:** Python 3.13 · PyTorch (ResNet-50) · Gradio · Mobile SAM · OpenCV · yt-dlp · HuggingFace `datasets`.

## Methodology

### Problem

Recognizing a game from one frame is easy on clean screenshots, but the real-world version is harder: people photograph a screen with a phone, at an angle, with glare and blur. We wanted a classifier that is accurate on screenshots **and** robust to photos-of-screens. That goal shaped every decision below — the data, the geometry, and the augmentations.

### 1 · Data — diversity by design, not just volume

We combine two complementary sources into a **17-class** set:

- **Public baseline** — HuggingFace `Bingsu/Gameplay_Images` (10 classic games), a reproducible reference.
- **Self-collected** — 10 newer games no public dataset covers (Valorant, CS2, ARC Raiders, Marvel Rivals, Rocket League, Subnautica 2, …), sampled from real YouTube gameplay.

They are merged and de-duplicated (Fortnite / Minecraft / Genshin Impact overlap) and balanced to 1,000 images per class. The methodology matters more than the count — the aim is a dataset that teaches the *game*, not a streamer or a map (full details in **[DATA_COLLECTION.md](DATA_COLLECTION.md)**):

- **Diversity over volume** — each game is gathered from several "no-commentary, recent-version" search queries across many videos.
- **Real frames only** — extracted from actual video (≤480p suffices for a 224px model), never thumbnails; blocked videos are skipped, not padded with low-quality fallbacks.
- **Quality control** — 2 fps sampling, perceptual-hash de-duplication, then a cleaning pass for corrupt / too-dark / low-information frames.
- **Provenance** — every image logs its source URL, timestamp, and hash, so the dataset is fully traceable.

### 2 · Model — geometry that matches the task

- **Backbone:** ResNet-50 pretrained on ImageNet (IMAGENET1K_V2), fully fine-tuned with the head replaced for 17 classes.
- **Letterboxing is the central design choice.** Game frames are 16:9. Squashing them to a square distorts the image; cropping discards edge HUD / minimap cues that are often the most discriminative signal. Instead we **aspect-preserve and pad to 224×224**, applying the *identical* transform in training, evaluation, and the demo. In an earlier 10-class experiment, unifying the geometry this way lifted test accuracy from 99.40% to 99.90% — geometry consistency, not a bigger model, was the win.
- **Augmentation simulates the photo-of-screen case.** On top of standard flips and color jitter, we add perspective warps, rotation, blur, and random erasing — mimicking the artifacts of photographing a screen at an angle.

### 3 · Robust inference — from a phone photo to a clean frame

The demo accepts photos of a screen, not just screenshots. To classify those reliably it first recovers a clean frame:

1. **Detect the screen** — Mobile SAM segmentation (primary), with classical-CV fallbacks (brightness percentile → Otsu → Canny edges).
2. **Correct perspective** — the detected quadrilateral is rectified with a homography.
3. **Classify** — the rectified frame goes through the same letterbox transform used in training.

## Results

| | |
|--|--|
| Architecture | ResNet-50, fine-tuned (17-class head) |
| Data | 17 classes × 1,000, stratified 80/10/10 |
| Best validation | **99.65%** (epoch 4, early-stopped) |
| **Test accuracy** | **99.53%** (1,700 held-out images) |

Per-class accuracy is **98–100%**. Accuracy saturated in just 4 epochs, so training was stopped early — evidence the bottleneck was data quality and geometry, not model capacity. The few misclassifications are between visually similar games (photorealistic shooters, or sandbox titles like Minecraft ↔ Roblox). See `classification_report.txt`, `confusion_matrix.png`, and `training_curves.png`.

## Project Structure

```
.
├── config.py                      # shared contract: classes, paths, hyperparameters
├── data/                          # DATA TEAM  → data/README.md
│   ├── data.py                    #   HF / local loading, stratified split, balancing, augmentation
│   ├── collect_data.py            #   YouTube download + frame sampling + pHash dedup (engine)
│   ├── collect_youtube_to_1000.py #   top up each YouTube class to 1,000 images
│   ├── export_hf_dataset.py       #   export the HF baseline to a local ImageFolder
│   ├── build_combined_dataset.py  #   merge dataset_hf + dataset_youtube_hq → 17 classes
│   ├── clean_data.py              #   filter corrupt / low-quality / duplicate frames
│   └── report_data.py             #   class distribution, size distribution, split plan
├── model/                         # MODEL TEAM  → model/README.md
│   ├── model.py                   #   ResNet-50 (FC layer replaced → 17 classes)
│   ├── train.py                   #   training loop, saves the best checkpoint
│   └── eval.py                    #   test evaluation, confusion matrix, training curves
├── demo/                          # DEMO TEAM  → demo/README.md
│   ├── app.py                     #   Gradio UI (screenshot / photo-of-screen)
│   └── screen_crop.py             #   screen detection + perspective correction
├── DATA_COLLECTION.md             # full data-collection methodology
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE                        # MIT
└── best_model.pth                 # shipped 17-class model (test 99.53%)
```

> **Run convention:** always run scripts **from the project root** (e.g. `python model/train.py`). Scripts add the root to `sys.path` to resolve `import config`; `best_model.pth` / `examples/` / `mobile_sam.pt` / datasets are all paths relative to the root.

## Getting Started

### Prerequisites

- Python 3.13
- An NVIDIA GPU is recommended for training (CPU works for the demo, just slower).

### Installation

```bash
pip install -r requirements.txt

# For photo-of-screen mode (Mobile SAM screen segmentation):
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

> **GPU note:** on RTX 50-series (Blackwell, sm_120), install the **cu128** torch build (nightly). cu121/cu124 builds lack kernels for that architecture — `cuda.is_available()` returns True but kernels error at runtime.

The repo ships a trained `best_model.pth`, so the demo runs without any training or data download.

## Usage

### Run the demo

```bash
python demo/app.py
```
Opens at **http://localhost:7860** and prints a public `*.gradio.live` link. Toggle between **Screenshot** and **Photo of Screen** modes (the latter auto-starts the webcam).

### Train & evaluate

Set `DATA_SOURCE=local` to use the 17-class merged set (otherwise it falls back to the online HF 10-class baseline):

```powershell
$env:DATA_SOURCE='local'; python model/train.py   # → best_model.pth + training_history.json
$env:DATA_SOURCE='local'; python model/eval.py    # → report + confusion matrix + curves
```

### Rebuild the dataset

```bash
python data/export_hf_dataset.py --output-dir dataset_hf   # classic 10 classes
python data/collect_youtube_to_1000.py                     # self-collected 10 classes
python data/build_combined_dataset.py                      # merge → 17 classes
```
Full data-collection methodology and options are in **[DATA_COLLECTION.md](DATA_COLLECTION.md)**; per-stage details are in the folder READMEs.

## Contributing

Contributions are welcome — see **[CONTRIBUTING.md](CONTRIBUTING.md)** for development setup, project conventions (e.g. keeping the letterbox geometry in sync across training/eval/demo), and the pull-request workflow.

## Authors

| Name | | Name |
|------|--|------|
| Yizhuo Li | | Elaine Wang |
| Cecilia Hua | | Cassie Li |

## Acknowledgments

- [Bingsu/Gameplay_Images](https://huggingface.co/datasets/Bingsu/Gameplay_Images) — public baseline dataset.
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) — lightweight Segment Anything used for screen detection.
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube downloading for data collection.
- Built with [PyTorch](https://pytorch.org/) and [Gradio](https://www.gradio.app/).

## License

Released under the [MIT License](LICENSE). Data comes from public sources (HuggingFace + YouTube gameplay) and is used for study and research only; all game imagery belongs to its respective publishers.
