# 🎮 Screenshot to Game Classifier

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--50-EE4C2C?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Demo-Gradio-F97316)
![Accuracy](https://img.shields.io/badge/test%20accuracy-99.53%25-success)
![License](https://img.shields.io/badge/license-MIT-green)

Identify which of **17 popular games** a screenshot belongs to. It works on direct screenshots **and** on a photo taken of a screen — the system detects the screen, corrects its perspective, then classifies.

Machine Learning 2 Final Project — by **Yizhuo Li**, **Elaine Wang**, **Cecilia Hua**, and **Cassie Li**.

---

## Problem

Recognizing a game from a single frame sounds easy on clean screenshots, but the real-world version is harder: people often photograph a screen with a phone, at an angle, with glare and blur. We wanted a classifier that is both accurate on screenshots **and** robust to photos-of-screens. That goal shaped every design decision below — the data we collected, the geometry we standardized on, and the augmentations we chose.

## Approach

One pipeline, three stages, tied together by a single shared image geometry so that what the model trains on is exactly what it sees at inference:

```
gameplay videos / public dataset
        │   collect · clean · de-dup · merge
        ▼
   17-class dataset  ──letterbox──►  ResNet-50 (fine-tuned)
        ▲                                   │
   screen detection + perspective correction ◄── photo of a screen
```

### 1 · Data — diversity by design, not just volume

We combine two complementary sources into a **17-class** set:

- **Public baseline** — HuggingFace `Bingsu/Gameplay_Images` (10 classic games), a reproducible reference point.
- **Self-collected** — 10 newer games no public dataset covers (Valorant, CS2, ARC Raiders, Marvel Rivals, Rocket League, Subnautica 2, …), sampled from real YouTube gameplay.

The two are merged and de-duplicated (Fortnite / Minecraft / Genshin Impact overlap) and balanced to 1,000 images per class.

The collection methodology matters more than the count — the goal is a dataset that teaches the *game*, not a streamer or a map (full details in **[DATA_COLLECTION.md](DATA_COLLECTION.md)**):

- **Diversity over volume** — each game is gathered from several "no-commentary, recent-version" search queries across many videos, so no single source dominates.
- **Real frames only** — extracted from actual video (≤480p is plenty for a 224px model), never thumbnails; videos YouTube blocks are skipped rather than padded with low-quality fallbacks.
- **Quality control** — 2 fps sampling, perceptual-hash de-duplication to drop near-identical frames, then a cleaning pass for corrupt / too-dark / low-information images.
- **Provenance** — every image logs its source URL, timestamp, and hash, so the dataset is fully traceable and reproducible.

### 2 · Model — geometry that matches the task

- **Backbone**: ResNet-50 pretrained on ImageNet (IMAGENET1K_V2), fully fine-tuned with the final layer replaced for 17 classes.
- **Letterboxing is the central design choice.** Game frames are 16:9. Squashing them into a square distorts the image; cropping to a square discards edge HUD / minimap cues that are often the most discriminative signal. Instead we **aspect-preserve and pad to 224×224**, and apply the *identical* transform in training, evaluation, and the demo. In an earlier 10-class experiment, unifying the geometry this way lifted test accuracy from 99.40% to 99.90% — geometry consistency, not a bigger model, was the win.
- **Augmentation simulates the photo-of-screen case.** On top of standard flips and color jitter, we add perspective warps, rotation, blur, and random erasing — deliberately mimicking the artifacts of photographing a screen at an angle, so the classifier stays robust to tilted, warped, and out-of-focus inputs.

### 3 · Robust inference — from a phone photo to a clean frame

The demo accepts photos of a screen, not just screenshots. To classify those reliably it first recovers a clean frame:

1. **Detect the screen** — Mobile SAM segmentation is the primary strategy, with classical-CV fallbacks (brightness percentile → Otsu → Canny edges) when SAM is unavailable.
2. **Correct perspective** — the detected quadrilateral is rectified with a homography.
3. **Classify** — the rectified frame goes through the same letterbox transform used in training.

This is what makes the model usable in practice (point a phone at a monitor), not only on pixel-perfect captures.

## Results

| | |
|--|--|
| Architecture | ResNet-50, fine-tuned (17-class head) |
| Data | 17 classes × 1,000, stratified 80/10/10 |
| Best validation | **99.65%** (epoch 4, early-stopped) |
| **Test accuracy** | **99.53%** (1,700 held-out images) |

Per-class accuracy is **98–100%**. Accuracy saturated in just 4 epochs, so training was stopped early — evidence that the bottleneck was never model capacity but data quality and geometry. The few misclassifications are between visually similar games (photorealistic shooters, or sandbox titles like Minecraft ↔ Roblox). Full report in `classification_report.txt`, confusion matrix in `confusion_matrix.png`, training curves in `training_curves.png`.

## Repository

```
config.py            shared config — classes, paths, hyperparameters
data/                collection · cleaning · merging · DataLoaders   → data/README.md
model/               ResNet-50 · training · evaluation               → model/README.md
demo/                Gradio app + screen detection                   → demo/README.md
DATA_COLLECTION.md   full data-collection methodology
best_model.pth       shipped 17-class model (test 99.53%)
```

**Quick start** (from the project root):

```bash
pip install -r requirements.txt
python demo/app.py        # launch the demo at http://localhost:7860
```

Detailed commands for data collection, training, and evaluation live in `DATA_COLLECTION.md` and the per-folder READMEs.

## Authors

Yizhuo Li · Elaine Wang · Cecilia Hua · Cassie Li

## License

MIT — see [LICENSE](LICENSE). Data comes from public sources (HuggingFace + YouTube gameplay) and is used for study and research only; all game imagery belongs to its respective publishers.
