# 🎮 Screenshot to Game Classifier

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--50-EE4C2C?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Demo-Gradio-F97316)
![Accuracy](https://img.shields.io/badge/test%20accuracy-99.53%25-success)
![License](https://img.shields.io/badge/license-MIT-green)

Identify which of **17 popular games** a screenshot belongs to. It works on direct game screenshots, and also on a photo taken of someone else's phone/monitor screen — it automatically detects the screen boundary, corrects the perspective, and then classifies.

Machine Learning 2 Final Project.

## Features

- **Direct screenshot classification** — upload a game screenshot, get Top-3 predictions with confidence.
- **Photo-of-screen mode** — take a photo of a phone/monitor running a game; the screen is detected and perspective-corrected before classification.
- **Screen detection pipeline**: Mobile SAM (semantic segmentation, primary) → brightness percentile → Otsu → Canny edges (fallbacks).
- **Data collection tools** — download gameplay videos from YouTube, sample frames over a time window, deduplicate with perceptual hashing.

## Classes (17-class merged set)

Classic HuggingFace 10-class + self-collected YouTube 10-class, de-duplicated on the overlapping games into **17 classes**:

ARCRaiders · Among Us · Apex Legends · CounterStrike2 · Fortnite · Forza Horizon · Free Fire · Genshin Impact · God of War · LeagueOfLegends · MarvelRivals · Minecraft · Roblox · RocketLeague · Subnautica2 · Terraria · Valorant

> Fortnite / Minecraft / Genshin Impact appear in both sources and were merged.

The default local dataset is `dataset_combined/` (built by `data/build_combined_dataset.py` from `dataset_hf/` + `dataset_youtube_hq/` via hard links, 20,000 images total; each class is balanced to 1,000 at training time).

## Model performance (17-class merged set)

| Item | Value |
|------|-------|
| Architecture | ResNet-50 (fine-tuned, FC layer replaced → 17 classes) |
| Pretrained weights | IMAGENET1K_V2 |
| Train / Val / Test | 13,600 / 1,700 / 1,700 (17 classes balanced to 1,000, stratified 80/10/10) |
| Best validation accuracy | **99.65%** (epoch 4, early-stopped) |
| Test accuracy | **99.53%** (1,700 images) |
| Training | saturates in 4 epochs (RTX 5090, ~1 min/epoch) |

> Validation accuracy reached 99.65% in just 4 epochs, so training was stopped early. Per-class test accuracy is 98–100%.

### Geometry (key design)

Inputs go through **letterboxing** (aspect-preserving resize + center-pad with black bars to 224×224), and **training / evaluation / demo inference all share the exact same geometry**.
Compared to squashing 16:9 into a square (distortion) or Resize+CenterCrop (drops edge HUD/minimap), letterboxing distorts nothing and loses nothing. (In an earlier 10-class experiment, unifying the geometry raised test accuracy from 99.40% to 99.90%.)

### Augmentation (training set only)

`LetterboxResize` → `RandomAffine(scale 0.85–1.0, translate 0.05)` → `RandomHorizontalFlip` · `RandomRotation(±15°)` · `RandomPerspective(0.4)` · `ColorJitter` · `RandomGrayscale` · `GaussianBlur` · `RandomErasing`

The perspective/rotation/blur augmentations specifically target the "photo of a screen" scenario, making the model robust to tilted, warped, and out-of-focus game frames.

## Project structure

The project is split into three sub-packages by team role; `config.py` is the shared contract and stays at the root:

```
├── config.py                      # [shared] hyperparameters, class names, path constants; DATASET_DIR=dataset_combined
├── data/                          # [data team]
│   ├── data.py                    #   HF / local loading, stratified split, balanced sampling, augmentation
│   ├── collect_data.py            #   YouTube download + time-window frame sampling + pHash dedup (engine)
│   ├── collect_youtube_to_1000.py #   top up each YouTube class to 1,000 images via preset search queries
│   ├── export_hf_dataset.py       #   export the HF baseline to a local ImageFolder
│   ├── build_combined_dataset.py  #   merge dataset_hf + dataset_youtube_hq → 17 classes
│   ├── clean_data.py              #   filter corrupt / low-quality / duplicate frames
│   └── report_data.py             #   class distribution, size distribution, split plan
├── model/                         # [model team]
│   ├── model.py                   #   ResNet-50 (FC layer replaced → 17 classes)
│   ├── train.py                   #   training loop, saves the best checkpoint
│   └── eval.py                    #   test evaluation, confusion matrix, training curves
├── demo/                          # [demo team]
│   ├── app.py                     #   Gradio UI (screenshot / photo-of-screen; webcam auto-starts in photo mode)
│   └── screen_crop.py             #   phone-screen detection + perspective correction
├── DATA_COLLECTION.md             # data collection method & reproduction steps
├── requirements.txt
├── LICENSE                        # MIT
└── best_model.pth                 # shipped: trained 17-class model (test 99.53%)
```

> **Run convention**: always run scripts **from the project root** (e.g. `python model/train.py`).
> Scripts auto-add the root to `sys.path` to resolve `import config`; `best_model.pth` /
> `examples/` / `mobile_sam.pt` / `dataset/` are all paths relative to the root, so running from another directory will not find them.

### Team roles

| Sub-package | Owner | Responsibilities |
|-------------|-------|------------------|
| `data/` | Data team | YouTube collection, quality cleaning, data reports, local/HF DataLoaders, augmentation |
| `model/` | Model team | Network architecture, training loop, evaluation & metrics |
| `demo/` | Demo team | Gradio frontend, photo-of-screen detection & perspective correction |
| `config.py` | Shared | Changes require coordination (hyperparameters/class names/paths affect all three) |

## Installation

```bash
pip install -r requirements.txt

# Recommended for photo-of-screen mode: Mobile SAM screen segmentation
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

> **GPU note**: on RTX 50-series (Blackwell, sm_120), install the cu128 build of torch (nightly). cu121/cu124 builds do not ship kernels for that architecture — `cuda.is_available()` returns True but kernels will error at runtime.

---

## Launch the demo

```bash
python demo/app.py
```

- After launch, open **http://localhost:7860** locally.
- `demo.launch(share=True)` in `app.py` also prints a public `*.gradio.live` link you can share temporarily.
- The UI has two modes (radio toggle): **Screenshot** (upload/paste) and **Photo of Screen** (switching to it auto-starts the webcam; take one shot and it detects the screen + corrects perspective automatically).
- The first time you enter photo mode, the Mobile SAM weights (~40 MB) are downloaded; if `huggingface.co` is unreachable, download manually first:

```bash
curl.exe -L https://hf-mirror.com/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt -o mobile_sam.pt
```

> Note: `app.py` reuses `data.py`'s `LetterboxResize` for inference preprocessing, kept strictly identical to the training geometry — any preprocessing change must be mirrored in all three places (`train_tf` / `eval_tf` / `app._transform`).

---

## Data collection

> 📄 **Full data collection methodology** (where the data comes from, how frames are sampled & deduplicated, how to reproduce) is in [`DATA_COLLECTION.md`](DATA_COLLECTION.md). Quick commands below.

### HuggingFace baseline

To use the public baseline dataset, download/export it locally:

```bash
python data/export_hf_dataset.py --output-dir dataset_hf
python data/report_data.py --dataset-dir dataset_hf --output dataset_hf_report.json
```

The local export is `dataset_hf/<GameName>/hf_XXXXX.png`, 10 classes × 1,000 images. It does **not** include Valorant, League of Legends, or Rocket League — those come from the self-collected YouTube set.

### Prerequisite: YouTube download dependency

YouTube now gates video formats behind a JS "n-challenge"; without a JS runtime you get `n challenge solving failed` / `No supported JavaScript runtime`. Install **deno**:

```bash
winget install denoland.deno
```

`collect_data.py` **auto-detects** a winget/scoop-installed deno and injects it into PATH (`ensure_deno_on_path()`), no manual env-var needed. Downloads go through the YouTube Android / Android VR client (`player_client=android_vr,android`) to bypass the web client's SABR restriction and get a real MP4 stream.

### Usage

```bash
# Single YouTube URL
python data/collect_data.py --game "Minecraft" --url "https://youtu.be/XXX"

# Sample only a time window (SS / MM:SS / HH:MM:SS)
python data/collect_data.py --game "Minecraft" --url "https://youtu.be/XXX" --start-time 1:30 --end-time 5:00

# YouTube search (downloads the top N results)
python data/collect_data.py --game "Fortnite" --search "Fortnite gameplay 2024 no commentary" --max-videos 3

# Top up all 10 YouTube classes to 1,000 images each
python data/collect_youtube_to_1000.py
```

> Commands are single-line — this project runs under PowerShell, which doesn't accept bash's `\` line-continuation (use a backtick `` ` ``). Single-line commands paste cleanly into both bash and PowerShell.

### Clean & report local data

```bash
# Dry run first: check corrupt / low-res / low-information / too-dark-or-bright / near-duplicate frames
python data/clean_data.py --dataset-dir dataset_youtube_hq

# After reviewing the report, move rejected images out
python data/clean_data.py --dataset-dir dataset_youtube_hq --apply

# Generate dataset statistics
python data/report_data.py --dataset-dir dataset_youtube_hq
```

### Build the 17-class training set

```bash
python data/build_combined_dataset.py
```

Hard-links `dataset_hf/` (classic 10) + `dataset_youtube_hq/` (self-collected 10) into `dataset_combined/`, de-duplicating Fortnite / Minecraft / Genshin Impact → 17 classes, 20,000 images, and writes `combined_manifest.csv` recording each image's source.

---

## Training & evaluation

The repo ships a trained `best_model.pth` (17 classes); to retrain, follow below. **You must set `DATA_SOURCE=local`**, otherwise it downloads the online HF 10-class baseline, which doesn't match the 17-class merged set.

```powershell
# Train — uses the local 17-class merged set; saves best_model.pth and training_history.json
$env:DATA_SOURCE='local'; python model/train.py

# Evaluate — prints the classification report, saves confusion_matrix.png / training_curves.png
$env:DATA_SOURCE='local'; python model/eval.py
```

> Without `DATA_SOURCE`, it defaults to the online HF 10-class baseline (keeps the original reproducible experiment).
> When running in the background, Python stdout is block-buffered — add `$env:PYTHONUNBUFFERED='1'` for live logs; use `nvidia-smi` to confirm the GPU is actually busy.

## Photo-of-screen mode

Upload a real photo of a phone/monitor running a game:

1. First, **Mobile SAM** semantically segments the screen region.
2. If SAM is unavailable, fall back in order to: brightness-percentile threshold → Otsu → Canny edge detection.
3. The detected quadrilateral is perspective-corrected via a homography.
4. The corrected image is fed to the classifier (also through `LetterboxResize`).

Green corner markers on the preview indicate a successful screen detection.

## Dataset

**17-class merged set** = two sources de-duplicated and combined, built into `dataset_combined/` by `data/build_combined_dataset.py`:

1. **Classic HF baseline** — [Bingsu/Gameplay_Images](https://huggingface.co/datasets/Bingsu/Gameplay_Images), 10 classes × 1,000, exported to `dataset_hf/`.
2. **Self-collected YouTube** — 10 classes × 1,000 frames sampled from real gameplay videos, in `dataset_youtube_hq/`.

After merging, Fortnite / Minecraft / Genshin Impact are de-duplicated → 17 classes, 20,000 images, 640×360 PNG. At training time, each class is balanced to 1,000 and `data.py` applies a stratified 80/10/10 split (`seed=42`, reproducible).

## Classification results (test set, 17 classes × 100 = 1,700)

Overall test accuracy **99.53%**, per-class 98–100%. Full report in `classification_report.txt`:

```
                 precision    recall  f1-score   support

     ARCRaiders       1.00      1.00      1.00       100
       Among Us       1.00      1.00      1.00       100
   Apex Legends       0.99      0.98      0.98       100
 CounterStrike2       1.00      1.00      1.00       100
       Fortnite       1.00      0.99      0.99       100
  Forza Horizon       1.00      1.00      1.00       100
      Free Fire       1.00      0.99      0.99       100
 Genshin Impact       0.99      0.99      0.99       100
     God of War       0.99      1.00      1.00       100
LeagueOfLegends       1.00      1.00      1.00       100
   MarvelRivals       0.99      1.00      1.00       100
      Minecraft       1.00      0.98      0.99       100
         Roblox       0.98      0.99      0.99       100
   RocketLeague       1.00      1.00      1.00       100
    Subnautica2       1.00      1.00      1.00       100
       Terraria       0.99      1.00      1.00       100
       Valorant       0.99      1.00      1.00       100

       accuracy                           1.00      1700
      macro avg       1.00      1.00      1.00      1700
   weighted avg       1.00      1.00      1.00      1700
```

Misclassifications are in the single digits, concentrated between photorealistic / same-genre games. See `confusion_matrix.png`.

## Authors

- Yizhuo Li
- Elaine Wang
- Cecilia Hua
- Cassie Li

## License

Released under the [MIT License](LICENSE).

> Data comes from [Bingsu/Gameplay_Images](https://huggingface.co/datasets/Bingsu/Gameplay_Images) and frames sampled from public YouTube gameplay videos, used for study and research only; all game imagery is the property of its respective publishers.
