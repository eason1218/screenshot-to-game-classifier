# Data team `data/`

Handles the data side: collecting gameplay frames from public videos, cleaning low-quality / duplicate images, reporting class balance, and feeding local data into the PyTorch training pipeline.

> 📄 For the full collection methodology and reproduction steps, see [`../DATA_COLLECTION.md`](../DATA_COLLECTION.md).

## Files

| File | Responsibility |
|------|----------------|
| `collect_data.py` | YouTube / local video frame extraction; time-window sampling; pHash dedup + provenance manifest |
| `collect_youtube_to_1000.py` | Top up each YouTube class to 1,000 images via preset search queries |
| `export_hf_dataset.py` | Download / export the HF `Bingsu/Gameplay_Images` baseline to a local ImageFolder |
| `build_combined_dataset.py` | Merge `dataset_hf` + `dataset_youtube_hq` into the 17-class `dataset_combined` |
| `clean_data.py` | Quality control: corrupt, low-res, too dark / bright, low-information, near-duplicate images |
| `report_data.py` | Dataset stats: class counts, balance ratio, image sizes, planned split |
| `data.py` | Train/val/test DataLoaders for both the HF baseline and local ImageFolder data |

## Recommended workflow

```bash
# 1. Classic baseline (10 classes)
python data/export_hf_dataset.py --output-dir dataset_hf

# 2. Self-collected YouTube frames (10 classes; needs deno — see DATA_COLLECTION.md)
python data/collect_youtube_to_1000.py
python data/clean_data.py --dataset-dir dataset_youtube_hq --apply
python data/report_data.py --dataset-dir dataset_youtube_hq

# 3. Merge into the 17-class training set
python data/build_combined_dataset.py
```

## Public API

```python
from data.data import get_dataloaders, get_local_dataloaders, get_transforms, LetterboxResize
```

- `get_dataloaders(source="hf")`: use the online HF `Bingsu/Gameplay_Images` baseline.
- `get_dataloaders(source="local")`: use `config.DATASET_DIR` (currently `dataset_combined`).
- `get_dataloaders(source="auto")`: prefer local; fall back to HF if local classes are insufficient.
- `LetterboxResize`: aspect-preserving resize + center-pad to 224×224 — must stay identical across training, evaluation, and the demo.

## Cross-team contract

- Adding / replacing game classes requires updating `config.CLASS_NAMES` and `config.NUM_CLASSES` in sync.
- Local `ImageFolder` labels follow alphabetical folder order; before the final training run, make `config.CLASS_NAMES` match the printed `Class mapping` to avoid label mismatch.
- `demo/app.py` reuses `LetterboxResize` for inference — don't change the demo preprocessing separately.
- The root `best_model.pth` is the trained 17-class model; changing classes means retraining.

## For the final report (data section)

Our data pipeline collects gameplay screenshots from publicly available gameplay videos. For each target game, we use high-resolution gameplay videos whenever possible, extract frames at a controlled sampling rate, resize frames to 640×360, and remove near-duplicate frames using perceptual hashing. We then run a quality-control script to filter invalid images, low-resolution frames, low-information frames, overly dark/bright frames, and remaining near-duplicates. The cleaned dataset is organized in ImageFolder format with one folder per game class. We generate a dataset report containing class counts, image size distribution, class-balance ratio, and the planned stratified 80/10/10 train/validation/test split. During training, images are loaded through a reproducible PyTorch DataLoader with stratified splitting, optional class balancing, and augmentations designed for real-world screenshots and photos of screens.
