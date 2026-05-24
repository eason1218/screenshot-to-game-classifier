# Model team `model/`

Owns the network architecture, training loop, and evaluation / metrics.

## Files

| File | Responsibility |
|------|----------------|
| `model.py` | ResNet-50 (IMAGENET1K_V2 pretrained, FC layer replaced); `get_model(num_classes, pretrained)` |
| `train.py` | Training loop, saves the best checkpoint |
| `eval.py` | Test-set evaluation, produces the report & figures |

## Usage (run from the project root)

```powershell
# Train -> saves best_model.pth + training_history.json (best on validation)
$env:DATA_SOURCE='local'; python model/train.py

# Evaluate -> classification_report.txt + confusion_matrix.png + training_curves.png
$env:DATA_SOURCE='local'; python model/eval.py
```

- Hyperparameters: Adam, lr=1e-4, batch 32 (see root `config.py`; `NUM_EPOCHS=5`).
- **Set `DATA_SOURCE=local`** to train on the 17-class merged set; without it, training falls back to the online HF 10-class baseline.
- Data comes from the data team: `from data.data import get_dataloaders`.

## Current metrics (17-class merged set)

| Metric | Value |
|--------|-------|
| Best validation accuracy | 99.65% (epoch 4, early-stopped) |
| Test accuracy | 99.53% (1,700 images) |

## Notes

- **Artifact paths are relative to the project root** (`best_model.pth` / `training_history.json`), so always run from the root.
- When running in the background, Python stdout is block-buffered; add `$env:PYTHONUNBUFFERED='1'` for live logs, and use `nvidia-smi` to confirm the GPU is actually busy rather than staring at an empty log.
- The GPU is RTX 50-series (Blackwell, sm_120), so torch must be the cu128 build, otherwise `cuda.is_available()` is True but kernels error at runtime.
- Changing the class count requires updating `config.NUM_CLASSES` + `config.CLASS_NAMES` and **retraining** (the FC dimension changes).

## Dependencies

`torch` `torchvision` `scikit-learn` `matplotlib` `seaborn` `tqdm`
