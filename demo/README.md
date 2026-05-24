# Demo team `demo/`

Owns the Gradio frontend and the photo-of-screen detection + perspective correction.

## Files

| File | Responsibility |
|------|----------------|
| `app.py` | Gradio Blocks UI, two modes (radio toggle) |
| `screen_crop.py` | Phone / monitor screen detection + perspective correction; `crop_screen()` / `load_sam()` |

## Usage (run from the project root)

```bash
python demo/app.py
```

- Local: http://localhost:7860
- `demo.launch(share=True)` also prints a public `*.gradio.live` link (valid for 1 week).
- Two modes: **Screenshot** / **Photo of Screen** (switching to it auto-starts the webcam; take a photo and it's screen-detected + perspective-corrected before classification).

## Screen detection pipeline (`crop_screen`)

Four strategies tried in order, returning the first that succeeds:
**Mobile SAM segmentation** (primary) → brightness-percentile threshold → Otsu → Canny edges (fallback).
The detected quadrilateral is perspective-corrected via a homography before being sent to the classifier.

## Notes

- **Preprocessing must match training**: `app.py`'s `_transform` reuses `from data.data import LetterboxResize`; if you change it, keep all three (`train_tf` / `eval_tf` / `app._transform`) in sync, or inference geometry breaks and accuracy drops.
- The `warnings.filterwarnings` calls at the top of `app.py` must come **before** `from demo.screen_crop import ...` (to suppress timm / mobile_sam noise) — don't reorder them.
- Depends on `best_model.pth` (produced by the model team, at the project root), so run from the root.
- Mobile SAM weights `mobile_sam.pt` (~40 MB) download automatically on first use; if `huggingface.co` is unreachable, download to the project root first:
  ```bash
  curl.exe -L https://hf-mirror.com/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt -o mobile_sam.pt
  ```
- Example images come from the project-root `examples/`; `load_sam()` warms up at startup so the first photo isn't slow.

## Dependencies

`gradio` `torch` `torchvision` `Pillow` `opencv-python` `numpy`
Screen segmentation: `mobile_sam` (`pip install git+https://github.com/ChaoningZhang/MobileSAM.git`)
