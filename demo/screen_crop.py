"""
screen_crop.py — Detect a phone/monitor screen in a photo and correct perspective.

Primary strategy  : Mobile SAM (semantic segmentation — robust to irregular,
                    tilted, or partially-occluded screens)
Fallback strategies: percentile brightness → Otsu → Canny edges

Mobile SAM is auto-downloaded (~40 MB) on first use if `mobile-sam` is installed:
    pip install mobile-sam
Without it the three traditional CV strategies run instead.
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

_ASPECT_LO = 1.1
_ASPECT_HI  = 4.5

# ── Mobile SAM (lazy singleton) ───────────────────────────────────────────────

_sam_gen    = None   # SamAutomaticMaskGenerator, or None
_sam_tried  = False  # True after the first load attempt

MOBILE_SAM_CKPT = "mobile_sam.pt"
MOBILE_SAM_URL  = (
    "https://huggingface.co/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt"
)


def load_sam() -> object | None:
    """
    Load Mobile SAM once and cache it.
    Returns the SamAutomaticMaskGenerator, or None if unavailable.
    Safe to call multiple times.
    """
    global _sam_gen, _sam_tried
    if _sam_tried:
        return _sam_gen
    _sam_tried = True
    try:
        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
        import torch

        ckpt = Path(MOBILE_SAM_CKPT)
        if not ckpt.exists():
            import urllib.request
            print("Downloading Mobile SAM weights (~40 MB)…")
            urllib.request.urlretrieve(MOBILE_SAM_URL, ckpt)
            print("Download complete.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_t"](checkpoint=str(ckpt))
        sam.to(device).eval()
        _sam_gen = SamAutomaticMaskGenerator(
            sam,
            points_per_side=16,
            pred_iou_thresh=0.85,
            stability_score_thresh=0.90,
            min_mask_region_area=200,
        )
        print(f"Mobile SAM ready ({device}).")
        return _sam_gen
    except Exception as e:
        print(f"Mobile SAM unavailable ({e}); using traditional CV.")
        return None


# ── Geometry ──────────────────────────────────────────────────────────────────

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Sort 4 corner points into TL, TR, BR, BL order."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s, diff = pts.sum(axis=1), np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _perspective_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Warp the detected quadrilateral into a flat rectangle."""
    rect = _order_points(pts)
    tl, tr, br, bl = rect
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if w < 10 or h < 10:
        return image
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w, h))


# ── Quad validation ───────────────────────────────────────────────────────────

def _valid_quad(approx: np.ndarray, img_h: int, img_w: int,
                min_area_ratio: float) -> np.ndarray | None:
    """
    Accept a 4-point polygon with:
      • area  ≥ min_area_ratio × image
      • aspect ratio in [_ASPECT_LO, _ASPECT_HI] (from actual edge lengths)
    """
    if len(approx) != 4:
        return None
    pts  = approx.reshape(4, 2).astype(np.float32)
    area = cv2.contourArea(pts)
    if area < min_area_ratio * img_h * img_w:
        return None
    tl, tr, br, bl = _order_points(pts)
    w = float(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = float(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if min(w, h) < 10:
        return None
    ratio = max(w, h) / min(w, h)
    if _ASPECT_LO <= ratio <= _ASPECT_HI:
        return pts
    return None


def _search_quads(binary: np.ndarray, img_h: int, img_w: int,
                  min_area_ratio: float) -> np.ndarray | None:
    """Scan contours of a binary image for the largest valid screen quad."""
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        if cv2.contourArea(cnt) < min_area_ratio * img_h * img_w:
            break
        peri = cv2.arcLength(cnt, True)
        for eps in [0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.13]:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            result = _valid_quad(approx, img_h, img_w, min_area_ratio)
            if result is not None:
                return result
    return None


# ── Quad extraction from a binary mask ───────────────────────────────────────

def _quad_from_mask(mask: np.ndarray, img_h: int, img_w: int,
                    min_area_ratio: float) -> np.ndarray | None:
    """
    Given a binary SAM mask, extract the best quadrilateral.

    Tries three approaches in order:
      1. ConvexHull + polygon approximation   — clean quad from complete screens
      2. MinAreaRect                           — handles irregular/incomplete masks
      3. Bounding rect                         — last resort
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)

    # 1. Convex hull approximation
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for eps in [0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.15, 0.18]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        pts = _valid_quad(approx, img_h, img_w, min_area_ratio)
        if pts is not None:
            return pts

    # 2. Minimum area rectangle (handles rotated / irregular shapes)
    rot_rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rot_rect).astype(np.float32)
    pts = _valid_quad(box.reshape(4, 1, 2).astype(np.int32),
                      img_h, img_w, min_area_ratio * 0.5)
    if pts is not None:
        return pts

    # 3. Axis-aligned bounding rectangle
    x, y, bw, bh = cv2.boundingRect(cnt)
    if bw * bh < min_area_ratio * 0.5 * img_h * img_w:
        return None
    box2 = np.array([[x, y], [x+bw, y], [x+bw, y+bh], [x, y+bh]], dtype=np.float32)
    return _valid_quad(box2.reshape(4, 1, 2).astype(np.int32),
                       img_h, img_w, min_area_ratio * 0.5)


# ── Strategy 0: Mobile SAM ────────────────────────────────────────────────────

def _by_sam(rgb: np.ndarray, img_h: int, img_w: int,
            min_area_ratio: float) -> np.ndarray | None:
    """
    Use Mobile SAM to segment all regions and pick the best screen quad.

    Advantages over traditional CV:
      • Works with complex/textured backgrounds
      • Handles partially-occluded or irregularly-lit screens
      • Not fooled by bright backgrounds or dark-content screens
    """
    gen = load_sam()
    if gen is None:
        return None
    try:
        masks = gen.generate(rgb)
    except Exception:
        return None

    # Sort by (area × predicted IoU) so we inspect the most confident large masks first
    masks.sort(key=lambda m: m["area"] * m.get("predicted_iou", 1.0), reverse=True)

    for m in masks:
        if m["area"] < min_area_ratio * img_h * img_w:
            continue
        seg = (m["segmentation"] * 255).astype(np.uint8)
        pts = _quad_from_mask(seg, img_h, img_w, min_area_ratio)
        if pts is not None:
            return pts
    return None


# ── Strategy 1: percentile brightness ────────────────────────────────────────

def _by_percentile(gray: np.ndarray, img_h: int, img_w: int,
                   min_area_ratio: float) -> np.ndarray | None:
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    kernel  = np.ones((25, 25), np.uint8)
    for pct in [60, 70, 50, 80]:
        threshold = int(np.percentile(blurred, pct))
        if threshold < 10:
            continue
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))
        result = _search_quads(closed, img_h, img_w, min_area_ratio)
        if result is not None:
            return result
    return None


# ── Strategy 2: Otsu threshold ────────────────────────────────────────────────

def _by_otsu(gray: np.ndarray, img_h: int, img_w: int,
             min_area_ratio: float) -> np.ndarray | None:
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((25, 25), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return _search_quads(closed, img_h, img_w, min_area_ratio)


# ── Strategy 3: bezel edge detection ─────────────────────────────────────────

def _by_edges(gray: np.ndarray, img_h: int, img_w: int,
              min_area_ratio: float) -> np.ndarray | None:
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    for blur_k in [7, 11, 15]:
        blurred = cv2.GaussianBlur(enhanced, (blur_k, blur_k), 0)
        for lo, hi in [(15, 60), (25, 100), (40, 140)]:
            edges   = cv2.Canny(blurred, lo, hi)
            dilated = cv2.dilate(edges, np.ones((9, 9), np.uint8), iterations=3)
            result  = _search_quads(dilated, img_h, img_w, min_area_ratio)
            if result is not None:
                return result
    return None


# ── Annotation ────────────────────────────────────────────────────────────────

def _draw_boundary(rgb: np.ndarray, pts: np.ndarray) -> np.ndarray:
    out     = rgb.copy()
    pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts_int], isClosed=True, color=(34, 197, 94), thickness=4)
    for pt in pts.astype(np.int32):
        cv2.circle(out, tuple(pt), 12, (34, 197, 94), -1)
        cv2.circle(out, tuple(pt), 12, (255, 255, 255), 2)
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def crop_screen(
    pil_image: Image.Image,
    min_area_ratio: float = 0.04,
) -> tuple[Image.Image, Image.Image | None, bool]:
    """
    Detect a phone/monitor screen in a photo and return the corrected content.

    Tries four strategies in order:
      0. Mobile SAM  (semantic, handles irregular / incomplete screens)
      1. Percentile brightness threshold
      2. Otsu threshold
      3. Bezel edge detection

    Returns:
        (annotated, corrected, found)
        annotated  – Original photo with detected corners marked in green.
        corrected  – Perspective-corrected screen, or None if not found.
        found      – True when a screen was successfully detected.
    """
    rgb  = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    corners = _by_sam(rgb, h, w, min_area_ratio)
    if corners is None:
        corners = _by_percentile(gray, h, w, min_area_ratio)
    if corners is None:
        corners = _by_otsu(gray, h, w, min_area_ratio)
    if corners is None:
        corners = _by_edges(gray, h, w, min_area_ratio)

    if corners is None:
        return pil_image, None, False

    annotated = _draw_boundary(rgb, corners)
    corrected = _perspective_transform(rgb, corners)
    return Image.fromarray(annotated), Image.fromarray(corrected), True
