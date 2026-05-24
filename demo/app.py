"""
app.py — Gradio demo with screen-detection preprocessing.

Accepts:
  • Direct screenshots
  • Photos taken of a screen (at any angle)
    → automatically detects the screen boundary and corrects perspective

Run:
    python demo/app.py
"""
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"mobile_sam.*")
warnings.filterwarnings(
    "ignore", category=UserWarning, message=r".*Overwriting tiny_vit.*"
)

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import torch
from datasets import load_dataset
from PIL import Image
from torchvision import transforms

import config
from data.data import LetterboxResize
from model.model import get_model
from demo.screen_crop import crop_screen, load_sam

# ── Model setup ───────────────────────────────────────────────────────────────
device = torch.device(config.DEVICE)
model  = get_model(num_classes=config.NUM_CLASSES, pretrained=False)
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
model.to(device).eval()

load_sam()

_transform = transforms.Compose([
    LetterboxResize(config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

GAME_ICONS = {
    # classic HF 10-class
    "Among Us":        "👾", "Apex Legends":   "🎯",
    "Fortnite":        "⚡", "Forza Horizon":  "🏎️",
    "Free Fire":       "🔥", "Genshin Impact": "✨",
    "God of War":      "⚔️", "Minecraft":      "🧱",
    "Roblox":          "🎮", "Terraria":       "🌍",
    # YouTube 10-class additions
    "ARCRaiders":      "🤖", "CounterStrike2": "🔫",
    "LeagueOfLegends": "🗡️", "MarvelRivals":   "🦸",
    "RocketLeague":    "🚗", "Subnautica2":    "🌊",
    "Valorant":        "💥",
}

# ── Inference ─────────────────────────────────────────────────────────────────

def _do_classify(image: Image.Image, mode: str):
    """Core classification — returns (preview_image, result_html)."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    if mode == "photo":
        annotated, corrected, found = crop_screen(image)
        if found:
            to_classify = corrected
            preview     = annotated
            status      = ("screen", "Screen detected — perspective corrected")
        else:
            to_classify = image
            preview     = image
            status      = ("full", "No screen found — classifying full image")
    else:
        to_classify = image
        preview     = image
        status      = ("screenshot", "Direct classification")

    tensor = _transform(to_classify).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().tolist()

    results = sorted(
        [(config.CLASS_NAMES[i], probs[i]) for i in range(config.NUM_CLASSES)],
        key=lambda x: x[1], reverse=True,
    )
    return preview, _result_html(results[:3], status)


def process(image: Optional[Image.Image], mode: str):
    """
    Triggered when a new image lands in img_input.
    Classifies it, clears img_input (left col goes back to upload zone),
    stores the image in State, and sends preview + results to the other columns.
    Returns 4 values: (img_input, img_preview, output, img_state).
    """
    if image is None:
        # Called again after we cleared img_input — do nothing.
        return gr.update(), gr.update(), gr.update(), gr.update()

    preview, html = _do_classify(image, mode)
    return None, preview, html, image  # clear left, fill middle, fill right, store


def reprocess(stored: Optional[Image.Image], mode: str):
    """Re-classify the stored image (used by btn click and mode change)."""
    if stored is None:
        return None, _empty_html()
    preview, html = _do_classify(stored, mode)
    return preview, html


# ── HTML builders ─────────────────────────────────────────────────────────────

def _result_html(top3: list, status: tuple) -> str:
    top_name, top_conf = top3[0]
    top_icon = GAME_ICONS.get(top_name, "🎮")

    conf_colors = ["#22d3ee", "#60a5fa", "#475569"]

    bars = ""
    for i, (name, conf) in enumerate(top3):
        pct  = conf * 100
        icon = GAME_ICONS.get(name, "🎮")
        bars += f"""
        <div style="background:rgba(15,23,42,0.6);
                    border:1px solid rgba(6,182,212,{0.35 if i==0 else 0.15});
                    border-radius:12px;padding:14px 16px;margin-bottom:10px;
                    transition:all 0.2s;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:9px;">
            <span style="font-size:1.15em;width:26px;text-align:center;">{icon}</span>
            <span style="flex:1;font-size:1em;font-family:system-ui,sans-serif;
                         font-weight:{'700' if i==0 else '500'};
                         color:{'#e2e8f0' if i==0 else '#64748b'};">{name}</span>
            <span style="font-size:0.95em;font-weight:700;font-family:monospace;
                         color:{conf_colors[i]};">{pct:.1f}%</span>
          </div>
          <div style="height:5px;background:rgba(15,23,42,0.8);border-radius:99px;overflow:hidden;">
            <div style="width:{pct:.1f}%;height:100%;border-radius:99px;
                        background:linear-gradient(90deg,#06b6d4,#3b82f6);
                        box-shadow:0 0 8px rgba(0,200,255,{0.6 if i==0 else 0.3});"></div>
          </div>
        </div>"""

    mode_label, mode_text = status
    if mode_label == "screen":
        badge_text = "[ SCREEN DETECTED ]"
    elif mode_label == "screenshot":
        badge_text = "[ ANALYZING... ]"
    else:
        badge_text = "[ NO SCREEN FOUND ]"

    return f"""
<div style="font-family:system-ui,-apple-system,sans-serif;padding:4px 2px;">
  <!-- Status badge -->
  <div style="display:flex;justify-content:center;margin-bottom:16px;">
    <span style="background:rgba(6,182,212,0.15);color:#22d3ee;
                 border:1px solid rgba(6,182,212,0.4);
                 border-radius:99px;padding:6px 18px;
                 font-size:0.75em;font-weight:700;font-family:monospace;
                 letter-spacing:1px;
                 box-shadow:0 0 15px rgba(0,200,255,0.2);">
      {badge_text}
    </span>
  </div>

  <!-- Winner card -->
  <div style="position:relative;overflow:hidden;
              background:linear-gradient(135deg,rgba(6,182,212,0.08),rgba(59,130,246,0.08));
              border:2px solid rgba(34,211,238,0.4);
              border-radius:16px;padding:22px 22px;margin-bottom:18px;
              box-shadow:0 0 40px rgba(0,200,255,0.15);">
    <!-- top glow line -->
    <div style="position:absolute;top:0;left:0;right:0;height:2px;
                background:linear-gradient(90deg,transparent,#22d3ee,transparent);"></div>
    <div style="font-size:0.65em;font-weight:700;font-family:monospace;
                color:#22d3ee;text-transform:uppercase;
                letter-spacing:3px;margin-bottom:8px;">&gt;&gt; TOP MATCH</div>
    <div style="font-size:2em;font-weight:900;letter-spacing:-0.5px;line-height:1.1;
                background:linear-gradient(90deg,#22d3ee,#93c5fd);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;">{top_name}</div>
    <div style="font-size:1.5em;font-weight:900;font-family:monospace;
                color:#22d3ee;margin-top:4px;">{top_conf * 100:.1f}%</div>
  </div>

  <!-- Predictions label -->
  <div style="font-size:0.68em;font-weight:700;font-family:monospace;
              color:rgba(103,232,249,0.6);text-transform:uppercase;
              letter-spacing:2px;margin-bottom:10px;">&gt; ALL PREDICTIONS</div>
  {bars}
</div>"""


def _empty_html() -> str:
    return """
<div style="font-family:system-ui,-apple-system,sans-serif;
            display:flex;flex-direction:column;align-items:center;justify-content:center;
            height:380px;text-align:center;
            border:2px dashed rgba(6,182,212,0.2);border-radius:16px;
            background:rgba(15,23,42,0.2);">
  <div style="font-size:3.5em;margin-bottom:16px;opacity:0.3;
              filter:drop-shadow(0 0 12px rgba(0,200,255,0.5));">🎮</div>
  <div style="font-size:0.88em;font-weight:500;font-family:monospace;
              color:rgba(103,232,249,0.5);letter-spacing:0.5px;">
    &gt; Upload an image to get started
  </div>
</div>"""


# ── Examples ──────────────────────────────────────────────────────────────────

def _load_examples(save_dir: str = "examples") -> list:
    os.makedirs(save_dir, exist_ok=True)
    targets = [
        (0,    "among_us"),
        (2000, "fortnite"),
        (5000, "genshin"),
        (7000, "minecraft"),
        (9000, "terraria"),
    ]
    paths = [os.path.join(save_dir, f"{n}.png") for _, n in targets]
    if all(os.path.exists(p) for p in paths):
        return paths
    try:
        print("Downloading example images …")
        ds = load_dataset("Bingsu/Gameplay_Images", split="train")
        for (idx, name), path in zip(targets, paths):
            if not os.path.exists(path):
                ds[idx]["image"].convert("RGB").save(path)
        return paths
    except Exception as exc:
        print(f"Warning: could not load examples ({exc})")
        return []


examples = _load_examples()

# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
/* ── Scan line animation ── */
@keyframes scan {
    0%   { top: -4px; }
    100% { top: 100vh; }
}
@keyframes cyan-pulse {
    0%, 100% { opacity: 0.5; }
    50%       { opacity: 1; }
}

/* ── Full-page black with cyan grid ── */
html {
    background: #000000 !important;
    background-color: #000000 !important;
}
body {
    background: #000000 !important;
    background-color: #000000 !important;
    background-image:
        linear-gradient(rgba(0,200,255,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,200,255,0.05) 1px, transparent 1px) !important;
    background-size: 50px 50px !important;
    color: #e2e8f0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
/* Gradio injects --body-background-fill via theme tokens; force override */
:root {
    --body-background-fill: #000000 !important;
    --block-background-fill: transparent !important;
    --panel-background-fill: transparent !important;
}
.gradio-container {
    background: transparent !important;
    background-color: transparent !important;
    max-width: 100% !important;
    width: 100% !important;
    min-height: 100vh !important;
    padding: 0 48px !important;
    box-sizing: border-box !important;
    margin: 0 !important;
    color: #e2e8f0 !important;
}
/* Hide Gradio sidebar / nav / footer */
.app > nav, .sidebar,
[class*="sidebar"], [class*="side-panel"] { display: none !important; }
footer { display: none !important; }
.block, .gr-form, .gr-box, .panel {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
/* Equal thirds, tops aligned, each col is its own content height */
.main-row {
    align-items: flex-start !important;
}
.main-row > .gr-column,
.main-row > div[class*="col"] {
    flex: 1 1 0 !important;
    min-width: 0 !important;
}

/* ── Mode toggle: pill tabs ── */
#mode-toggle {
    display: flex !important;
    justify-content: center !important;
    margin: 0 0 32px !important;
}
#mode-toggle .wrap {
    display: inline-flex !important;
    background: rgba(15,23,42,0.6) !important;
    border: 1px solid rgba(6,182,212,0.3) !important;
    border-radius: 50px !important;
    padding: 5px !important;
    gap: 4px !important;
    box-shadow: 0 0 20px rgba(0,200,255,0.15) !important;
    backdrop-filter: blur(8px) !important;
}
#mode-toggle label {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    background: transparent !important;
    border: none !important;
    border-radius: 40px !important;
    padding: 12px 32px !important;
    cursor: pointer !important;
    font-size: 1em !important;
    font-weight: 600 !important;
    color: rgba(103,232,249,0.5) !important;
    transition: all 0.2s !important;
    white-space: nowrap !important;
}
#mode-toggle label:has(input[type="radio"]:checked) {
    background: linear-gradient(90deg, #22d3ee, #60a5fa) !important;
    color: #000000 !important;
    box-shadow: 0 0 20px rgba(0,200,255,0.5) !important;
}
#mode-toggle input[type="radio"] { display: none !important; }

/* ── Shared base: upload + preview boxes ── */
.upload-col [data-testid="image"],
.preview-image {
    background: rgba(15,23,42,0.35) !important;
    border-radius: 16px !important;
    box-shadow: inset 0 0 30px rgba(0,200,255,0.07) !important;
    height: 380px !important;
    max-height: 380px !important;
    flex-shrink: 0 !important;
    overflow: hidden !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
/* Upload: dashed border */
.upload-col [data-testid="image"] {
    border: 2px dashed rgba(6,182,212,0.3) !important;
}
.upload-col [data-testid="image"]:hover,
.upload-col [data-testid="image"]:focus-within {
    border-color: #22d3ee !important;
    box-shadow: 0 0 30px rgba(0,200,255,0.25),
                inset 0 0 30px rgba(0,200,255,0.08) !important;
}
/* Preview: solid border */
.preview-image {
    border: 1px solid rgba(6,182,212,0.3) !important;
}
/* Strip inner Gradio borders/bg so they don't double up */
.upload-col [data-testid="image"] > div,
.preview-image [data-testid="image"],
.preview-image [data-testid="image"] > div {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}
/* Icons + placeholder text → cyan */
.upload-col svg, .preview-image svg {
    color: rgba(34,211,238,0.55) !important;
}
.preview-image svg {
    width: 80px !important;
    height: 80px !important;
}
.upload-col .wrap p,
.preview-image .wrap p,
.preview-image p {
    color: rgba(103,232,249,0.65) !important;
    font-family: monospace !important;
    font-size: 1em !important;
    letter-spacing: 0.5px !important;
}

/* ── Identify Game button ── */
button.primary {
    background: linear-gradient(90deg, #06b6d4 0%, #3b82f6 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #000 !important;
    font-weight: 800 !important;
    font-size: 1em !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 0 30px rgba(0,200,255,0.4) !important;
    transition: transform 0.15s, box-shadow 0.2s !important;
}
button.primary:hover {
    transform: scale(1.03) !important;
    box-shadow: 0 0 50px rgba(0,200,255,0.65) !important;
}
button.primary:active { transform: scale(0.98) !important; }

/* ── Examples ── */
.examples tbody td button, .examples .example {
    background: rgba(15,23,42,0.5) !important;
    border: 1px solid rgba(6,182,212,0.2) !important;
    border-radius: 8px !important;
    color: rgba(103,232,249,0.7) !important;
    font-size: 0.78em !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.examples tbody td button:hover {
    border-color: #22d3ee !important;
    box-shadow: 0 0 15px rgba(0,200,255,0.3) !important;
}

/* ── Strip Gradio's internal padding so results box sits flush at the top ── */
.results-col > *,
.results-col > * > *,
.results-col .wrap,
.results-col .prose {
    padding: 0 !important;
    margin: 0 !important;
}

/* ── Caption text ── */
.caption-text {
    color: rgba(103,232,249,0.45);
    font-size: 0.76em;
    margin-top: 10px;
    text-align: center;
    font-family: monospace;
}

/* ── Webcam (photo mode) ──
   The .upload-col rules pin [data-testid=image] to 380px with overflow:hidden,
   which clips the webcam capture button out of view. Undo that for .cam-box so
   the shutter button is visible and clickable. ── */
.cam-box [data-testid="image"],
.cam-box [data-testid="image"] > div {
    overflow: visible !important;
    height: auto !important;
    max-height: none !important;
    min-height: 360px !important;
}
.cam-box button {
    pointer-events: auto !important;
    z-index: 50 !important;
    opacity: 1 !important;
}
"""

HEADER_HTML = """
<!-- Scan line (fixed, full viewport) -->
<div style="position:fixed;left:0;right:0;height:2px;
            background:rgba(0,200,255,0.12);
            animation:scan 5s linear infinite;
            pointer-events:none;z-index:9999;top:0;"></div>
<!-- Corner brackets -->
<div style="position:fixed;top:0;left:0;width:48px;height:48px;
            border-left:2px solid rgba(0,200,255,0.35);
            border-top:2px solid rgba(0,200,255,0.35);
            pointer-events:none;z-index:9998;"></div>
<div style="position:fixed;top:0;right:0;width:48px;height:48px;
            border-right:2px solid rgba(0,200,255,0.35);
            border-top:2px solid rgba(0,200,255,0.35);
            pointer-events:none;z-index:9998;"></div>
<div style="position:fixed;bottom:0;left:0;width:48px;height:48px;
            border-left:2px solid rgba(0,200,255,0.35);
            border-bottom:2px solid rgba(0,200,255,0.35);
            pointer-events:none;z-index:9998;"></div>
<div style="position:fixed;bottom:0;right:0;width:48px;height:48px;
            border-right:2px solid rgba(0,200,255,0.35);
            border-bottom:2px solid rgba(0,200,255,0.35);
            pointer-events:none;z-index:9998;"></div>

<div style="text-align:center;padding:52px 24px 32px;position:relative;">
  <div style="display:flex;align-items:center;justify-content:center;
              gap:18px;margin-bottom:16px;flex-wrap:wrap;">
    <span style="font-size:3.2em;line-height:1;
                 filter:drop-shadow(0 0 12px rgba(0,200,255,0.8));
                 animation:cyan-pulse 2s ease-in-out infinite;">🎮</span>
    <span style="font-size:3.2em;font-weight:900;letter-spacing:-1px;
                 background:linear-gradient(90deg,#22d3ee 0%,#60a5fa 55%,#a78bfa 100%);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 background-clip:text;">
      Game Screenshot Classifier
    </span>
    <span style="font-size:1.8em;animation:cyan-pulse 1.5s ease-in-out infinite;">⚡</span>
  </div>
  <div style="color:rgba(103,232,249,0.75);font-size:1em;max-width:520px;
              margin:0 auto;line-height:1.6;font-family:monospace;letter-spacing:0.3px;">
    &gt; Identify games from screenshots using AI-powered image recognition
  </div>
</div>
"""

# When Photo mode reveals the webcam, auto-click its start button so the user
# doesn't have to. Runs right after the mode-switch (still within the user's
# click), so the browser allows getUserMedia; first time it may still show a
# one-off permission prompt, after which the camera opens immediately.
START_WEBCAM_JS = """
() => {
  const click = (n) => {
    const box = document.querySelector('.cam-box');
    if (!box || box.offsetParent === null) return;       // hidden → Screenshot mode
    if (box.querySelector('video')) return;              // camera already live
    let btn = box.querySelector('button[aria-label*="webcam" i], button[aria-label*="camera" i], button[title*="camera" i]');
    if (!btn) btn = box.querySelector('button');
    if (btn) { btn.click(); return; }
    if (n > 0) setTimeout(() => click(n - 1), 200);
  };
  setTimeout(() => click(25), 200);
}
"""

# ── Layout ─────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Game Screenshot Classifier") as demo:

    gr.HTML(HEADER_HTML)

    mode = gr.Radio(
        choices=[
            ("🖼️  Screenshot", "screenshot"),
            ("📷  Photo of Screen", "photo"),
        ],
        value="screenshot",
        label="",
        container=False,
        elem_id="mode-toggle",
    )

    with gr.Row(equal_height=True, elem_classes=["main-row"]):

        with gr.Column(scale=1, elem_classes=["upload-col"]):
            # Screenshot mode: upload/clipboard. Shown by default.
            img_input = gr.Image(
                type="pil",
                sources=["upload", "clipboard"],
                label="",
                height=320,
                show_label=False,
            )
            # Photo mode: live webcam. Hidden until "Photo of Screen" is selected.
            cam_input = gr.Image(
                type="pil",
                sources=["webcam"],
                label="",
                show_label=False,
                visible=False,
                elem_classes=["cam-box"],
            )
            btn = gr.Button("⚡  Identify Game", variant="primary", size="lg")
            if examples:
                gr.Examples(examples=[[p] for p in examples], inputs=img_input, label="")

        with gr.Column(scale=1, elem_classes=["preview-col"]):
            img_preview = gr.Image(
                label="",
                show_label=False,
                height=320,
                interactive=False,
                elem_classes=["preview-image"],
            )
            gr.HTML('<div class="caption-text">&gt; Image preview</div>')

        with gr.Column(scale=1, elem_classes=["results-col"]):
            output = gr.HTML(value=_empty_html(), show_label=False)

    img_state = gr.State(None)

    # Upload → classify, clear left col, store image in state
    img_input.change(
        fn=process,
        inputs=[img_input, mode],
        outputs=[img_input, img_preview, output, img_state],
    )
    # Webcam capture (photo mode) → same classify path
    cam_input.change(
        fn=process,
        inputs=[cam_input, mode],
        outputs=[cam_input, img_preview, output, img_state],
    )
    # Button → re-classify the stored image
    btn.click(fn=reprocess, inputs=[img_state, mode], outputs=[img_preview, output])

    # Mode switch → swap upload/webcam input AND re-classify the stored image.
    # "Photo of Screen" reveals the live webcam; "Screenshot" reveals upload.
    def _on_mode_change(stored, mode):
        is_photo = (mode == "photo")
        if is_photo:
            # Photo mode: go straight to a clean live-camera view, ready to
            # capture. Do NOT re-run screen detection ("scan") on any previously
            # loaded image — the user switched here to take a NEW shot.
            return (
                gr.update(visible=False),   # hide upload
                gr.update(visible=True),    # show webcam (live view)
                None,                        # clear preview
                _empty_html(),               # clear results
            )
        # Screenshot mode: restore upload and re-classify the stored image.
        preview, html = reprocess(stored, mode)
        return (
            gr.update(visible=True),        # show upload
            gr.update(visible=False),       # hide webcam
            preview,
            html,
        )

    mode.change(
        fn=_on_mode_change,
        inputs=[img_state, mode],
        outputs=[img_input, cam_input, img_preview, output],
    ).then(fn=None, js=START_WEBCAM_JS)

if __name__ == "__main__":
    dark_theme = gr.themes.Base(
        primary_hue="cyan",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#000000",
        body_text_color="#e2e8f0",
        block_background_fill="transparent",
        block_border_color="transparent",
        block_border_width="0px",
        panel_background_fill="transparent",
        panel_border_color="transparent",
        button_primary_background_fill="linear-gradient(90deg,#06b6d4,#3b82f6)",
        button_primary_text_color="#000000",
        button_primary_background_fill_hover="linear-gradient(90deg,#22d3ee,#60a5fa)",
    )
    demo.launch(
        share=True,
        css=CSS,
        theme=dark_theme,
    )
