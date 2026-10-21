"""
Epithelium mask annotation tool (MedSAM-assisted).

Draw a box over the epithelium with the plot's box-select tool; MedSAM
proposes the outline inside the box. Add more boxes, add or erase plain
rectangles, then save. Masks are saved at full image resolution.

The images to annotate come from
annotations/epithelium/annotation_set.csv
(python -m scripts.segmentation.select_epithelium_set).

Run from the project root (local only):
    streamlit run app/epithelium_annotator.py --server.address localhost

MedSAM's slow step (the image embedding, ~4.5 min per image on a laptop
CPU) is precomputed on a Kaggle GPU and saved in
annotations/epithelium/embeddings/ (notebooks/medsam_embeddings_kaggle.ipynb).
With those files, each box takes a few seconds.
"""

import base64
import io
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

from oral_cancer.config import DATA_ROOT, OUTPUT_ROOT, relative, resolve
from oral_cancer.tissue import tissue_mask


# ============================================================
# CONFIGURATION
# ============================================================

ANN_DIR = OUTPUT_ROOT / "annotations" / "epithelium"
SET_CSV = ANN_DIR / "annotation_set.csv"
LOG_CSV = ANN_DIR / "annotations.csv"
MASK_DIR = ANN_DIR / "masks"
EMB_DIR = ANN_DIR / "embeddings"   # precomputed on Kaggle (fast) or absent (CPU fallback)

LOCAL_MODEL = DATA_ROOT / "medsam_hf"
HUB_MODEL = "flaviagiammarino/medsam-vit-base"

DISPLAY_WIDTH = 740           # image shown at this width; boxes are scaled back
OVERLAY_COLOR = np.array([27, 175, 122], dtype=np.float32)   # green
OVERLAY_ALPHA = 0.45
MAX_UNDO = 20

CLASS_DISPLAY = {
    "OSCC": "OSCC",
    "Leukoplakia_with_dysplasia": "Leukoplakia with dysplasia",
    "Leukoplakia_without_dysplasia": "Leukoplakia without dysplasia",
    "Normal": "Normal",
}


# ============================================================
# MEDSAM
# ============================================================

@st.cache_resource(show_spinner="Loading MedSAM...")
def load_medsam():
    from transformers import SamModel, SamProcessor
    source = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else HUB_MODEL
    processor = SamProcessor.from_pretrained(source)
    model = SamModel.from_pretrained(source).eval()
    return processor, model


def image_embedding(image):
    """Heavy step, once per image.

    Uses the precomputed embedding (scripts/segmentation/precompute_embeddings.py,
    run on a Kaggle GPU) when available; otherwise computes it on the CPU
    (several minutes). Cached in the session either way.
    """
    key = ("embedding", st.session_state.current)
    if key not in st.session_state:
        precomputed = EMB_DIR / f"{Path(st.session_state.current).stem}.pt"
        if precomputed.exists():
            st.session_state[key] = torch.load(precomputed, weights_only=True).float()
        else:
            processor, model = load_medsam()
            with torch.no_grad():
                pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
                st.session_state[key] = model.get_image_embeddings(pixel_values)
    return st.session_state[key]


def medsam_mask(image, box):
    """Binary mask (H, W) inside one box [x0, y0, x1, y1] in full-resolution pixels."""
    processor, model = load_medsam()
    embedding = image_embedding(image)
    inputs = processor(images=image, input_boxes=[[box]], return_tensors="pt")
    inputs.pop("pixel_values")
    with torch.no_grad():
        out = model(**inputs, image_embeddings=embedding, multimask_output=False)
    masks = processor.image_processor.post_process_masks(
        out.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )
    return masks[0][0, 0].numpy().astype(bool)


# ============================================================
# STATE HELPERS
# ============================================================

def mask_path(row):
    return MASK_DIR / f"{Path(row['image_path']).stem}.png"


def load_log():
    if LOG_CSV.exists():
        return pd.read_csv(LOG_CSV)
    return pd.DataFrame(columns=["image_path", "mask_path", "status", "epithelium_fraction",
                                 "edits", "annotator", "saved_at"])


def set_mask(new_mask):
    hist = st.session_state.setdefault("history", [])
    hist.append(st.session_state.mask.copy())
    del hist[:-MAX_UNDO]
    st.session_state.mask = new_mask
    st.session_state.edits += 1


def open_image(index, rows):
    """Load image index into the session (and its saved mask, if any)."""
    row = rows.iloc[index]
    st.session_state.current = row["image_path"]
    st.session_state.index = index
    with Image.open(resolve(row["image_path"])) as img:
        image = img.convert("RGB")
    st.session_state.image = image
    saved = mask_path(row)
    if saved.exists():
        st.session_state.mask = np.asarray(Image.open(saved)) > 127
    else:
        st.session_state.mask = np.zeros((image.height, image.width), dtype=bool)
    st.session_state.history = []
    st.session_state.edits = 0
    st.session_state.box = None
    st.session_state.plot_key = st.session_state.get("plot_key", 0) + 1


def overlay(image, mask, scale):
    small = image.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
    rgb = np.asarray(small).astype(np.float32)
    m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(small.size, Image.NEAREST)) > 0
    rgb[m] = (1 - OVERLAY_ALPHA) * rgb[m] + OVERLAY_ALPHA * OVERLAY_COLOR
    return rgb.astype(np.uint8)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(page_title="Epithelium annotator", layout="wide")

if not SET_CSV.exists():
    st.error("No annotation set. Run `python -m scripts.segmentation.select_epithelium_set` first.")
    st.stop()

rows = pd.read_csv(SET_CSV)
log = load_log()
done = set(log.loc[log["status"].isin(["done", "no_epithelium"]), "image_path"])

if "index" not in st.session_state:
    first_open = next((i for i, p in enumerate(rows["image_path"]) if p not in done), 0)
    open_image(first_open, rows)

row = rows.iloc[st.session_state.index]
image = st.session_state.image
scale = DISPLAY_WIDTH / image.width

# ------------------------------------------------------------
# Sidebar: progress, navigation, instructions
# ------------------------------------------------------------
with st.sidebar:
    st.header("Epithelium annotator")
    st.progress(len(done) / len(rows), text=f"{len(done)} of {len(rows)} images saved")
    annotator = st.text_input("Annotator name", value=st.session_state.get("annotator", ""))
    st.session_state.annotator = annotator

    labels = [f"{'✓ ' if p in done else ''}{i + 1}. {Path(p).name}" for i, p in enumerate(rows["image_path"])]
    choice = st.selectbox("Image", range(len(rows)), index=st.session_state.index,
                          format_func=lambda i: labels[i])
    if choice != st.session_state.index:
        open_image(choice, rows)
        st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("◀ Previous", disabled=st.session_state.index == 0, use_container_width=True):
        open_image(st.session_state.index - 1, rows)
        st.rerun()
    if c2.button("Next ▶", disabled=st.session_state.index == len(rows) - 1, use_container_width=True):
        open_image(st.session_state.index + 1, rows)
        st.rerun()

    st.markdown(
        "**How to annotate**\n"
        "1. Pick the **box-select** tool (top right of the image) and drag a box over the epithelium.\n"
        "2. **Add with MedSAM** to outline it, or add/erase a plain rectangle.\n"
        "3. Repeat until the green area covers the epithelium only.\n"
        "4. **Save mask**. If there is no epithelium, use **Save: no epithelium**.\n\n"
        "**Epithelium** = the surface cell layer (and its dysplastic or tumour cells). "
        "Leave out connective tissue, keratin flakes and glass."
    )

# ------------------------------------------------------------
# Main: image with mask overlay and box selection
# ------------------------------------------------------------
st.subheader(f"{st.session_state.index + 1}/{len(rows)} · {Path(row['image_path']).name}")
st.caption(
    f"{CLASS_DISPLAY[row['class']]} · {row['source']} · {row['estimated_zoom']} · fold {row['fold']} · "
    f"epithelium marked: {st.session_state.mask.mean():.1%}"
)

shown = overlay(image, st.session_state.mask, scale)
h, w = shown.shape[:2]
# Send the picture as a compressed JPEG (raw pixel arrays make Plotly very slow)
buf = io.BytesIO()
Image.fromarray(shown).save(buf, format="JPEG", quality=85)
src = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
fig = go.Figure(go.Image(source=src))
# Invisible grid of points: box selection needs selectable points on the plot
gx, gy = np.meshgrid(np.linspace(0, w - 1, 40), np.linspace(0, h - 1, 30))
fig.add_trace(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode="markers",
                         marker=dict(size=1, opacity=0), hoverinfo="skip", showlegend=False))
box = st.session_state.box
if box is not None:
    x0, y0, x1, y1 = (v * scale for v in box)
    fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="#eda100", width=2))
fig.update_layout(dragmode="select", margin=dict(l=0, r=0, t=0, b=0), width=w, height=h,
                  xaxis=dict(visible=False, range=[0, w]), yaxis=dict(visible=False, range=[h, 0]))

event = st.plotly_chart(fig, on_select="rerun", selection_mode="box", width="content",
                        key=f"plot-{st.session_state.plot_key}", config={"displaylogo": False})

selection = event.get("selection", {}) if event else {}
if selection and selection.get("box"):
    b = selection["box"][-1]
    xs, ys = b["x"], b["y"]
    new_box = [
        max(0, min(xs) / scale), max(0, min(ys) / scale),
        min(image.width - 1, max(xs) / scale), min(image.height - 1, max(ys) / scale),
    ]
    if new_box != st.session_state.box:
        st.session_state.box = new_box
        st.rerun()

# ------------------------------------------------------------
# Actions
# ------------------------------------------------------------
box = st.session_state.box
cols = st.columns(4)

if cols[0].button("Add with MedSAM", type="primary", disabled=box is None, use_container_width=True):
    with st.spinner("MedSAM is outlining..."):
        set_mask(st.session_state.mask | medsam_mask(image, box))
    st.session_state.box = None
    st.session_state.plot_key += 1
    st.rerun()

if cols[1].button("Add rectangle", disabled=box is None, use_container_width=True):
    m = st.session_state.mask.copy()
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    m[y0:y1 + 1, x0:x1 + 1] = True
    set_mask(m)
    st.session_state.box = None
    st.session_state.plot_key += 1
    st.rerun()

if cols[2].button("Erase rectangle", disabled=box is None, use_container_width=True):
    m = st.session_state.mask.copy()
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    m[y0:y1 + 1, x0:x1 + 1] = False
    set_mask(m)
    st.session_state.box = None
    st.session_state.plot_key += 1
    st.rerun()

if cols[3].button("Undo", disabled=not st.session_state.get("history"), use_container_width=True):
    st.session_state.mask = st.session_state.history.pop()
    st.session_state.plot_key += 1
    st.rerun()

cols = st.columns(4)

if cols[0].button("Mark all tissue", use_container_width=True,
                  help="For images that are entirely epithelium (common at 40x)"):
    set_mask(tissue_mask(image))
    st.session_state.plot_key += 1
    st.rerun()

if cols[1].button("Clear mask", use_container_width=True):
    set_mask(np.zeros_like(st.session_state.mask))
    st.session_state.plot_key += 1
    st.rerun()


def save(status):
    if not st.session_state.annotator.strip():
        st.warning("Enter your name in the sidebar before saving.")
        return
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    out = mask_path(row)
    Image.fromarray(st.session_state.mask.astype(np.uint8) * 255).save(out)
    entry = {
        "image_path": row["image_path"],
        "mask_path": relative(out),
        "status": status,
        "epithelium_fraction": round(float(st.session_state.mask.mean()), 4),
        "edits": st.session_state.edits,
        "annotator": st.session_state.annotator.strip(),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    current = load_log()
    current = current[current["image_path"] != row["image_path"]]
    pd.concat([current, pd.DataFrame([entry])], ignore_index=True).to_csv(LOG_CSV, index=False)
    st.toast(f"Saved {out.name}")
    if st.session_state.index < len(rows) - 1:
        open_image(st.session_state.index + 1, rows)
    st.rerun()


if cols[2].button("Save mask", type="primary", disabled=not st.session_state.mask.any(),
                  use_container_width=True):
    save("done")

if cols[3].button("Save: no epithelium", use_container_width=True):
    st.session_state.mask = np.zeros_like(st.session_state.mask)
    save("no_epithelium")

if box is not None:
    st.caption(f"Selected box: x {box[0]:.0f}–{box[2]:.0f}, y {box[1]:.0f}–{box[3]:.0f} (full-resolution pixels)")
else:
    st.caption("Draw a box with the box-select tool to enable the box actions.")
