"""
Generate the figures for the draft paper (docs/paper/figures/).

The README figures in docs/images/ are tall grids meant for a web page.
A two-column paper needs compact versions, and an architecture figure
drawn from real data (input images, masks, tiles, predictions) rather
than plain text boxes. Every image and number in these figures comes
from the project's own data and runs:

    architecture.png        both pipelines, illustrated with real images
    dataset_samples.jpg     whole images and one patch per class
    medsam_examples.jpg     box prompt and MedSAM mask per class
    epithelium_examples.jpg held-out whole images: image / annotation / U-Net
    gradcam_examples.jpg    patch and Grad-CAM heatmap per class
    prediction_example.jpg, confusion_matrix.png, training_curves.png,
    roi_by_source.png       copied from docs/images/

Usage:
    python -m scripts.reporting.make_paper_figures
"""

import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image

from oral_cancer.config import OUTPUT_ROOT, PROJECT_DIR, resolve
from oral_cancer.model import CLASSES_4


# ============================================================
# CONFIGURATION
# ============================================================

OUT_DIR = PROJECT_DIR / "docs" / "paper" / "figures"
README_IMAGES = PROJECT_DIR / "docs" / "images"
RUNS = OUTPUT_ROOT / "runs"

MAIN_RUN = "convnext_tiny_4class_224px"
ROW_A_WHOLE = "dataset_1_ndb_ufes/whole_images/extracted/images/0091.png"
ROW_B_WHOLE = "dataset_1_ndb_ufes/whole_images/extracted/images/0158.png"
EXAMPLE_PATCH = "preprocessed_normalized/test/OSCC/p3209.png"   # input for the feature maps and Grad-CAM

SHORT = {"OSCC": "OSCC", "Leukoplakia_with_dysplasia": "With dysplasia",
         "Leukoplakia_without_dysplasia": "Without dysplasia", "Normal": "Normal"}
BAR_LABEL = {"OSCC": "OSCC", "Leukoplakia_with_dysplasia": "Dysplasia",
             "Leukoplakia_without_dysplasia": "No dysplasia", "Normal": "Normal"}
CLASS_COLOURS = ["#c0392b", "#e67e22", "#f1c40f", "#2e86c1"]
EPI_GREEN = np.array([27, 175, 122])
SEG_FILL, CLS_FILL, DATA_FILL = "#e3f4ea", "#e4ecfa", "#f4f4f2"
EDGE = "#4a4a4a"

plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})


# ============================================================
# HELPERS
# ============================================================

def load_rgb(path, reduce=1):
    with Image.open(resolve(path)) as im:
        im = im.convert("RGB")
        return np.asarray(im.reduce(reduce) if reduce > 1 else im)


def overlay(image, mask, alpha=0.45):
    out = image.astype(np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * EPI_GREEN
    return out.astype(np.uint8)


def show(ax, img, x, y, w, h, edge=EDGE, lw=0.6):
    ax.imshow(img, extent=(x, x + w, y, y + h), zorder=3, interpolation="lanczos")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=edge, lw=lw, zorder=4))


def arrow(ax, p, q, style="-|>", ls="-", color="#333333", lw=1.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, lw=lw, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=5))


def rbox(ax, x, y, w, h, text, fc, fs=6.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=EDGE, lw=0.7, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, zorder=4)


def step(ax, x, y, n, label):
    ax.add_patch(plt.Circle((x, y), 0.17, fc="#333333", ec="none", zorder=6))
    ax.text(x, y, str(n), color="white", ha="center", va="center", fontsize=6.5, weight="bold", zorder=7)
    ax.text(x + 0.26, y, label, ha="left", va="center", fontsize=6.8, weight="bold", zorder=7)


def lighten(hex_colour, amount):
    rgb = np.array(matplotlib.colors.to_rgb(hex_colour))
    rgb = rgb + (1 - rgb) * amount if amount > 0 else rgb * (1 + amount)
    return tuple(np.clip(rgb, 0, 1))


def prob_bars(ax, x, y, w, probs, title=None):
    """Bars start at x; class labels sit just left of x."""
    if title:
        ax.text(x - 0.75, y + 1.32, title, fontsize=6.3, weight="bold", va="bottom")
    for i, (cls, p) in enumerate(zip(CLASSES_4, probs)):
        yy = y + 1.0 - i * 0.33
        ax.add_patch(Rectangle((x, yy), w, 0.22, fc="#eeeeee", ec="none", zorder=3))
        ax.add_patch(Rectangle((x, yy), w * p, 0.22, fc=CLASS_COLOURS[i], ec="none", zorder=4))
        ax.text(x - 0.05, yy + 0.11, BAR_LABEL[cls], ha="right", va="center", fontsize=5.6)
        ax.text(x + w + 0.05, yy + 0.11, f"{p:.2f}", ha="left", va="center", fontsize=5.6)


def save(fig, name, **kwargs):
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03, **kwargs)
    plt.close(fig)
    print(f"  {path.relative_to(PROJECT_DIR)}  ({path.stat().st_size // 1024} KB)")


def crop_panels(path, n_rows, n_cols, background=245):
    """Cut a grid figure (e.g. docs/images/*.jpg) back into its image panels."""
    img = np.asarray(Image.open(path).convert("RGB"))
    dark = img.mean(axis=2) < background

    def runs(profile, n):
        # Contiguous stretches that are mostly image (not page background);
        # titles and labels are short stretches, so keep the n longest
        on = np.concatenate([[False], profile > 0.5, [False]])
        starts = np.flatnonzero(~on[:-1] & on[1:])
        ends = np.flatnonzero(on[:-1] & ~on[1:])
        spans = sorted(zip(starts, ends), key=lambda s: s[1] - s[0], reverse=True)[:n]
        return sorted(spans)

    rows = runs(dark.mean(axis=1), n_rows)
    cols = runs(dark.mean(axis=0), n_cols)
    return [[img[r0:r1, c0:c1] for c0, c1 in cols] for r0, r1 in rows]


# ============================================================
# 1. ARCHITECTURE
# ============================================================

def example_prediction():
    """Probabilities and Grad-CAM of the main model for the example patch."""
    from oral_cancer.predictor import Predictor

    norm = pd.read_csv(resolve("splits/normalized_dataset.csv"))
    raw_path = norm.loc[norm["image_path"] == EXAMPLE_PATCH, "original_path"].iloc[0]
    with Image.open(resolve(raw_path)) as im:
        result = Predictor(RUNS / MAIN_RUN, device="cpu").predict(im)
    probs = [result.probabilities[c] for c in CLASSES_4]
    return raw_path, probs, np.asarray(result.overlay)


def class_patches():
    norm = pd.read_csv(resolve("splits/normalized_dataset.csv"))
    test = norm[norm["split"] == "test"]
    picks = {"OSCC": EXAMPLE_PATCH}
    for cls in CLASSES_4[1:]:
        picks[cls] = test[test["class"] == cls]["image_path"].iloc[3]
    return picks


def stage_feature_maps(patch_path, n=3):
    """Real ConvNeXt-Tiny activations for one patch: the n strongest channels of each stage."""
    import torch
    from oral_cancer.model import build_model, build_transforms

    model = build_model(len(CLASSES_4), pretrained=False)
    model.load_state_dict(torch.load(RUNS / MAIN_RUN / "best_model.pt", map_location="cpu"))
    model.eval()
    outputs = {}
    for i in (1, 3, 5, 7):  # stage outputs in torchvision's convnext_tiny.features
        model.features[i].register_forward_hook(lambda m, inp, out, i=i: outputs.__setitem__(i, out[0]))
    _, transform = build_transforms(224, False)
    with Image.open(resolve(patch_path)) as im:
        with torch.no_grad():
            model(transform(im.convert("RGB")).unsqueeze(0))

    stages = []
    for i in (1, 3, 5, 7):
        fmap = outputs[i].numpy()
        top = np.argsort(fmap.mean(axis=(1, 2)))[::-1][:n]
        maps = []
        for c in top:
            m = fmap[c]
            maps.append((m - m.min()) / (m.max() - m.min() + 1e-8))
        stages.append((maps, fmap.shape))
    return stages


def draw_feature_stack(ax, maps, shape, x, y_mid, size, title):
    """A small stack of real feature maps with its true size underneath."""
    off = 0.09
    for k in reversed(range(len(maps))):
        x0, y0 = x + k * off, y_mid - size / 2 + k * off
        ax.imshow(maps[k], cmap="magma", extent=(x0, x0 + size, y0, y0 + size), zorder=3 + (len(maps) - k) * 0.01,
                  interpolation="nearest")
        ax.add_patch(Rectangle((x0, y0), size, size, fill=False, ec=EDGE, lw=0.4, zorder=3.5))
    c, h, w = shape
    ax.text(x + size / 2 + off, y_mid - size / 2 - 0.08, f"{title}\n{h}x{w}x{c}", ha="center", va="top", fontsize=4.7)


def class_outputs():
    """One correctly classified test patch per class (median confidence), from the main run."""
    preds = pd.read_csv(RUNS / MAIN_RUN / "test_predictions.csv")
    rows = []
    for cls in CLASSES_4:
        ok = preds[(preds["class"] == cls) & (preds["predicted"] == cls)].sort_values(f"prob_{cls}")
        rows.append(ok.iloc[len(ok) // 2])
    return rows


def draw_unet(ax, x0, y0):
    """Encoder going down, decoder going up, skip connections across."""
    levels = [(0.12, 1.30), (0.16, 1.00), (0.20, 0.72), (0.24, 0.48)]  # (width, height) per level
    enc_x, dec_x, top = x0, x0 + 1.75, y0 + 1.75
    centres = []
    for i, (w, h) in enumerate(levels):
        yc = top - i * 0.42
        ax.add_patch(Rectangle((enc_x + i * 0.18, yc - h / 2), w, h, fc="#7fc9a0", ec=EDGE, lw=0.5, zorder=3))
        ax.add_patch(Rectangle((dec_x - i * 0.18, yc - h / 2), w, h, fc="#b9e2c9", ec=EDGE, lw=0.5, zorder=3))
        centres.append((enc_x + i * 0.18 + w, dec_x - i * 0.18, yc))
    bx, bw, bh, by = enc_x + 0.80, 0.32, 0.36, top - 4 * 0.42
    ax.add_patch(Rectangle((bx, by - bh / 2), bw, bh, fc="#3d9b6a", ec=EDGE, lw=0.5, zorder=3))
    for (ex, dx, yc) in centres:
        arrow(ax, (ex + 0.02, yc), (dx - 0.02, yc), style="-|>", ls=(0, (2, 1.5)), lw=0.6, color="#3d9b6a")
    arrow(ax, (centres[-1][0] - 0.1, top - 3 * 0.42 - 0.26), (bx + 0.05, by + bh / 2), lw=0.6)
    arrow(ax, (bx + bw - 0.05, by + bh / 2), (centres[-1][1] + 0.12, top - 3 * 0.42 - 0.26), lw=0.6)
    for i, ch in enumerate([64, 64, 128, 256]):
        w, h = levels[i]
        ax.text(enc_x + i * 0.18 + w / 2, top - i * 0.42 + h / 2 + 0.03, str(ch), ha="center", va="bottom", fontsize=4.6)
    ax.text(bx + bw / 2, by, "512", ha="center", va="center", fontsize=4.6, color="white", zorder=4)
    ax.text(enc_x - 0.05, by - 0.05, "Encoder\n(ResNet-34)", ha="left", va="top", fontsize=5.6)
    ax.text(dec_x + 0.25, by - 0.05, "Decoder", ha="right", va="top", fontsize=5.6)
    ax.text((enc_x + dec_x) / 2 + 0.15, top + 0.2, "skip connections", ha="center", fontsize=5.4, color="#3d9b6a")


def fig_architecture():
    raw_path, probs_a, cam_overlay = example_prediction()  # noqa: F841 (probs shown per class below)
    patches = class_patches()

    fig = plt.figure(figsize=(8.4, 4.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 9.9)
    ax.axis("off")

    # Row backgrounds
    for y, h, title in [(5.15, 4.65, "(a) Patch pipeline: main four-class classifier"),
                        (0.08, 4.85, "(b) Whole-image pipeline: segmentation -> ROI -> classification")]:
        ax.add_patch(FancyBboxPatch((0.08, y), 16.84, h, boxstyle="round,pad=0,rounding_size=0.2",
                                    fc="#fbfbfa", ec="#bbbbbb", lw=0.8, zorder=0))
        ax.text(0.3, y + h - 0.22, title.replace("->", "→"), fontsize=7.6, weight="bold", va="top")

    # ---------------- Row (a) ----------------
    ya = 5.55
    whole = load_rgb(ROW_A_WHOLE, reduce=4)
    show(ax, whole, 0.35, ya + 0.7, 2.4, 1.8)
    for k in range(1, 4):
        ax.plot([0.35 + k * 0.6, 0.35 + k * 0.6], [ya + 0.7, ya + 2.5], color="white", lw=0.5, zorder=4)
    for k in range(1, 3):
        ax.plot([0.35, 2.75], [ya + 0.7 + k * 0.6, ya + 0.7 + k * 0.6], color="white", lw=0.5, zorder=4)
    step(ax, 0.45, ya + 3.15, 1, "Input")
    ax.text(1.55, ya + 0.55, "NDB-UFES / Rahman image\n(2048x1536) -> 512x512 patches", ha="center", va="top", fontsize=5.6)

    for i, cls in enumerate(CLASSES_4):  # cascade of class patches
        show(ax, load_rgb(patches[cls], reduce=4), 3.25 + i * 0.32, ya + 0.55 + (3 - i) * 0.32, 1.3, 1.3, lw=0.5)
    ax.text(4.35, ya + 0.38, "4 classes, 6,626 patches", ha="center", va="top", fontsize=5.6)
    arrow(ax, (2.85, ya + 1.6), (3.2, ya + 1.6))

    pre = pd.read_csv(resolve("splits/normalized_dataset.csv"))
    normal_row = pre[(pre["class"] == "Normal") & (pre["split"] == "test")].iloc[5]
    show(ax, load_rgb(normal_row["preprocessed_path"], reduce=4), 5.95, ya + 1.55, 1.05, 1.05)
    show(ax, load_rgb(normal_row["image_path"], reduce=4), 5.95, ya + 0.25, 1.05, 1.05)
    ax.text(7.1, ya + 2.07, "tile +\nblur", fontsize=5.6, va="center")
    ax.text(7.1, ya + 0.77, "Macenko\nnormalised", fontsize=5.6, va="center")
    arrow(ax, (6.47, ya + 1.52), (6.47, ya + 1.33), lw=0.7)
    step(ax, 5.95, ya + 3.15, 2, "Preprocessing")
    arrow(ax, (5.55, ya + 1.6), (5.9, ya + 1.6))

    medsam = crop_panels(README_IMAGES / "segmentation_examples.jpg", 4, 3)
    show(ax, medsam[0][0], 8.15, ya + 1.4, 0.95, 0.95)
    ax.add_patch(Rectangle((8.2, ya + 1.45), 0.85, 0.85, fill=False, ec="#f1c40f", lw=1.0, zorder=5))
    rbox(ax, 9.25, ya + 1.75, 1.05, 0.6, "Image\nencoder ViT-B", SEG_FILL, fs=5.2)
    rbox(ax, 9.25, ya + 0.95, 1.05, 0.55, "Prompt\nencoder (box)", SEG_FILL, fs=5.2)
    rbox(ax, 10.5, ya + 1.3, 0.8, 0.7, "Mask\ndecoder", SEG_FILL, fs=5.2)
    arrow(ax, (9.1, ya + 1.95), (9.25, ya + 2.02), lw=0.7)
    arrow(ax, (10.3, ya + 2.02), (10.5, ya + 1.8), lw=0.7)
    arrow(ax, (10.3, ya + 1.2), (10.5, ya + 1.45), lw=0.7)
    show(ax, medsam[0][1], 10.4, ya + 0.08, 0.95, 0.95)
    ax.text(9.65, ya + 0.55, "mask covers\n95-98% of patch", ha="center", va="center", fontsize=5.4)
    arrow(ax, (10.88, ya + 1.3), (10.88, ya + 1.06), lw=0.7)
    step(ax, 8.15, ya + 3.15, 3, "MedSAM (ROI)")
    arrow(ax, (7.75, ya + 1.6), (8.1, ya + 1.6))

    stages = stage_feature_maps(EXAMPLE_PATCH)
    ymid = ya + 1.95
    for (maps, shape), x, size, title in zip(stages, [11.45, 12.68, 13.72, 14.62], [0.9, 0.74, 0.58, 0.44],
                                             ["Stage 1 (x3)", "Stage 2 (x3)", "Stage 3 (x9)", "Stage 4 (x3)"]):
        draw_feature_stack(ax, maps, shape, x, ymid, size, title)
    for x0, x1 in [(12.55, 12.66), (13.6, 13.7), (14.48, 14.6)]:
        arrow(ax, (x0, ymid), (x1, ymid), lw=0.6)
    step(ax, 11.5, ya + 3.15, 4, "ConvNeXt-Tiny feature maps")
    arrow(ax, (11.35, ya + 1.6), (11.48, ya + 1.85))
    ax.text(12.6, ya + 0.55, "real activations of the trained model\nfor the OSCC patch on the left\n(3 strongest channels per stage)",
            ha="center", va="center", fontsize=4.7, color="#555555")

    show(ax, cam_overlay[::4, ::4], 14.05, ya + 0.1, 0.72, 0.72)
    ax.text(14.82, ya + 0.46, "Grad-CAM\n(stage 4)", fontsize=4.8, va="center")

    arrow(ax, (15.27, ymid), (15.45, ymid), lw=0.7)
    step(ax, 15.6, ya + 3.15, 5, "Output")
    for i, row in enumerate(class_outputs()):
        y = ya + 2.5 - i * 0.6
        show(ax, load_rgb(row["image_path"], reduce=8), 15.5, y, 0.44, 0.44, lw=0.4)
        cls = row["class"]
        ax.add_patch(Rectangle((15.98, y + 0.03), 0.06, 0.38, fc=CLASS_COLOURS[i], ec="none", zorder=4))
        ax.text(16.06, y + 0.22, f"{BAR_LABEL[cls]}\np = {row[f'prob_{cls}']:.2f}", va="center", fontsize=4.4)

    # ---------------- Row (b) ----------------
    yb = 0.45
    big = load_rgb(ROW_B_WHOLE)
    small = load_rgb(ROW_B_WHOLE, reduce=4)
    with Image.open(resolve("segmentation/whole_images/" + ROW_B_WHOLE.rsplit("/", 1)[-1])) as m:
        mask = np.asarray(m.convert("L").resize((small.shape[1], small.shape[0]), Image.NEAREST)) > 127
    show(ax, small, 0.35, yb + 0.85, 2.4, 1.8)
    step(ax, 0.45, yb + 3.45, 1, "Whole image")
    ax.text(1.55, yb + 0.7, "1,461 images,\npatient-level 5-fold split", ha="center", va="top", fontsize=5.6)
    arrow(ax, (2.85, yb + 1.75), (3.2, yb + 1.75))

    draw_unet(ax, 3.35, yb + 0.6)
    step(ax, 3.35, yb + 3.45, 2, "U-Net segmenter")
    arrow(ax, (5.85, yb + 1.75), (6.2, yb + 1.75))

    show(ax, overlay(small, mask), 6.3, yb + 0.85, 2.4, 1.8)
    step(ax, 6.3, yb + 3.45, 3, "Epithelium mask")
    ax.text(7.5, yb + 0.7, "green = epithelium\n(Dice 0.85 on held-out patients)", ha="center", va="top", fontsize=5.6)
    arrow(ax, (8.8, yb + 1.75), (9.15, yb + 1.75))

    tiles = pd.read_csv(resolve("splits/whole_image_tiles.csv"))
    tiles = tiles[tiles["parent_image"] == ROW_B_WHOLE]
    show(ax, small, 9.25, yb + 0.85, 2.4, 1.8)
    unit = 2.4 / (big.shape[1] / 512)
    for t in tiles.itertuples():
        x = 9.25 + t.tile_col * unit
        y = yb + 0.85 + 1.8 - (t.tile_row + 1) * unit
        colour = "#1baf7a" if t.in_roi else "#e67e22"
        ax.add_patch(Rectangle((x + 0.03, y + 0.03), unit - 0.06, unit - 0.06, fill=False, ec=colour,
                               lw=1.1, ls="-" if t.in_roi else (0, (2, 1)), zorder=5))
    step(ax, 9.25, yb + 3.45, 4, "512x512 tiles")
    ax.text(10.45, yb + 0.7, "solid green: ROI tile (>=50% epithelium)\ndashed orange: tissue tile only",
            ha="center", va="top", fontsize=5.4)
    arrow(ax, (11.75, yb + 1.75), (12.0, yb + 1.75))

    rbox(ax, 12.05, yb + 1.3, 1.65, 0.9, "ConvNeXt-Tiny\n(same architecture,\nretrained per fold;\none tile at a time)", CLS_FILL, fs=5.4)
    xb2 = 13.7
    step(ax, 12.05, yb + 3.45, 5, "Classifier")

    preds = pd.read_csv(RUNS / "roi_image_predictions.csv")
    row = preds[(preds["parent_image"] == ROW_B_WHOLE) & (preds["setting"] == "roi")].iloc[0]
    probs_b = [row[f"prob_{c}"] for c in CLASSES_4]
    prob_bars(ax, 15.6, yb + 1.25, 0.85, probs_b, "Mean over ROI tiles")
    ax.text(15.3, yb + 0.95, f"true class: {SHORT[row['class']]}", ha="center", va="top", fontsize=5.6)
    step(ax, 14.25, yb + 3.45, 6, "Image prediction")
    arrow(ax, (xb2 + 0.02, yb + 1.55), (14.4, yb + 1.85))

    save(fig, "architecture.png", dpi=300)


# ============================================================
# 2. DATASET SAMPLES
# ============================================================

def fig_dataset_samples():
    folds = pd.read_csv(resolve("splits/whole_image_folds.csv"))
    wholes = [
        (ROW_B_WHOLE, "NDB-UFES whole image (10x)"),
        (folds[folds["source"] == "Dataset2_Normal"]["image_path"].iloc[0], "Rahman: normal mucosa (100x)"),
        (folds[(folds["source"] == "Dataset3_Rahman_OSCC") & (folds["estimated_zoom"] == "10x")]["image_path"].iloc[2],
         "Rahman: OSCC (100x)"),
    ]
    patches = class_patches()

    fig = plt.figure(figsize=(7.0, 3.55))
    for i, (path, title) in enumerate(wholes):
        ax = fig.add_axes([0.005 + i * 0.335, 0.53, 0.32, 0.42])
        img = load_rgb(path, reduce=4)
        if img.shape[0] > img.shape[1]:
            img = np.rot90(img)
        ax.imshow(img)
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    for i, cls in enumerate(CLASSES_4):
        ax = fig.add_axes([0.03 + i * 0.245, 0.04, 0.2, 0.39])
        ax.imshow(load_rgb(patches[cls], reduce=2))
        ax.set_title(f"{SHORT[cls]} patch", fontsize=7)
        ax.axis("off")
    save(fig, "dataset_samples.jpg", dpi=220, pil_kwargs={"quality": 88})


# ============================================================
# 3. SEGMENTATION OUTPUTS
# ============================================================

def fig_medsam():
    panels = crop_panels(README_IMAGES / "segmentation_examples.jpg", 4, 3)
    fig, axes = plt.subplots(2, 4, figsize=(3.45, 1.95))
    for j, cls in enumerate(CLASSES_4):
        for i in range(2):
            axes[i, j].imshow(panels[j][i])
            axes[i, j].axis("off")
        axes[0, j].set_title(SHORT[cls], fontsize=6)
    axes[0, 0].text(-0.08, 0.5, "box prompt", transform=axes[0, 0].transAxes, rotation=90,
                    ha="right", va="center", fontsize=6)
    axes[1, 0].text(-0.08, 0.5, "MedSAM mask", transform=axes[1, 0].transAxes, rotation=90,
                    ha="right", va="center", fontsize=6)
    plt.subplots_adjust(wspace=0.04, hspace=0.06, left=0.06, right=1, top=0.9, bottom=0)
    save(fig, "medsam_examples.jpg", dpi=300, pil_kwargs={"quality": 88})


def fig_epithelium():
    per_image = pd.read_csv(RUNS / "segmenter" / "test_per_image.csv")
    log = pd.read_csv(PROJECT_DIR / "annotations" / "epithelium" / "annotations.csv")
    per_image = per_image.merge(log[["image_path", "mask_path"]], on="image_path")
    picks = []
    for _, g in per_image[per_image["zoom"] == "10x"].groupby(["source", "class"]):
        picks.append(g.iloc[(g["segmenter_dice"] - g["segmenter_dice"].median()).abs().argsort().iloc[0]])

    fig, axes = plt.subplots(3, len(picks), figsize=(7.0, 3.6))
    for j, row in enumerate(picks):
        img = load_rgb(row["image_path"], reduce=4)
        size = (img.shape[1], img.shape[0])
        with Image.open(PROJECT_DIR / row["mask_path"]) as m:
            truth = np.asarray(m.convert("L").resize(size, Image.NEAREST)) > 127
        stem = row["image_path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        with Image.open(resolve(f"segmentation/whole_images/{stem}.png")) as m:
            pred = np.asarray(m.convert("L").resize(size, Image.NEAREST)) > 127
        source = "NDB-UFES" if row["source"] == "NDB-UFES" else "Rahman"
        for i, panel in enumerate([img, overlay(img, truth), overlay(img, pred)]):
            axes[i, j].imshow(panel)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        axes[0, j].set_title(f"{SHORT[row['class']]} ({source})", fontsize=6.5)
        axes[2, j].set_xlabel(f"Dice {row['segmenter_dice']:.2f}", fontsize=6.5)
    for i, label in enumerate(["Image", "Annotation", "U-Net output"]):
        axes[i, 0].set_ylabel(label, fontsize=6.5)
    plt.subplots_adjust(wspace=0.04, hspace=0.06, left=0.03, right=1, top=0.93, bottom=0.06)
    save(fig, "epithelium_examples.jpg", dpi=250, pil_kwargs={"quality": 88})


# ============================================================
# 4. CLASSIFICATION OUTPUTS
# ============================================================

def fig_gradcam():
    panels = crop_panels(README_IMAGES / "gradcam_examples.jpg", 4, 4)
    fig, axes = plt.subplots(2, 4, figsize=(3.45, 1.95))
    for j, cls in enumerate(CLASSES_4):
        axes[0, j].imshow(panels[j][0])
        axes[1, j].imshow(panels[j][1])
        axes[0, j].set_title(SHORT[cls], fontsize=6)
        for i in range(2):
            axes[i, j].axis("off")
    axes[0, 0].text(-0.08, 0.5, "patch", transform=axes[0, 0].transAxes, rotation=90, ha="right", va="center", fontsize=6)
    axes[1, 0].text(-0.08, 0.5, "Grad-CAM", transform=axes[1, 0].transAxes, rotation=90, ha="right", va="center", fontsize=6)
    plt.subplots_adjust(wspace=0.04, hspace=0.06, left=0.06, right=1, top=0.9, bottom=0)
    save(fig, "gradcam_examples.jpg", dpi=300, pil_kwargs={"quality": 88})


def copy_readme_figures():
    for name in ["prediction_example.jpg", "confusion_matrix.png", "training_curves.png", "roi_by_source.png"]:
        shutil.copy(README_IMAGES / name, OUT_DIR / name)
        print(f"  docs/paper/figures/{name}  (copied)")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Writing paper figures:")
    fig_architecture()
    fig_dataset_samples()
    fig_medsam()
    fig_epithelium()
    fig_gradcam()
    copy_readme_figures()
