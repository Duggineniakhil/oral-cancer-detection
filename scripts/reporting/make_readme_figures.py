"""
Generate the figures used in README.md (written to docs/images/).

Run from the project root after the data pipeline, training and
Grad-CAM steps:

    python -m scripts.reporting.make_readme_figures

Inputs:
    splits/preprocessed_dataset.csv, splits/normalized_dataset.csv
    runs/<run>/history.csv, test_predictions.csv, metrics.json
    runs/convnext_tiny_4class_224px/gradcam/border_check_summary.csv

The MedSAM segmentation numbers come from the Kaggle run
(notebooks/medsam_segmentation_kaggle.ipynb) and are recorded below in
SEGMENTATION_SUMMARY, because the full segmentation output is not kept
locally.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

from oral_cancer.config import OUTPUT_ROOT, PROJECT_DIR, resolve
from oral_cancer.model import CLASSES_4, GradCAM, build_model, build_transforms


# ============================================================
# STYLE
# ============================================================

OUT_DIR = PROJECT_DIR / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = OUTPUT_ROOT / "runs"
MAIN_RUN = "convnext_tiny_4class_224px"

# Validated categorical palette (fixed order, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

CLASS_LABELS = {
    "OSCC": "OSCC",
    "Leukoplakia_with_dysplasia": "Leukoplakia\nwith dysplasia",
    "Leukoplakia_without_dysplasia": "Leukoplakia\nwithout dysplasia",
    "Normal": "Normal",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "text.color": TEXT_PRIMARY,
    "font.size": 10,
    "legend.frameon": False,
})

# From the Kaggle MedSAM run on all 6,626 patches (segmentation_results.csv):
# quantiles of the mask area as a fraction of the patch.
SEGMENTATION_SUMMARY = {
    "OSCC": {"q25": 0.933, "median": 0.983, "q75": 0.999, "mean_iou": 0.656},
    "Leukoplakia_with_dysplasia": {"q25": 0.902, "median": 0.964, "q75": 0.993, "mean_iou": 0.664},
    "Leukoplakia_without_dysplasia": {"q25": 0.873, "median": 0.951, "q75": 0.992, "mean_iou": 0.673},
    "Normal": {"q25": 0.867, "median": 0.947, "q75": 0.990, "mean_iou": 0.657},
}


def clean_axes(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name, photo=False):
    path = OUT_DIR / name
    if photo:
        fig.savefig(path, dpi=110, bbox_inches="tight", pil_kwargs={"quality": 85})
    else:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(PROJECT_DIR)}  ({path.stat().st_size // 1024} KB)")


def thumbnail(path, size=256):
    with Image.open(resolve(path)) as img:
        return np.asarray(img.convert("RGB").resize((size, size), Image.LANCZOS))


# ============================================================
# 1. DATASET SAMPLES
# ============================================================

def fig_dataset_samples(pre):
    rng = 7
    fig, axes = plt.subplots(4, 4, figsize=(9, 9.6))

    for r, cls in enumerate(CLASSES_4):
        rows = pre[pre["class"] == cls].sample(4, random_state=rng)
        for c, (_, row) in enumerate(rows.iterrows()):
            ax = axes[r, c]
            ax.imshow(thumbnail(row["image_path"]))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        n = (pre["class"] == cls).sum()
        axes[r, 0].set_ylabel(f"{CLASS_LABELS[cls]}\n(n = {n:,})", rotation=0,
                              ha="right", va="center", labelpad=12, color=TEXT_PRIMARY, fontsize=10)

    fig.suptitle("Dataset: one row per class (512×512 patches)", fontweight="bold", y=0.995)
    fig.text(0.5, 0.005, "OSCC and leukoplakia: NDB-UFES (CC BY 4.0).  Normal: tiles from Rahman et al. (CC BY 4.0).",
             ha="center", color=TEXT_SECONDARY, fontsize=8)
    plt.tight_layout(rect=[0, 0.01, 1, 0.985])
    save(fig, "dataset_samples.jpg", photo=True)


# ============================================================
# 2. PREPROCESSING: TILING + STAIN NORMALIZATION
# ============================================================

def fig_preprocessing(pre, norm):
    normal = pre[pre["class"] == "Normal"]

    # A source image where some tiles were discarded (shows the filter)
    counts = normal.groupby("original_path").size()
    source = counts[(counts >= 7) & (counts <= 9)].index.sort_values()[0]
    kept = normal[normal["original_path"] == source]

    with Image.open(resolve(source)) as img:
        full = img.convert("RGB")
    w, h = full.size
    scale = 0.25
    shown = np.asarray(full.resize((int(w * scale), int(h * scale))))

    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(2, 4, width_ratios=[2.2, 1, 1, 0.02], wspace=0.08, hspace=0.25)

    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(shown)
    kept_pos = set(zip(kept["tile_row"], kept["tile_col"]))
    t = 512 * scale
    for r in range(h // 512):
        for c in range(w // 512):
            ok = (r, c) in kept_pos
            ax.add_patch(mpatches.Rectangle(
                (c * t, r * t), t, t, fill=not ok,
                facecolor=(1, 1, 1, 0.8) if not ok else None,
                edgecolor="white", linewidth=1.2, hatch="//" if not ok else None))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("A. Normal image tiled into 512×512 patches", loc="left")
    ax.set_xlabel(f"{len(kept)}/{(h // 512) * (w // 512)} tiles kept\nhatched = discarded (<60% tissue)",
                  color=TEXT_SECONDARY, fontsize=9)

    # Before / after stain normalization
    ndb_row = norm[(norm["source"] == "NDB-UFES") & (norm["class"] == "OSCC")].sample(1, random_state=3).iloc[0]
    nor_row = norm[norm["image_path"].str.contains(Path_stem(kept.iloc[0]["image_path"]))].iloc[0]

    pairs = [("NDB-UFES (OSCC)", ndb_row), ("Normal dataset", nor_row)]
    for i, (label, row) in enumerate(pairs):
        for j, (col, title) in enumerate([("preprocessed_path", "before"), ("image_path", "after")]):
            ax = fig.add_subplot(gs[i, 1 + j])
            ax.imshow(thumbnail(row[col]))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{label}: {title}", fontsize=9, fontweight="normal", color=TEXT_SECONDARY)

    fig.text(0.535, 0.97, "B. Macenko stain normalization", fontweight="bold", fontsize=12)
    fig.text(0.535, 0.03, "Normal tiles are also blurred (σ = 2.5 px) to match NDB-UFES sharpness",
             color=TEXT_SECONDARY, fontsize=9)
    save(fig, "preprocessing.jpg", photo=True)


def Path_stem(path):
    from pathlib import PurePosixPath
    return PurePosixPath(path).stem


# ============================================================
# 3. SEGMENTATION: MASK COVERAGE
# ============================================================

def fig_segmentation_coverage():
    fig, ax = plt.subplots(figsize=(8, 3.4))
    classes = CLASSES_4
    y = np.arange(len(classes))[::-1]

    for yi, cls in zip(y, classes):
        s = SEGMENTATION_SUMMARY[cls]
        ax.plot([s["q25"] * 100, s["q75"] * 100], [yi, yi], color=SERIES[0], linewidth=6,
                solid_capstyle="round", alpha=0.35)
        ax.plot(s["median"] * 100, yi, "o", color=SERIES[0], markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(s["median"] * 100, yi - 0.32, f"{s['median']:.1%}", ha="center", va="top", fontsize=9)

    ax.set_yticks(y, [CLASS_LABELS[c].replace("\n", " ") for c in classes])
    ax.set_ylim(-0.8, len(classes) - 0.5)
    ax.set_xlim(80, 101)
    ax.set_xlabel("MedSAM mask area (% of patch)  ·  dot = median, bar = interquartile range")
    ax.set_title("MedSAM masks cover almost the entire patch (n = 6,626)", loc="left")
    clean_axes(ax, grid_axis="x")
    save(fig, "segmentation_mask_coverage.png")


# ============================================================
# 4. CLASSIFICATION: TRAINING CURVES
# ============================================================

RUN_LABELS = {
    "convnext_tiny_4class_224px": "4-class",
    "convnext_tiny_4class_gray_224px": "4-class, greyscale",
    "convnext_tiny_3class_224px": "3-class (NDB-UFES)",
}


def fig_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for color, (run, label) in zip(SERIES, RUN_LABELS.items()):
        hist = pd.read_csv(RUNS / run / "history.csv")
        ax.plot(hist["epoch"], hist["val_macro_f1"] * 100, color=color, linewidth=2, label=label)
        best = hist.loc[hist["val_macro_f1"].idxmax()]
        ax.plot(best["epoch"], best["val_macro_f1"] * 100, "o", color=color, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_title("Validation macro-F1 (dot = selected epoch)", loc="left")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro-F1 (%)")
    ax.legend(loc="lower right")
    clean_axes(ax)

    ax = axes[1]
    hist = pd.read_csv(RUNS / MAIN_RUN / "history.csv")
    ax.plot(hist["epoch"], hist["train_loss"], color=SERIES[0], linewidth=2, label="train")
    ax.plot(hist["epoch"], hist["val_loss"], color=SERIES[1], linewidth=2, label="validation")
    ax.set_title("Loss — 4-class run", loc="left")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy (label smoothing 0.1)")
    ax.legend(loc="upper right")
    clean_axes(ax)

    plt.tight_layout()
    save(fig, "training_curves.png")


# ============================================================
# 5. CLASSIFICATION: CONFUSION MATRIX
# ============================================================

def draw_confusion(ax, run, classes, title):
    preds = pd.read_csv(RUNS / run / "test_predictions.csv")
    cm = confusion_matrix(preds["class"], preds["predicted"], labels=classes)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    n = len(classes)

    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    labels = [CLASS_LABELS[c] for c in classes]
    ax.set_xticks(range(n), labels, fontsize=9)
    ax.set_yticks(range(n), labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{pct[i, j]:.1f}%\n({cm[i, j]})", ha="center", va="center", fontsize=9,
                    color="white" if pct[i, j] > 55 else TEXT_PRIMARY)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{title} (n = {cm.sum():,})", loc="left")


def fig_confusion_matrix():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [3, 4]})
    draw_confusion(axes[0], "convnext_tiny_3class_224px", CLASSES_4[:3], "3-class, NDB-UFES (primary)")
    draw_confusion(axes[1], MAIN_RUN, CLASSES_4, "4-class")
    fig.suptitle("Test confusion matrices", fontweight="bold", x=0.01, ha="left")
    plt.tight_layout()
    save(fig, "confusion_matrix.png")


# ============================================================
# 6. CLASSIFICATION: PER-CLASS F1, COLOUR VS GREYSCALE
# ============================================================

def fig_per_class_f1():
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(4)
    width = 0.36

    runs = [("convnext_tiny_4class_224px", "colour"), ("convnext_tiny_4class_gray_224px", "greyscale")]
    for k, (run, label) in enumerate(runs):
        preds = pd.read_csv(RUNS / run / "test_predictions.csv")
        f1 = f1_score(preds["class"], preds["predicted"], labels=CLASSES_4, average=None) * 100
        bars = ax.bar(x + (k - 0.5) * (width + 0.02), f1, width, color=SERIES[k], label=label)
        for b, v in zip(bars, f1):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=9)

    ax.set_xticks(x, [CLASS_LABELS[c] for c in CLASSES_4])
    ax.set_ylim(75, 102)
    ax.set_ylabel("Test F1 (%)")
    ax.set_title("Per-class test F1 — Normal stays at ~99% even without colour", loc="left")
    ax.legend(loc="upper left", ncol=2)
    clean_axes(ax)
    save(fig, "per_class_f1.png")


# ============================================================
# 6b. CLASSIFICATION: MULTI-SEED RESULTS
# ============================================================

MAIA_2024_BALANCED_ACC = 91.91  # DenseNet-121, NDB-UFES 3-class, same official folds


def fig_multiseed():
    per_run_csv = RUNS / "multiseed_per_run.csv"
    if not per_run_csv.exists():
        print("  (skipping multiseed figure: run summarize_seeds first)")
        return

    per_run = pd.read_csv(per_run_csv)
    rows = [
        ("convnext_tiny_3class_224px", "balanced_accuracy", "3-class, NDB-UFES (primary)"),
        ("convnext_tiny_4class_224px", "balanced_accuracy", "4-class"),
        ("convnext_tiny_4class_224px", "ndb_balanced_accuracy", "4-class, NDB-UFES patches only"),
        ("convnext_tiny_4class_gray_224px", "balanced_accuracy", "4-class, greyscale"),
    ]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    y = np.arange(len(rows))[::-1]

    for yi, (setting, metric, _) in zip(y, rows):
        vals = per_run.loc[per_run["setting"] == setting, metric].values * 100
        mean, sd = vals.mean(), vals.std(ddof=1)
        ax.plot([mean - sd, mean + sd], [yi, yi], color=SERIES[0], linewidth=6,
                solid_capstyle="round", alpha=0.3)
        ax.scatter(vals, [yi] * len(vals), s=40, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5,
                   zorder=3, label="single seed" if yi == y[0] else None)
        ax.plot(mean, yi, "D", color=TEXT_PRIMARY, markersize=7, zorder=4,
                label="mean ± SD" if yi == y[0] else None)
        ax.text(mean, yi - 0.34, f"{mean:.1f} ± {sd:.1f}%", ha="center", va="top", fontsize=9, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none"))

    ax.axvline(MAIA_2024_BALANCED_ACC, color=SERIES[1], linewidth=1.5, linestyle="--")
    ax.text(MAIA_2024_BALANCED_ACC + 0.1, y.max() + 0.42, "Maia et al. 2024, DenseNet-121, 3-class (91.9%)",
            fontsize=8, color=TEXT_SECONDARY)

    ax.set_yticks(y, [label for _, _, label in rows])
    ax.set_ylim(y.min() - 0.8, y.max() + 0.7)
    ax.set_xlim(86, 96)
    ax.set_xlabel("Test balanced accuracy (%)")
    ax.set_title("Balanced accuracy over 3 seeds", loc="left")
    ax.legend(loc="lower right")
    clean_axes(ax, grid_axis="x")
    save(fig, "multiseed_results.png")


# ============================================================
# 7. GRAD-CAM EXAMPLES
# ============================================================

def fig_gradcam():
    cfg = json.load(open(RUNS / MAIN_RUN / "config.json"))
    model = build_model(4, pretrained=False)
    model.load_state_dict(torch.load(RUNS / MAIN_RUN / "best_model.pt", map_location="cpu"))
    model.eval()
    cam_model = GradCAM(model, model.features[-1])
    _, eval_tf = build_transforms(cfg["img_size"], cfg["grayscale"])

    samples = pd.read_csv(RUNS / MAIN_RUN / "gradcam" / "gradcam_samples.csv")
    samples = samples[samples["kind"] == "correct"].groupby("true").head(2)

    fig, axes = plt.subplots(4, 4, figsize=(9, 9.6))
    for r, cls in enumerate(CLASSES_4):
        rows = samples[samples["true"] == cls]
        for k, (_, row) in enumerate(rows.iterrows()):
            with Image.open(resolve(row["image_path"])) as img:
                img = img.convert("RGB")
                tensor = eval_tf(img).unsqueeze(0)
                shown = np.asarray(img.resize((cfg["img_size"], cfg["img_size"])))
            cam, probs = cam_model(tensor)
            axes[r, 2 * k].imshow(shown)
            axes[r, 2 * k + 1].imshow(shown)
            axes[r, 2 * k + 1].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            axes[r, 2 * k + 1].set_title(f"p = {probs.max():.2f}", fontsize=9,
                                         fontweight="normal", color=TEXT_SECONDARY)
        axes[r, 0].set_ylabel(CLASS_LABELS[cls], rotation=0, ha="right", va="center",
                              labelpad=12, color=TEXT_PRIMARY)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle("Grad-CAM on correctly classified test patches (image | heatmap)", fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    save(fig, "gradcam_examples.jpg", photo=True)


# ============================================================
# 8. GRAD-CAM BORDER CHECK
# ============================================================

def fig_border_check():
    summary = pd.read_csv(RUNS / MAIN_RUN / "gradcam" / "border_check_summary.csv")
    order = [
        ("OSCC", "OSCC"),
        ("Leukoplakia_with_dysplasia", "Leukoplakia with dysplasia"),
        ("Leukoplakia_without_dysplasia", "Leukoplakia without dysplasia"),
        ("Normal (interior of source image)", "Normal: interior tiles"),
        ("Normal (edge of source image)", "Normal: edge tiles"),
    ]
    groups = [g for g, _ in order]

    # Before = model trained before sharpness matching (runs_before_blur/),
    # after = current model (runs/)
    series = [("before sharpness matching", RUNS.parent / "runs_before_blur"), ("after", RUNS)]
    series = [(label, root) for label, root in series
              if (root / MAIN_RUN / "gradcam" / "border_check_summary.csv").exists()]

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    y = np.arange(len(order))[::-1].astype(float)
    height = 0.8 / len(series)

    for k, (label, root) in enumerate(series):
        s = pd.read_csv(root / MAIN_RUN / "gradcam" / "border_check_summary.csv").set_index("group").loc[groups]
        vals = s["border_share_mean"].values * 100
        offset = (len(series) - 1) / 2 * height - k * height
        ax.barh(y + offset, vals, height=height * 0.9, color=SERIES[k], label=label)
        for yi, v in zip(y + offset, vals):
            ax.text(v + 0.8, yi, f"{v:.0f}%", va="center", fontsize=8.5)

    ax.axvline(49.0, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    ax.text(49.6, y.min() - 0.68, "even spread (49%)", ha="left", fontsize=8, color=TEXT_SECONDARY)
    ax.set_ylim(y.min() - 0.9, y.max() + 0.6)
    ax.set_yticks(y, [label for _, label in order])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of Grad-CAM heatmap on the patch border (%), all 1,049 test images")
    ax.set_title("Normal predictions still rely on patch borders after sharpness matching", loc="left")
    ax.legend(loc="upper right")
    clean_axes(ax, grid_axis="x")
    save(fig, "gradcam_border_check.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    pre = pd.read_csv(resolve("splits/preprocessed_dataset.csv"))
    norm = pd.read_csv(resolve("splits/normalized_dataset.csv"))

    print("Writing figures:")
    fig_dataset_samples(pre)
    fig_preprocessing(pre, norm)
    fig_segmentation_coverage()
    fig_training_curves()
    fig_confusion_matrix()
    fig_per_class_f1()
    fig_multiseed()
    fig_gradcam()
    fig_border_check()
