"""
MedSAM segmentation + ROI extraction for the whole dataset.

For every image:
    1. Build a box prompt from the image itself (bounding box of the
       stained tissue). The prompt never uses the class label, so the
       same procedure is applied at training and at inference time.
    2. MedSAM predicts a mask inside that box.
    3. The ROI is the mask's bounding box (plus padding), cropped at
       native resolution. Resizing is left to the classifier.

Outputs (under OUTPUT_ROOT/segmentation/):
    masks/<split>/<class>/<name>.png      binary mask (0/255)
    rois/<split>/<class>/<name>.png       ROI crop
    visualizations/<class>/<name>.png     a few examples per class
    segmentation_results.csv              one row per image

Usage:
    python auto_segment.py --per-class 3          # quick test
    python auto_segment.py --csv segmentation_subset/subset.csv
    python auto_segment.py                        # full dataset
    python auto_segment.py --resume               # continue after a stop

NOTE: these masks are automatic MedSAM output, not ground truth.
Segmentation quality can only be measured against reference masks
(e.g. the ORCA dataset or manual annotations).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from config import DATA_ROOT, OUTPUT_ROOT, resolve, relative


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CSV = "splits/normalized_dataset.csv"

LOCAL_MODEL = DATA_ROOT / "medsam_hf"
HUB_MODEL = "flaviagiammarino/medsam-vit-base"

OUT_DIR = OUTPUT_ROOT / "segmentation"
RESULTS_CSV = OUT_DIR / "segmentation_results.csv"

# Tissue box prompt (label-agnostic)
LIGHT_INTENSITY = 240
OD_THRESHOLD = 0.15
BOX_MARGIN = 8              # pixels added around the tissue box
MIN_TISSUE_FRACTION = 0.05  # below this, prompt with the whole image

# ROI extraction
ROI_PADDING = 10

SAVE_EVERY = 50             # batches between results-CSV checkpoints
RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def tissue_box(rgb):
    """Bounding box [x0, y0, x1, y1] of stained tissue in the image."""
    h, w = rgb.shape[:2]

    od = -np.log((rgb.astype(np.float32) + 1) / LIGHT_INTENSITY)
    tissue = od.max(axis=2) > OD_THRESHOLD

    if tissue.mean() < MIN_TISSUE_FRACTION:
        return [0, 0, w - 1, h - 1]

    ys, xs = np.where(tissue)

    return [
        max(0, int(xs.min()) - BOX_MARGIN),
        max(0, int(ys.min()) - BOX_MARGIN),
        min(w - 1, int(xs.max()) + BOX_MARGIN),
        min(h - 1, int(ys.max()) + BOX_MARGIN),
    ]


def extract_roi(rgb, mask):
    """Crop the mask's bounding box (+ padding) at native resolution."""
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return rgb, "empty_mask_full_image"

    h, w = rgb.shape[:2]

    x0 = max(0, xs.min() - ROI_PADDING)
    y0 = max(0, ys.min() - ROI_PADDING)
    x1 = min(w, xs.max() + 1 + ROI_PADDING)
    y1 = min(h, ys.max() + 1 + ROI_PADDING)

    return rgb[y0:y1, x0:x1], "ok"


def output_path(kind, row):
    name = Path(row["image_path"]).stem + ".png"
    return OUT_DIR / kind / row["split"] / row["class"] / name


def save_visualization(rgb, box, mask, roi, iou, row):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    vis_path = OUT_DIR / "visualizations" / row["class"] / (Path(row["image_path"]).stem + ".png")
    vis_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    x0, y0, x1, y1 = box
    axes[0].add_patch(patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=2, edgecolor="yellow", facecolor="none"
    ))
    axes[0].set_title(f"{row['class']}\nbox prompt (tissue extent)")

    axes[1].imshow(rgb)
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask] = [1, 0, 0, 0.4]
    axes[1].imshow(overlay)
    axes[1].set_title(f"MedSAM mask ({mask.mean():.0%} of image)\nMedSAM predicted IoU: {iou:.2f}")

    axes[2].imshow(roi)
    axes[2].set_title(f"ROI crop {roi.shape[1]}x{roi.shape[0]}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(vis_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def load_model(device):
    from transformers import SamModel, SamProcessor

    source = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else HUB_MODEL
    print(f"Loading MedSAM from: {source}")

    processor = SamProcessor.from_pretrained(source)
    model = SamModel.from_pretrained(source).to(device).eval()

    return processor, model


# ============================================================
# MAIN
# ============================================================

def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("WARNING: no GPU found. This works but is slow; use Kaggle for full runs.")

    # --------------------------------------------------------
    # Select images
    # --------------------------------------------------------
    df = pd.read_csv(resolve(args.csv))

    if args.splits:
        df = df[df["split"].isin(args.splits)]

    if args.per_class:
        n = min(args.per_class, df["class"].value_counts().min())
        df = df.groupby("class").sample(n=n, random_state=RANDOM_SEED)

    df = df.reset_index(drop=True)

    done = pd.DataFrame()

    if args.resume and RESULTS_CSV.exists():
        done = pd.read_csv(RESULTS_CSV)
        df = df[~df["image_path"].isin(done["image_path"])].reset_index(drop=True)
        print(f"Resuming: {len(done)} already done")

    print(f"Images to segment: {len(df)}")

    if len(df) == 0:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    processor, model = load_model(device)

    # --------------------------------------------------------
    # Segment in batches
    # --------------------------------------------------------
    records = done.to_dict("records")
    vis_counts = {}

    batches = range(0, len(df), args.batch_size)

    for b, start in enumerate(tqdm(batches, desc="Segmenting")):

        rows = df.iloc[start:start + args.batch_size]

        images = [
            np.asarray(Image.open(resolve(p)).convert("RGB"))
            for p in rows["image_path"]
        ]
        boxes = [tissue_box(rgb) for rgb in images]

        inputs = processor(
            images=images,
            input_boxes=[[box] for box in boxes],
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        ious = outputs.iou_scores.cpu().numpy().reshape(len(images))

        for (_, row), rgb, box, mask_t, iou in zip(rows.iterrows(), images, boxes, masks, ious):

            mask = mask_t[0, 0].numpy().astype(bool)
            roi, roi_status = extract_roi(rgb, mask)

            mask_path = output_path("masks", row)
            roi_path = output_path("rois", row)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            roi_path.parent.mkdir(parents=True, exist_ok=True)

            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
            Image.fromarray(roi).save(roi_path)

            if vis_counts.get(row["class"], 0) < args.vis_per_class:
                save_visualization(rgb, box, mask, roi, float(iou), row)
                vis_counts[row["class"]] = vis_counts.get(row["class"], 0) + 1

            records.append({
                **row.to_dict(),
                "mask_path": relative(mask_path),
                "roi_path": relative(roi_path),
                "box_xmin": box[0],
                "box_ymin": box[1],
                "box_xmax": box[2],
                "box_ymax": box[3],
                "mask_fraction": float(mask.mean()),
                "medsam_iou_score": float(iou),
                "roi_width": roi.shape[1],
                "roi_height": roi.shape[0],
                "roi_status": roi_status,
            })

        if (b + 1) % SAVE_EVERY == 0:
            pd.DataFrame(records).to_csv(RESULTS_CSV, index=False)

    results = pd.DataFrame(records)
    results.to_csv(RESULTS_CSV, index=False)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n========================================")
    print("SEGMENTATION COMPLETE")
    print("========================================")

    print(f"\nImages segmented: {len(results)}")
    print(f"Results CSV: {RESULTS_CSV}")

    print("\nPer class (median values):")
    print(
        results.groupby("class")[["mask_fraction", "medsam_iou_score", "roi_width", "roi_height"]]
        .median()
        .round(3)
        .to_string()
    )

    print("\nROI status:")
    print(results["roi_status"].value_counts().to_string())

    print("\nThese are automatic masks, not ground truth. Review the")
    print(f"visualizations in {OUT_DIR / 'visualizations'} before using them.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MedSAM segmentation and ROI extraction")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Input CSV (relative to the data root)")
    parser.add_argument("--splits", nargs="+", help="Only these splits, e.g. --splits train validation")
    parser.add_argument("--per-class", type=int, help="Random sample of N images per class (for testing)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vis-per-class", type=int, default=5, help="Visualizations saved per class")
    parser.add_argument("--resume", action="store_true", help="Skip images already in the results CSV")

    main(parser.parse_args())
