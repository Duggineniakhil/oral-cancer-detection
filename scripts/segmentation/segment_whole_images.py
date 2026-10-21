"""
Predict an epithelium mask for every whole image in the patient-level
folds, using the segmenter trained by train_segmenter.py.

Each image is segmented by the model of the round whose TEST fold is
the image's own fold. That model never trained on the image's patient
(nor, for the 50 annotated images, on the image's own mask), so the
masks are as honest as they would be on new data.

Usage:
    python -m scripts.segmentation.segment_whole_images
    python -m scripts.segmentation.segment_whole_images --smoke-test   # few images, round-1 smoke model

Outputs:
    OUTPUT_ROOT/segmentation/whole_images/<stem>.png   mask at half resolution (0/255)
    OUTPUT_ROOT/splits/whole_image_masks.csv           image_path, mask_path, fold, epithelium_fraction
"""

import argparse
import time

import pandas as pd
import torch
from PIL import Image

from oral_cancer.config import OUTPUT_ROOT, relative, resolve
from oral_cancer.segmenter import THRESHOLD, downscale, load_segmenter, predict


# ============================================================
# CONFIGURATION
# ============================================================

FOLDS_CSV = "splits/whole_image_folds.csv"
MASK_DIR = OUTPUT_ROOT / "segmentation" / "whole_images"
OUTPUT_CSV = OUTPUT_ROOT / "splits" / "whole_image_masks.csv"


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segmenter-dir", default="runs/segmenter")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(resolve(FOLDS_CSV))
    seg_dir = OUTPUT_ROOT / args.segmenter_dir

    if args.smoke_test:
        seg_dir = OUTPUT_ROOT / "runs" / "segmenter_smoke"
        df = df.groupby(["fold", "source"]).head(1)

    stems = df["image_path"].map(lambda p: p.rsplit("/", 1)[-1].rsplit(".", 1)[0])
    assert stems.is_unique, "image file names must be unique across sources"

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    records, start = [], time.time()
    for k, group in df.groupby("fold"):
        model_path = seg_dir / f"round{k}" / "best_model.pt"
        if args.smoke_test and not model_path.exists():
            model_path = seg_dir / "round1" / "best_model.pt"
        model, scale = load_segmenter(model_path, device)
        print(f"Fold {k}: {len(group)} images, model {relative(model_path)}")

        for path in group["image_path"]:
            with Image.open(resolve(path)) as im:
                image = downscale(im, scale)
            mask = predict(model, image, device) > THRESHOLD

            out = MASK_DIR / f"{path.rsplit('/', 1)[-1].rsplit('.', 1)[0]}.png"
            Image.fromarray(mask.astype("uint8") * 255).save(out)
            records.append({"image_path": path, "mask_path": relative(out), "fold": k,
                            "epithelium_fraction": round(float(mask.mean()), 4)})

    out = pd.DataFrame(records)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSegmented {len(out)} images in {time.time() - start:.0f}s -> {relative(OUTPUT_CSV)}")
    summary = out.merge(df[["image_path", "source", "estimated_zoom"]], on="image_path")
    print("Mean epithelium fraction by source and zoom:")
    print(summary.groupby(["source", "estimated_zoom"])["epithelium_fraction"].mean().round(3).unstack().to_string())


if __name__ == "__main__":
    main()
