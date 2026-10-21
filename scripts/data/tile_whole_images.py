"""
Tile the whole images into 512x512 patches for the ROI experiment, and
write one train/validation/test CSV per round and setting.

Two settings are compared on exactly the same images and folds:

    all   every tile with >= 60% tissue (no segmentation; the same rule
          as the patch pipeline)
    roi   every tile with >= 50% predicted epithelium (segmentation ->
          ROI -> classification)

If an image has no tile that passes a rule, its best tile for that rule
is kept, so every image gets a prediction in both settings.

Tiles get the same treatment as the patch pipeline:
    - native 512x512 crops, never resized (cell size stays comparable)
    - Gaussian blur (sigma 2.5) on the Rahman images (Normal AND OSCC),
      which are ~5x sharper than NDB-UFES (see check_sharpness.py).
      Blurring by source, not by class, keeps sharpness from
      separating the classes.
    - Macenko normalization to the NDB-UFES reference

Rounds follow the whole-image protocol (patient-level folds):
    round k: test = fold k, validation = fold (k mod 5) + 1, train = rest

Usage:
    python -m scripts.data.tile_whole_images
    python -m scripts.data.tile_whole_images --smoke-test   # images from segment_whole_images --smoke-test

Outputs:
    OUTPUT_ROOT/tiles_whole/<stem>_r<row>_c<col>.png
    OUTPUT_ROOT/splits/whole_image_tiles.csv
    OUTPUT_ROOT/splits/roi/{all,roi}_round<k>.csv
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from oral_cancer.config import OUTPUT_ROOT, relative, resolve
from oral_cancer.stain import load_reference, normalize
from oral_cancer.tissue import MIN_TISSUE_FRACTION, iter_tiles, tissue_mask


# ============================================================
# CONFIGURATION
# ============================================================

FOLDS_CSV = "splits/whole_image_folds.csv"
MASKS_CSV = "splits/whole_image_masks.csv"

TILE_DIR = OUTPUT_ROOT / "tiles_whole"
TILES_CSV = OUTPUT_ROOT / "splits" / "whole_image_tiles.csv"
ROUNDS_DIR = OUTPUT_ROOT / "splits" / "roi"

N_FOLDS = 5
MIN_EPITHELIUM_FRACTION = 0.5

# Same value as scripts/data/preprocess.py (calibrated with
# check_sharpness --calibrate-blur)
TILE_BLUR_SIGMA = 2.5
BLURRED_SOURCES = {"Dataset2_Normal", "Dataset3_Rahman_OSCC"}

REFERENCE = load_reference()


# ============================================================
# TILING
# ============================================================

def tile_image(row):
    """Tile one whole image; returns one record per saved tile."""
    with Image.open(resolve(row["image_path"])) as im:
        img = im.convert("RGB")
    with Image.open(resolve(row["mask_path"])) as m:
        epithelium = np.asarray(m.convert("L").resize(img.size, Image.NEAREST)) > 127
    tissue = tissue_mask(img)

    tiles = []
    for r, c, (x0, y0, x1, y1) in iter_tiles(img):
        tiles.append({"r": r, "c": c, "box": (x0, y0, x1, y1),
                      "tissue_fraction": float(tissue[y0:y1, x0:x1].mean()),
                      "epithelium_fraction": float(epithelium[y0:y1, x0:x1].mean())})
    if not tiles:
        return []

    for t in tiles:
        t["in_all"] = t["tissue_fraction"] >= MIN_TISSUE_FRACTION
        t["in_roi"] = t["epithelium_fraction"] >= MIN_EPITHELIUM_FRACTION
    for flag, key in (("in_all", "tissue_fraction"), ("in_roi", "epithelium_fraction")):
        if not any(t[flag] for t in tiles):
            max(tiles, key=lambda t: t[key])[flag] = True

    stem = row["image_path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
    records = []
    for t in tiles:
        if not (t["in_all"] or t["in_roi"]):
            continue
        tile = img.crop(t["box"])
        if row["source"] in BLURRED_SOURCES:
            tile = tile.filter(ImageFilter.GaussianBlur(TILE_BLUR_SIGMA))
        rgb = np.asarray(tile)
        normalized = normalize(rgb, *REFERENCE)
        out = TILE_DIR / f"{stem}_r{t['r']}_c{t['c']}.png"
        Image.fromarray(normalized if normalized is not None else rgb).save(out)

        records.append({"image_path": relative(out), "parent_image": row["image_path"],
                        "class": row["class"], "source": row["source"], "zoom": row["estimated_zoom"],
                        "fold": row["fold"], "tile_row": t["r"], "tile_col": t["c"],
                        "tissue_fraction": round(t["tissue_fraction"], 4),
                        "epithelium_fraction": round(t["epithelium_fraction"], 4),
                        "in_all": t["in_all"], "in_roi": t["in_roi"]})
    return records


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    folds = pd.read_csv(resolve(FOLDS_CSV))
    masks = pd.read_csv(resolve(MASKS_CSV))
    df = masks[["image_path", "mask_path"]].merge(folds, on="image_path")
    if len(df) < len(folds) and not args.smoke_test:
        raise SystemExit(f"Only {len(df)} of {len(folds)} images have masks; run segment_whole_images first.")

    TILE_DIR.mkdir(parents=True, exist_ok=True)
    ROUNDS_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()
    rows = df.to_dict("records")
    with Pool(args.workers) as pool:
        records = [r for rs in pool.imap_unordered(tile_image, rows, chunksize=4) for r in rs]
    tiles = pd.DataFrame(records).sort_values("image_path").reset_index(drop=True)
    tiles.to_csv(TILES_CSV, index=False)
    print(f"Tiled {len(df)} images into {len(tiles)} tiles in {time.time() - start:.0f}s -> {relative(TILES_CSV)}")

    for setting in ("all", "roi"):
        subset = tiles[tiles[f"in_{setting}"]]
        print(f"\n{setting}: {len(subset)} tiles from {subset['parent_image'].nunique()} images")
        print(subset.groupby(["class", "zoom"]).size().unstack().to_string())

        for k in range(1, N_FOLDS + 1):
            val_fold = k % N_FOLDS + 1
            split = np.where(subset["fold"] == k, "test", np.where(subset["fold"] == val_fold, "validation", "train"))
            subset.assign(split=split).to_csv(ROUNDS_DIR / f"{setting}_round{k}.csv", index=False)

    print(f"\nRound CSVs in {relative(ROUNDS_DIR)}")


if __name__ == "__main__":
    main()
