from pathlib import Path
from collections import Counter
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import DATA_ROOT, OUTPUT_ROOT, resolve, relative


# ============================================================
# PATHS
# ============================================================

CSV_PATH = DATA_ROOT / "splits" / "final_dataset.csv"

OUTPUT_DIR = OUTPUT_ROOT / "preprocessed"
OUTPUT_CSV = OUTPUT_ROOT / "splits" / "preprocessed_dataset.csv"

# Remove stale outputs from previous runs (e.g. old resized Normals)
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# NDB-UFES patches are 512x512 crops of 2048x1536 microscope
# images (Leica ICC50 HD, 10x/40x objectives). The Normal images
# come from the same camera at the same objectives, so they are
# TILED into 512x512 crops rather than resized. Resizing would
# shrink cells ~4x and give the classifier a scale shortcut.
TILE_SIZE = 512

# Fraction of a tile that must be tissue. NDB-UFES patches were
# curated to exclude blank areas (~99% of them are >= 0.6 tissue),
# so Normal tiles below 0.6 are discarded to match that curation.
MIN_TISSUE_FRACTION = 0.6

# Tissue detection uses optical density (OD) relative to each
# image's own background (glass) colour. The Normal slides have
# strong, varying colour casts (cyan/green glass), so fixed
# colour or brightness thresholds fail on them.
BACKGROUND_PERCENTILE = 99
OD_THRESHOLD = 0.2


# ============================================================
# HELPERS
# ============================================================

def tissue_mask(img):
    """Boolean mask of pixels that absorb light relative to the glass."""
    rgb = np.asarray(img).astype(np.float32)

    # Glass is the brightest region in brightfield microscopy
    background = np.percentile(
        rgb.reshape(-1, 3),
        BACKGROUND_PERCENTILE,
        axis=0
    )

    od = -np.log(np.clip((rgb + 1) / (background + 1), 1e-6, 1.0))

    return od.max(axis=2) > OD_THRESHOLD


def iter_tiles(img):
    """Yield (row, col, box) for a non-overlapping TILE_SIZE grid."""
    w, h = img.size
    for r in range(h // TILE_SIZE):
        for c in range(w // TILE_SIZE):
            box = (
                c * TILE_SIZE,
                r * TILE_SIZE,
                (c + 1) * TILE_SIZE,
                (r + 1) * TILE_SIZE
            )
            yield r, c, box


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Total source images:", len(df))
print()


# ============================================================
# CREATE SPLIT / CLASS SUBDIRECTORIES
# ============================================================

splits = df["split"].unique()
classes = df["class"].unique()

for split in splits:
    for cls in classes:
        (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)


# ============================================================
# PREPROCESS
# ============================================================

print(f"Preprocessing: copy {TILE_SIZE}x{TILE_SIZE} patches, tile larger images...")

records = []
copied_count = 0
tiles_kept = 0
tiles_discarded = 0
original_sizes = Counter()

for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):

    src_path = resolve(row["image_path"])
    out_dir = OUTPUT_DIR / row["split"] / row["class"]

    with Image.open(src_path) as img:

        img = img.convert("RGB")
        original_size = img.size
        original_sizes[original_size] += 1

        base = {
            "original_path": row["image_path"],
            "class": row["class"],
            "source": row["source"],
            "split": row["split"],
            "folder": row.get("folder"),
            "magnification": row.get("magnification"),
            "original_width": original_size[0],
            "original_height": original_size[1],
        }

        # Already a patch: save as-is (no filtering, already curated)
        if original_size == (TILE_SIZE, TILE_SIZE):

            dest_path = out_dir / (src_path.stem + ".png")
            img.save(dest_path, format="PNG")
            copied_count += 1

            records.append({
                "image_path": relative(dest_path),
                **base,
                "tile_row": 0,
                "tile_col": 0,
                "tissue_fraction": np.nan,  # not filtered
            })
            continue

        # Larger image: cut into tiles, keep tiles with enough tissue
        mask = tissue_mask(img)

        for r, c, box in iter_tiles(img):

            x0, y0, x1, y1 = box
            fraction = float(mask[y0:y1, x0:x1].mean())

            if fraction < MIN_TISSUE_FRACTION:
                tiles_discarded += 1
                continue

            dest_path = out_dir / f"{src_path.stem}_r{r}_c{c}.png"
            img.crop(box).save(dest_path, format="PNG")
            tiles_kept += 1

            records.append({
                "image_path": relative(dest_path),
                **base,
                "tile_row": r,
                "tile_col": c,
                "tissue_fraction": fraction,
            })


# ============================================================
# SAVE UPDATED CSV
# ============================================================

preprocessed_df = pd.DataFrame(records)

preprocessed_df["folder"] = preprocessed_df["folder"].astype("Int64")

preprocessed_df.to_csv(OUTPUT_CSV, index=False)


# ============================================================
# VERIFICATION
# ============================================================

print("\n========================================")
print("PREPROCESSING COMPLETE")
print("========================================")

print("\nOriginal size distribution:")
for (w, h), count in original_sizes.most_common():
    print(f"  {w}x{h}: {count} images")

print(f"\nPatches copied as-is: {copied_count}")
print(f"Tiles kept:           {tiles_kept}")
print(f"Tiles discarded (<{MIN_TISSUE_FRACTION:.0%} tissue): {tiles_discarded}")
print(f"Total output images:  {len(preprocessed_df)}")

print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Output CSV:       {OUTPUT_CSV}")

print("\nClass × Split:")
print(
    pd.crosstab(
        preprocessed_df["class"],
        preprocessed_df["split"],
        margins=True
    )
)

# Every source image must map to exactly one split
leaks = preprocessed_df.groupby("original_path")["split"].nunique()
leaks = leaks[leaks > 1]

if len(leaks) == 0:
    print("\nNo source image appears in more than one split.")
else:
    print(f"\nWARNING: {len(leaks)} source images appear in multiple splits!")

normal_sources = df[df["class"] == "Normal"]["image_path"]
kept_sources = set(preprocessed_df["original_path"])
dropped_sources = [p for p in normal_sources if p not in kept_sources]

if dropped_sources:
    print(f"\nNormal images with no tile kept: {len(dropped_sources)}")
    for p in dropped_sources:
        print(f"  {Path(p).name}")


# ============================================================
# VERIFY OUTPUT
# ============================================================

print("\nVerifying output images...")

verify_errors = 0

for _, row in preprocessed_df.sample(min(50, len(preprocessed_df)), random_state=42).iterrows():

    path = resolve(row["image_path"])

    if not path.exists():
        print(f"  MISSING: {path}")
        verify_errors += 1
        continue

    with Image.open(path) as img:
        if img.size != (TILE_SIZE, TILE_SIZE):
            print(f"  WRONG SIZE: {path} is {img.size}")
            verify_errors += 1

if verify_errors == 0:
    print("  All verified samples OK")
else:
    print(f"  {verify_errors} errors found!")
