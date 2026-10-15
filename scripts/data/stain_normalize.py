from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from oral_cancer.config import DATA_ROOT, OUTPUT_ROOT, resolve, relative
from oral_cancer.stain import fit_stains, normalize


# ============================================================
# PATHS
# ============================================================

CSV_PATH = DATA_ROOT / "splits" / "preprocessed_dataset.csv"

OUTPUT_DIR = OUTPUT_ROOT / "preprocessed_normalized"
OUTPUT_CSV = OUTPUT_ROOT / "splits" / "normalized_dataset.csv"

# Remove stale outputs from previous runs
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# Macenko normalization lives in oral_cancer/stain.py so the prediction
# tool applies exactly the same transform. The reference is fitted on
# NDB-UFES TRAINING patches only, so no validation/test information is
# used.
REFERENCE_SAMPLES = 200
RANDOM_SEED = 42


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Total images:", len(df))


# ============================================================
# FIT REFERENCE (NDB-UFES TRAINING PATCHES ONLY)
# ============================================================

ref_pool = df[(df["source"] == "NDB-UFES") & (df["split"] == "train")]
ref_pool = ref_pool.sample(REFERENCE_SAMPLES, random_state=RANDOM_SEED)

print(f"\nFitting reference on {len(ref_pool)} NDB-UFES training patches...")

all_stains = []
all_max_c = []

for path in ref_pool["image_path"]:
    with Image.open(resolve(path)) as img:
        fit = fit_stains(np.asarray(img.convert("RGB")))
    if fit is not None:
        all_stains.append(fit[0])
        all_max_c.append(fit[1])

ref_stains = np.median(np.stack(all_stains), axis=0)
ref_stains /= np.linalg.norm(ref_stains, axis=0)
ref_max_c = np.median(np.stack(all_max_c), axis=0)

print(f"  Fitted on {len(all_stains)} patches")
print("  Reference stain matrix (columns: H, E; rows: R, G, B OD):")
print(np.round(ref_stains, 3))
print("  Reference max concentrations (H, E):", np.round(ref_max_c, 3))

np.savez(
    OUTPUT_DIR / "macenko_reference.npz",
    stains=ref_stains,
    max_c=ref_max_c
)


# ============================================================
# NORMALIZE ALL IMAGES
# ============================================================

print("\nNormalizing...")

records = []
failed = 0

for _, row in tqdm(df.iterrows(), total=len(df), desc="Normalizing"):

    src_path = resolve(row["image_path"])
    out_dir = OUTPUT_DIR / row["split"] / row["class"]
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_path = out_dir / src_path.name

    with Image.open(src_path) as img:
        rgb = np.asarray(img.convert("RGB"))

    normalized = normalize(rgb, ref_stains, ref_max_c)

    # Too little tissue to fit: keep original rather than corrupt it
    if normalized is None:
        normalized = rgb
        failed += 1

    Image.fromarray(normalized).save(dest_path, format="PNG")

    record = row.to_dict()
    record["image_path"] = relative(dest_path)
    record["preprocessed_path"] = row["image_path"]
    record["stain_normalized"] = normalized is not rgb
    records.append(record)


# ============================================================
# SAVE CSV
# ============================================================

normalized_df = pd.DataFrame(records)
normalized_df["folder"] = normalized_df["folder"].astype("Int64")
normalized_df.to_csv(OUTPUT_CSV, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("STAIN NORMALIZATION COMPLETE")
print("========================================")

print(f"\nNormalized: {len(df) - failed}")
print(f"Kept original (too little tissue to fit): {failed}")

if failed:
    print("\nNot normalized, by class:")
    print(normalized_df[~normalized_df["stain_normalized"]]["class"].value_counts())

print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Output CSV:       {OUTPUT_CSV}")
print(f"Reference:        {OUTPUT_DIR / 'macenko_reference.npz'}")
