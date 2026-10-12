from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import DATA_ROOT, OUTPUT_ROOT, resolve, relative


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

# Macenko et al., "A method for normalizing histology slides for
# quantitative analysis", ISBI 2009.
#
# NDB-UFES (pink) and the Normal dataset (blue-purple) were stained
# and imaged at different institutions, so colour alone separates
# Normal from the other classes. Every image is mapped to one
# reference H&E appearance to remove that shortcut.
#
# The reference is fitted on NDB-UFES TRAINING patches only, so no
# validation/test information is used.
LIGHT_INTENSITY = 240      # Io: brightness of the light source
OD_BETA = 0.15             # ignore near-transparent pixels
ANGLE_PERCENTILE = 1       # alpha: robust extreme stain angles
MAX_C_PERCENTILE = 99      # robust maximum stain concentration
MIN_TISSUE_PIXELS = 500    # below this, the fit is unreliable

REFERENCE_SAMPLES = 200
RANDOM_SEED = 42


# ============================================================
# MACENKO
# ============================================================

def to_od(rgb):
    """RGB uint8 (H, W, 3) -> optical density (N, 3)."""
    rgb = rgb.reshape(-1, 3).astype(np.float64)
    return -np.log((rgb + 1) / LIGHT_INTENSITY)


def fit_stains(rgb):
    """
    Estimate the H&E stain matrix (3x2, columns = H, E) and the
    robust maximum concentration of each stain.

    Returns None if the image has too little tissue to fit.
    """
    od = to_od(rgb)

    tissue = od[np.all(od > OD_BETA, axis=1)]

    if len(tissue) < MIN_TISSUE_PIXELS:
        return None

    # Plane spanned by the two largest eigenvectors of the OD cloud
    _, eigvecs = np.linalg.eigh(np.cov(tissue.T))
    plane = eigvecs[:, 1:3]

    projected = tissue @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])

    min_angle = np.percentile(angles, ANGLE_PERCENTILE)
    max_angle = np.percentile(angles, 100 - ANGLE_PERCENTILE)

    v_min = plane @ np.array([np.cos(min_angle), np.sin(min_angle)])
    v_max = plane @ np.array([np.cos(max_angle), np.sin(max_angle)])

    # Haematoxylin absorbs more red light than eosin -> larger OD_R
    if v_min[0] > v_max[0]:
        stains = np.array([v_min, v_max]).T
    else:
        stains = np.array([v_max, v_min]).T

    # Eigenvector signs are arbitrary; stain OD vectors are positive
    stains *= np.sign(stains.sum(axis=0))
    stains /= np.linalg.norm(stains, axis=0)

    concentrations = np.linalg.lstsq(stains, od.T, rcond=None)[0]
    max_c = np.percentile(concentrations, MAX_C_PERCENTILE, axis=1)

    return stains, max_c, concentrations


def normalize(rgb, ref_stains, ref_max_c):
    """Map an RGB image onto the reference stain appearance."""
    fit = fit_stains(rgb)

    if fit is None:
        return None

    _, max_c, concentrations = fit

    concentrations = concentrations * (ref_max_c / max_c)[:, None]

    od = ref_stains @ concentrations
    out = LIGHT_INTENSITY * np.exp(-od)

    return np.clip(out.T, 0, 255).reshape(rgb.shape).astype(np.uint8)


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
