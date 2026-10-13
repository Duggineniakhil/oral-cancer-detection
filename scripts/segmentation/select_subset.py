from pathlib import Path
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from oral_cancer.config import DATA_ROOT, OUTPUT_ROOT, resolve


# ============================================================
# PATHS
# ============================================================

CSV_PATH = DATA_ROOT / "splits" / "normalized_dataset.csv"

OUTPUT_DIR = OUTPUT_ROOT / "segmentation_subset"
OUTPUT_CSV = OUTPUT_DIR / "subset.csv"

# Clean output directory (including any subfolders from earlier runs)
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLES_PER_CLASS = 10

# At most this many images from the same SOURCE image. This matters
# for Normal, where up to 12 tiles come from one microscope image.
# (NDB-UFES has no patch-to-patient/image mapping, so each patch is
# its own group. Its "folder" column is a CV fold, not a patient.)
MAX_PER_GROUP = 2

classes = [
    "OSCC",
    "Leukoplakia_with_dysplasia",
    "Leukoplakia_without_dysplasia",
    "Normal"
]


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(CSV_PATH)

df["group"] = df["original_path"]

print("Total images:", len(df))
print()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image_path):
    """
    Extract a compact feature vector for diversity selection.

    Uses:
    - RGB channel means and standard deviations (6 features)
    - Spatial grid means: divide image into 4x4 grid,
      compute mean intensity per cell (16 features)
    - Color histogram: 8 bins per RGB channel (24 features)

    Total: 46-dimensional feature vector
    """
    with Image.open(image_path) as img:
        arr = np.array(img.convert("RGB"), dtype=np.float32)

    features = []

    # Per-channel mean and std
    for c in range(3):
        features.append(arr[:, :, c].mean())
        features.append(arr[:, :, c].std())

    # 4x4 spatial grid means (grayscale)
    gray = arr.mean(axis=2)
    h, w = gray.shape
    gh, gw = h // 4, w // 4

    for i in range(4):
        for j in range(4):
            cell = gray[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
            features.append(cell.mean())

    # Color histogram (8 bins per channel)
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=8, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-8
        features.extend(hist.tolist())

    return np.array(features, dtype=np.float32)


def group_aware_farthest_point_sampling(
    features, groups, n_select, max_per_group
):
    """
    Farthest-point sampling with a per-group cap.

    Ensures at most max_per_group images from the same group
    (source image) are selected.
    """
    n_total = len(features)

    if n_select >= n_total:
        return list(range(n_total))

    # Normalize features
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    normed = (features - mean) / std

    group_counts = defaultdict(int)

    # Start with the image closest to the centroid
    centroid = normed.mean(axis=0)
    first_idx = int(np.argmin(np.linalg.norm(normed - centroid, axis=1)))

    selected = [first_idx]
    group_counts[groups[first_idx]] += 1

    # Track minimum distance from each point to any selected point
    min_dists = np.linalg.norm(normed - normed[first_idx], axis=1)

    for _ in range(n_select - 1):

        candidate_dists = min_dists.copy()
        candidate_dists[selected] = -1

        next_idx = None

        # Farthest candidate whose group is not full
        for candidate in np.argsort(-candidate_dists):

            if candidate_dists[candidate] <= 0:
                break

            if group_counts[groups[candidate]] >= max_per_group:
                continue

            next_idx = int(candidate)
            break

        if next_idx is None:
            print("    WARNING: Could not find valid candidate, relaxing constraint")
            next_idx = int(np.argmax(candidate_dists))

        selected.append(next_idx)
        group_counts[groups[next_idx]] += 1

        new_dists = np.linalg.norm(normed - normed[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    return selected


# ============================================================
# SELECT DIVERSE IMAGES PER CLASS
# ============================================================

all_selected = []

for class_name in classes:

    print(f"Processing: {class_name}")

    class_df = df[df["class"] == class_name].reset_index(drop=True)

    print(f"  Total available: {len(class_df)}")
    print(f"  Source images:   {class_df['group'].nunique()}")

    print(f"  Extracting features...")

    features_array = np.stack([
        extract_features(resolve(p))
        for p in class_df["image_path"]
    ])

    print(f"  Selecting {SAMPLES_PER_CLASS} diverse images (max {MAX_PER_GROUP}/source image)...")

    selected_indices = group_aware_farthest_point_sampling(
        features_array,
        class_df["group"].tolist(),
        SAMPLES_PER_CLASS,
        MAX_PER_GROUP
    )

    selected_rows = class_df.iloc[selected_indices]

    all_selected.append(selected_rows)

    print(f"  Selected: {len(selected_rows)} images")
    print()


# ============================================================
# COPY IMAGES (for browsing) AND SAVE CSV
# ============================================================

print("Copying images to subset directory...")

subset_df = pd.concat(all_selected, ignore_index=True).drop(columns=["group"])

preview_dir = OUTPUT_DIR / "images"
preview_dir.mkdir(parents=True, exist_ok=True)

for _, row in subset_df.iterrows():
    src = resolve(row["image_path"])
    shutil.copy2(src, preview_dir / f"{row['class']}__{src.name}")

# image_path keeps pointing at the dataset image, so segmentation
# results can be joined back to the full dataset
subset_df["folder"] = subset_df["folder"].astype("Int64")
subset_df.to_csv(OUTPUT_CSV, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("SEGMENTATION SUBSET CREATED")
print("========================================")

print(f"\nTotal images: {len(subset_df)}")

print("\nPer-class:")
print(subset_df["class"].value_counts().to_string())

print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Subset CSV:       {OUTPUT_CSV}")

print("\nSelected images:")
for class_name in classes:
    print(f"\n  --- {class_name} ---")
    for p in subset_df[subset_df["class"] == class_name]["image_path"]:
        print(f"    {Path(p).name}")
