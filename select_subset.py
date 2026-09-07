from pathlib import Path
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"E:\Oral_cancer")

CSV_PATH = ROOT / "splits" / "preprocessed_dataset.csv"

D1_METADATA = ROOT / "dataset_1_ndb_ufes" / "metadata"

OUTPUT_DIR = ROOT / "segmentation_subset_40"
OUTPUT_CSV = OUTPUT_DIR / "subset.csv"

# Clean output directory
if OUTPUT_DIR.exists():
    for f in OUTPUT_DIR.glob("*"):
        f.unlink()

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLES_PER_CLASS = 10
MAX_PER_FOLDER = 2

classes = [
    "OSCC",
    "Leukoplakia_with_dysplasia",
    "Leukoplakia_without_dysplasia",
    "Normal"
]


# ============================================================
# LOAD DATASET + METADATA
# ============================================================

df = pd.read_csv(CSV_PATH)

# Load Dataset 1 metadata (contains folder/lesion info)
train_meta = pd.read_csv(D1_METADATA / "sabpatch_parsed_folders.csv")
test_meta = pd.read_csv(D1_METADATA / "sabpatch_parsed_test.csv")
d1_meta = pd.concat([train_meta, test_meta], ignore_index=True)

# Create lookup: patch filename -> folder
folder_lookup = dict(zip(d1_meta["path"], d1_meta["folder"]))

# Add folder column to main dataframe
df["patch_name"] = df["original_path"].apply(lambda x: Path(x).name)
df["folder"] = df["patch_name"].map(folder_lookup)

print("Total images:", len(df))
print("Images with folder metadata:", df["folder"].notna().sum())
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


def folder_aware_farthest_point_sampling(
    features, folders, n_select, max_per_folder
):
    """
    Farthest-point sampling with folder/patient constraint.

    Ensures at most max_per_folder images from the same
    folder (lesion/patient group) are selected.

    If folders is None (Dataset 2), runs unconstrained.
    """
    n_total = len(features)

    if n_select >= n_total:
        return list(range(n_total))

    # Normalize features
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    normed = (features - mean) / std

    # Track folder counts
    folder_counts = defaultdict(int)
    has_folders = folders is not None

    # Start with the image closest to the centroid
    centroid = normed.mean(axis=0)
    dists_to_centroid = np.linalg.norm(normed - centroid, axis=1)

    # Sort by distance to centroid, pick first valid one
    sorted_indices = np.argsort(dists_to_centroid)
    first_idx = int(sorted_indices[0])

    if has_folders and folders[first_idx] is not None:
        folder_counts[folders[first_idx]] += 1

    selected = [first_idx]

    # Track minimum distance from each point to any selected point
    min_dists = np.linalg.norm(normed - normed[first_idx], axis=1)

    for _ in range(n_select - 1):

        # Mask already selected
        candidate_dists = min_dists.copy()
        candidate_dists[selected] = -1

        # Sort candidates by distance (farthest first)
        sorted_candidates = np.argsort(-candidate_dists)

        # Find the farthest valid candidate
        next_idx = None

        for candidate in sorted_candidates:

            if candidate_dists[candidate] <= 0:
                break

            # Check folder constraint
            if has_folders and folders[candidate] is not None:
                folder = folders[candidate]
                if folder_counts[folder] >= max_per_folder:
                    continue

            next_idx = int(candidate)
            break

        if next_idx is None:
            print("    WARNING: Could not find valid candidate, relaxing constraint")
            # Fallback: pick farthest regardless of folder
            candidate_dists[selected] = -1
            next_idx = int(np.argmax(candidate_dists))

        selected.append(next_idx)

        if has_folders and folders[next_idx] is not None:
            folder_counts[folders[next_idx]] += 1

        # Update minimum distances
        new_dists = np.linalg.norm(normed - normed[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    return selected, dict(folder_counts)


# ============================================================
# SELECT DIVERSE IMAGES PER CLASS
# ============================================================

all_selected = []

for class_name in classes:

    print(f"Processing: {class_name}")

    class_df = df[df["class"] == class_name].reset_index(drop=True)

    print(f"  Total available: {len(class_df)}")

    # Extract features
    print(f"  Extracting features...")

    features_list = []

    for _, row in class_df.iterrows():
        feat = extract_features(row["image_path"])
        features_list.append(feat)

    features_array = np.stack(features_list)

    # Get folder info (None for Dataset 2)
    if class_df["folder"].notna().any():
        folders = class_df["folder"].tolist()
        unique_folders = set(f for f in folders if f is not None and not (isinstance(f, float) and np.isnan(f)))
        print(f"  Unique folders/lesions: {len(unique_folders)}")
    else:
        folders = None
        print(f"  No folder metadata (Dataset 2)")

    # Folder-aware farthest-point sampling
    print(f"  Selecting {SAMPLES_PER_CLASS} diverse images (max {MAX_PER_FOLDER}/folder)...")

    selected_indices, folder_dist = folder_aware_farthest_point_sampling(
        features_array,
        folders,
        SAMPLES_PER_CLASS,
        MAX_PER_FOLDER
    )

    selected_rows = class_df.iloc[selected_indices]

    # Report folder distribution
    if folders is not None:
        print(f"  Folder distribution:")
        for folder, count in sorted(folder_dist.items()):
            print(f"    Folder {int(folder)}: {count} images")

    all_selected.append(selected_rows)

    print(f"  Selected: {len(selected_rows)} images")
    print()


# ============================================================
# COPY IMAGES
# ============================================================

print("Copying images to subset directory...")

records = []

for selected_df in all_selected:

    for _, row in selected_df.iterrows():

        src = Path(row["image_path"])

        dest_name = f"{row['class']}__{src.name}"
        dest = OUTPUT_DIR / dest_name

        shutil.copy2(src, dest)

        folder_val = row.get("folder", None)
        if isinstance(folder_val, float) and np.isnan(folder_val):
            folder_val = None

        records.append({
            "image_path": str(dest),
            "original_path": row.get("original_path", str(src)),
            "class": row["class"],
            "source": row["source"],
            "split": row["split"],
            "folder": int(folder_val) if folder_val is not None else ""
        })


# ============================================================
# SAVE CSV
# ============================================================

subset_df = pd.DataFrame(records)
subset_df.to_csv(OUTPUT_CSV, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("SEGMENTATION SUBSET CREATED")
print("========================================")

print(f"\nTotal images: {len(records)}")

print("\nPer-class:")
for class_name in classes:
    count = sum(1 for r in records if r["class"] == class_name)
    print(f"  {class_name}: {count}")

print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Subset CSV:       {OUTPUT_CSV}")

print("\nSelected images:")
for class_name in classes:
    class_records = [r for r in records if r["class"] == class_name]
    print(f"\n  --- {class_name} ---")
    for r in class_records:
        folder_str = f"  [folder {r['folder']}]" if r["folder"] != "" else ""
        print(f"    {Path(r['image_path']).name}{folder_str}")
