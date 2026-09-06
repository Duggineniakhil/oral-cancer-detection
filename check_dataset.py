from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"E:\Oral_cancer")

D1_IMAGES = ROOT / "dataset_1_ndb_ufes" / "images"
D1_METADATA = ROOT / "dataset_1_ndb_ufes" / "metadata"

TRAIN_CSV = D1_METADATA / "sabpatch_parsed_folders.csv"
TEST_CSV = D1_METADATA / "sabpatch_parsed_test.csv"

D2_IMAGES = ROOT / "dataset_2_normal" / "images"


# ============================================================
# LOAD CSV FILES
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("\n========== DATASET 1 ==========")

print("Train CSV rows:", len(train_df))
print("Test CSV rows :", len(test_df))

print("\nTrain columns:")
print(train_df.columns.tolist())

print("\nTest columns:")
print(test_df.columns.tolist())

print("\nTrain sample:")
print(train_df.head())

print("\nTest sample:")
print(test_df.head())


# ============================================================
# DATASET 1 IMAGE COUNT
# ============================================================

extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

d1_images = [
    p for p in D1_IMAGES.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

print("\nDataset 1 actual images:", len(d1_images))


# ============================================================
# DATASET 2 IMAGE COUNT
# ============================================================

d2_images = [
    p for p in D2_IMAGES.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

print("Dataset 2 Normal images:", len(d2_images))


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\n========== LABEL DISTRIBUTION ==========")

if "label" in train_df.columns:
    print("\nTrain:")
    print(train_df["label"].value_counts().sort_index())

if "label" in test_df.columns:
    print("\nTest:")
    print(test_df["label"].value_counts().sort_index())