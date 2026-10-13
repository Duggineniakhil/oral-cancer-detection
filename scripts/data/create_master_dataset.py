from pathlib import Path
import pandas as pd

from oral_cancer.config import DATA_ROOT, OUTPUT_ROOT, resolve, relative

# ============================================================
# PATHS
# ============================================================

ROOT = DATA_ROOT

D1_IMAGES = ROOT / "dataset_1_ndb_ufes" / "images"
D1_METADATA = ROOT / "dataset_1_ndb_ufes" / "metadata"

D2_IMAGES = ROOT / "dataset_2_normal" / "images"

OUTPUT = OUTPUT_ROOT / "master_dataset.csv"


# ============================================================
# LOAD DATASET 1
# ============================================================

train_df = pd.read_csv(
    D1_METADATA / "sabpatch_parsed_folders.csv"
)

test_df = pd.read_csv(
    D1_METADATA / "sabpatch_parsed_test.csv"
)

train_df["split"] = "train"
test_df["split"] = "test"

d1_df = pd.concat(
    [train_df, test_df],
    ignore_index=True
)


# ============================================================
# CONVERT DATASET 1 LABELS
# ============================================================

label_mapping = {
    0: "Leukoplakia_without_dysplasia",
    1: "OSCC",
    2: "Leukoplakia_with_dysplasia"
}

d1_df["class"] = d1_df["label_number"].map(label_mapping)

d1_df["source"] = "NDB-UFES"

d1_df["image_path"] = d1_df["path"].apply(
    lambda x: relative(D1_IMAGES / x)
)


# ============================================================
# DATASET 2 — NORMAL
# ============================================================

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp"
}

d2_images = sorted([
    p for p in D2_IMAGES.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
])

d2_df = pd.DataFrame({
    "image_path": [relative(p) for p in d2_images],
    "class": "Normal",
    "source": "Dataset2_Normal"
})

# Magnification from filename, e.g. Normal_100x_12.jpg -> 100x
d2_df["magnification"] = [
    p.name.split("_")[1] for p in d2_images
]

# Dataset 2 split will be decided later
d2_df["split"] = "unassigned"


# ============================================================
# COMBINE
# ============================================================

# "folder" is the NDB-UFES cross-validation fold (1-5 train, 6 test),
# created by StratifiedKFold over patches. It is NOT a patient ID.
master_df = pd.concat(
    [
        d1_df[["image_path", "class", "source", "split", "folder"]],
        d2_df[["image_path", "class", "source", "split", "magnification"]]
    ],
    ignore_index=True
)

master_df["folder"] = master_df["folder"].astype("Int64")


# ============================================================
# VERIFY
# ============================================================

print("\n========== MASTER DATASET ==========")

print("Total images:", len(master_df))

print("\nClass distribution:")
print(master_df["class"].value_counts())

print("\nSource distribution:")
print(master_df["source"].value_counts())

print("\nSplit distribution:")
print(master_df["split"].value_counts())


# ============================================================
# CHECK FILE EXISTENCE
# ============================================================

master_df["exists"] = master_df["image_path"].apply(
    lambda x: resolve(x).exists()
)

missing = master_df[~master_df["exists"]]

print("\nMissing images:", len(missing))

if len(missing) > 0:
    print("\nMissing files:")
    print(missing["image_path"].to_string(index=False))


# ============================================================
# SAVE
# ============================================================

master_df.drop(columns=["exists"]).to_csv(
    OUTPUT,
    index=False
)

print("\nMaster dataset saved to:")
print(OUTPUT)