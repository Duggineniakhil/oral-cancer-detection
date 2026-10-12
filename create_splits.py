from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"E:\Oral_cancer")

MASTER_CSV = ROOT / "master_dataset.csv"
OUTPUT_DIR = ROOT / "splits"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# NDB-UFES ships with predefined folds (see the authors' code,
# lmlima/IM_ComplementaryData_OralLeukoplakia):
#   folder 6     -> official test set
#   folders 1-5  -> training folds
# We keep the official test set and hold out one training fold
# as validation, so results are comparable with published work
# (e.g. Maia et al., 2024).
#
# NOTE: the folds were made with StratifiedKFold over patches,
# not patients, so they are NOT patient-disjoint. This is a
# known limitation of the benchmark and must be reported.
TEST_FOLD = 6
VAL_FOLD = 5

# Normal images (Dataset 2) are split at the IMAGE level, before
# tiling, so all tiles of one image stay in the same split.
D2_VAL_FRACTION = 0.15
D2_TEST_FRACTION = 0.15

RANDOM_SEED = 42


# ============================================================
# LOAD MASTER DATASET
# ============================================================

df = pd.read_csv(MASTER_CSV)

print("Original dataset:")
print(df["class"].value_counts())


# ============================================================
# DATASET 1 — OFFICIAL NDB-UFES FOLDS
# ============================================================

d1 = df[df["source"] == "NDB-UFES"].copy()

d1["split"] = "train"
d1.loc[d1["folder"] == VAL_FOLD, "split"] = "validation"
d1.loc[d1["folder"] == TEST_FOLD, "split"] = "test"

print("\nDataset 1 (official folds):")
print(f"Validation fold: {VAL_FOLD} | Test fold: {TEST_FOLD}")
print(d1["split"].value_counts())


# ============================================================
# DATASET 2 — NORMAL (IMAGE-LEVEL SPLIT)
# ============================================================

d2 = df[df["source"] == "Dataset2_Normal"].copy()

print("\nDataset 2 Normal images:", len(d2))

# Stratify by magnification so every split has 100x and 400x
d2_train_val, d2_test = train_test_split(
    d2,
    test_size=D2_TEST_FRACTION,
    stratify=d2["magnification"],
    random_state=RANDOM_SEED
)

d2_train, d2_val = train_test_split(
    d2_train_val,
    test_size=D2_VAL_FRACTION / (1 - D2_TEST_FRACTION),
    stratify=d2_train_val["magnification"],
    random_state=RANDOM_SEED
)

d2_train["split"] = "train"
d2_val["split"] = "validation"
d2_test["split"] = "test"

print("\nDataset 2 split (source images, before tiling):")
print("Train:", len(d2_train))
print("Validation:", len(d2_val))
print("Test:", len(d2_test))


# ============================================================
# COMBINE
# ============================================================

final_df = pd.concat(
    [
        d1,
        d2_train,
        d2_val,
        d2_test
    ],
    ignore_index=True
)

final_df["folder"] = final_df["folder"].astype("Int64")


# ============================================================
# SHUFFLE
# ============================================================

final_df = final_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

final_csv = OUTPUT_DIR / "final_dataset.csv"

final_df.to_csv(
    final_csv,
    index=False
)


# ============================================================
# VERIFY
# ============================================================

print("\n========================================")
print("FINAL DATASET (source images)")
print("========================================")

print("\nTotal:", len(final_df))

print("\nSplit distribution:")
print(
    final_df["split"]
    .value_counts()
)

print("\nClass × Split:")
print(
    pd.crosstab(
        final_df["class"],
        final_df["split"]
    )
)

print("\nNDB-UFES fold × Split:")
print(
    pd.crosstab(
        final_df["folder"],
        final_df["split"]
    )
)

print("\nNormal magnification × Split:")
print(
    pd.crosstab(
        final_df["magnification"],
        final_df["split"]
    )
)

# No image may appear in more than one split
assert final_df["image_path"].is_unique, "Duplicate image paths found"

print("\nSaved to:")
print(final_csv)
print("\nNext: run preprocess.py to tile the Normal images.")
