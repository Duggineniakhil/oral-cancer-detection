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
# LOAD MASTER DATASET
# ============================================================

df = pd.read_csv(MASTER_CSV)

print("Original dataset:")
print(df["class"].value_counts())


# ============================================================
# DATASET 1
# ============================================================

d1 = df[df["source"] == "NDB-UFES"].copy()

d1_train = d1[d1["split"] == "train"].copy()
d1_test = d1[d1["split"] == "test"].copy()

print("\nDataset 1:")
print("Training:", len(d1_train))
print("Test:", len(d1_test))


# ============================================================
# CREATE VALIDATION SET FROM DATASET 1 TRAINING SET
# ============================================================

d1_train, d1_val = train_test_split(
    d1_train,
    test_size=0.20,
    stratify=d1_train["class"],
    random_state=42
)

print("\nDataset 1 after split:")
print("Train:", len(d1_train))
print("Validation:", len(d1_val))
print("Test:", len(d1_test))


# ============================================================
# DATASET 2 — NORMAL
# ============================================================

d2 = df[df["source"] == "Dataset2_Normal"].copy()

print("\nDataset 2 Normal:", len(d2))


# First: 15% test
d2_train_val, d2_test = train_test_split(
    d2,
    test_size=0.15,
    random_state=42
)

# Then: 15% of total for validation
# 15 / 85 = approximately 17.65% of remaining data
d2_train, d2_val = train_test_split(
    d2_train_val,
    test_size=(0.15 / 0.85),
    random_state=42
)

print("\nDataset 2 split:")
print("Train:", len(d2_train))
print("Validation:", len(d2_val))
print("Test:", len(d2_test))


# ============================================================
# ASSIGN SPLITS
# ============================================================

d1_train["split"] = "train"
d1_val["split"] = "validation"
d1_test["split"] = "test"

d2_train["split"] = "train"
d2_val["split"] = "validation"
d2_test["split"] = "test"


# ============================================================
# COMBINE
# ============================================================

final_df = pd.concat(
    [
        d1_train,
        d1_val,
        d1_test,
        d2_train,
        d2_val,
        d2_test
    ],
    ignore_index=True
)


# ============================================================
# SHUFFLE
# ============================================================

final_df = final_df.sample(
    frac=1,
    random_state=42
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
print("FINAL DATASET")
print("========================================")

print("\nTotal:", len(final_df))

print("\nClass distribution:")
print(
    final_df["class"]
    .value_counts()
    .sort_index()
)

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

print("\nSource × Split:")
print(
    pd.crosstab(
        final_df["source"],
        final_df["split"]
    )
)

print("\nSaved to:")
print(final_csv)