from pathlib import Path
import shutil

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"E:\Oral_cancer")

CSV_PATH = ROOT / "splits" / "final_dataset.csv"

ANNOTATION_DIR = ROOT / "annotation"
ANNOTATION_IMAGES = ANNOTATION_DIR / "images"

ANNOTATION_DIR.mkdir(exist_ok=True)
ANNOTATION_IMAGES.mkdir(exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLES_PER_CLASS = 50
RANDOM_SEED = 42

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


# ============================================================
# STRATIFIED SAMPLING
# ============================================================

selected = []

for class_name in classes:

    class_df = df[df["class"] == class_name]

    available = len(class_df)

    n = min(SAMPLES_PER_CLASS, available)

    sampled = class_df.sample(
        n=n,
        random_state=RANDOM_SEED
    )

    selected.append(sampled)

    print(f"{class_name}: selected {n} / {available}")


annotation_df = pd.concat(selected, ignore_index=True)


# ============================================================
# COPY IMAGES TO ANNOTATION DIRECTORY
# ============================================================

print("\nCopying images...")

records = []

for _, row in annotation_df.iterrows():

    src = Path(row["image_path"])

    # Use class prefix to avoid filename collisions
    dest_name = f"{row['class']}__{src.name}"

    dest = ANNOTATION_IMAGES / dest_name

    shutil.copy2(src, dest)

    records.append({
        "image_name": dest_name,
        "image_path": str(dest),
        "original_path": str(src),
        "class": row["class"],
        "source": row["source"],
        "split": row["split"],
        "mask_path": ""  # to be filled after annotation
    })


# ============================================================
# SAVE ANNOTATION CSV
# ============================================================

annotation_csv = ANNOTATION_DIR / "annotation_set.csv"

pd.DataFrame(records).to_csv(
    annotation_csv,
    index=False
)


# ============================================================
# CREATE CLASS SUBDIRECTORIES (for organized annotation)
# ============================================================

for class_name in classes:

    class_dir = ANNOTATION_IMAGES / class_name

    class_dir.mkdir(exist_ok=True)

    # Also copy into class subdirectories
    class_records = [
        r for r in records
        if r["class"] == class_name
    ]

    for rec in class_records:
        src = Path(rec["image_path"])
        dest = class_dir / Path(rec["image_name"]).name
        shutil.copy2(src, dest)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("ANNOTATION SET PREPARED")
print("========================================")

print(f"\nTotal images: {len(records)}")

print(f"\nImages copied to: {ANNOTATION_IMAGES}")
print(f"Annotation CSV:   {annotation_csv}")

print("\nPer-class breakdown:")
for class_name in classes:
    count = sum(1 for r in records if r["class"] == class_name)
    print(f"  {class_name}: {count}")

print("\n--- Next Steps ---")
print("1. Open the images in an annotation tool (CVAT / Label Studio / napari)")
print("2. Create binary masks for each image")
print("3. Save masks as: annotation/masks/{image_name}_mask.png")
print("4. Update annotation_set.csv with mask_path column")
