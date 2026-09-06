from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"E:\Oral_cancer")

CSV_PATH = ROOT / "splits" / "final_dataset.csv"

OUTPUT_DIR = ROOT / "preprocessed"
OUTPUT_CSV = ROOT / "splits" / "preprocessed_dataset.csv"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_SIZE = (512, 512)

RESAMPLE = Image.LANCZOS


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Total images:", len(df))
print()


# ============================================================
# INSPECT ORIGINAL SIZES
# ============================================================

print("Scanning original image sizes...")

original_sizes = Counter()

for _, row in df.iterrows():
    with Image.open(row["image_path"]) as img:
        original_sizes[img.size] += 1

print("\nOriginal size distribution:")
for size, count in original_sizes.most_common():
    w, h = size
    needs_resize = "<-- needs resize" if (w, h) != TARGET_SIZE else "<-- already target size"
    print(f"  {w}x{h}: {count} images  {needs_resize}")


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

print(f"\nPreprocessing all images to {TARGET_SIZE[0]}x{TARGET_SIZE[1]}...")

records = []
resized_count = 0
skipped_count = 0

for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):

    src_path = Path(row["image_path"])

    # Output: preprocessed/{split}/{class}/{filename}.png
    dest_name = src_path.stem + ".png"
    dest_path = OUTPUT_DIR / row["split"] / row["class"] / dest_name

    with Image.open(src_path) as img:

        original_size = img.size

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if dimensions don't match target
        if img.size != TARGET_SIZE:
            img = img.resize(TARGET_SIZE, resample=RESAMPLE)
            resized_count += 1
        else:
            skipped_count += 1

        img.save(dest_path, format="PNG")

    records.append({
        "image_path": str(dest_path),
        "original_path": str(src_path),
        "class": row["class"],
        "source": row["source"],
        "split": row["split"],
        "original_width": original_size[0],
        "original_height": original_size[1],
        "preprocessed_width": TARGET_SIZE[0],
        "preprocessed_height": TARGET_SIZE[1]
    })


# ============================================================
# SAVE UPDATED CSV
# ============================================================

preprocessed_df = pd.DataFrame(records)

preprocessed_df.to_csv(OUTPUT_CSV, index=False)


# ============================================================
# VERIFICATION
# ============================================================

print("\n========================================")
print("PREPROCESSING COMPLETE")
print("========================================")

print(f"\nTotal processed: {len(records)}")
print(f"Already {TARGET_SIZE[0]}x{TARGET_SIZE[1]}: {skipped_count}")
print(f"Resized to {TARGET_SIZE[0]}x{TARGET_SIZE[1]}: {resized_count}")

print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Output CSV:       {OUTPUT_CSV}")

print("\nPer-split counts:")
for split in ["train", "validation", "test"]:
    count = sum(1 for r in records if r["split"] == split)
    print(f"  {split}: {count}")

print("\nPer-class counts:")
for cls in sorted(set(r["class"] for r in records)):
    count = sum(1 for r in records if r["class"] == cls)
    print(f"  {cls}: {count}")

print("\nImages that were resized (by original size):")
resized_records = [r for r in records if r["original_width"] != TARGET_SIZE[0] or r["original_height"] != TARGET_SIZE[1]]
resize_counter = Counter(
    (r["original_width"], r["original_height"]) for r in resized_records
)
for (w, h), count in resize_counter.most_common():
    print(f"  {w}x{h} -> {TARGET_SIZE[0]}x{TARGET_SIZE[1]}: {count} images")


# ============================================================
# VERIFY OUTPUT
# ============================================================

print("\nVerifying output images...")

verify_errors = 0

for _, row in preprocessed_df.sample(min(50, len(preprocessed_df)), random_state=42).iterrows():

    path = Path(row["image_path"])

    if not path.exists():
        print(f"  MISSING: {path}")
        verify_errors += 1
        continue

    with Image.open(path) as img:
        if img.size != TARGET_SIZE:
            print(f"  WRONG SIZE: {path} is {img.size}")
            verify_errors += 1

if verify_errors == 0:
    print("  All verified samples OK")
else:
    print(f"  {verify_errors} errors found!")
