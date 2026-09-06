from pathlib import Path
import pandas as pd
from PIL import Image
from collections import Counter

# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"E:\Oral_cancer")

CSV_PATH = ROOT / "splits" / "final_dataset.csv"


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Total images:", len(df))


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking image files...")

valid_images = 0
corrupted_images = []
sizes = Counter()
modes = Counter()

for _, row in df.iterrows():

    image_path = Path(row["image_path"])

    try:
        with Image.open(image_path) as img:

            # Verify image can actually be read
            img.verify()

        # Re-open after verify
        with Image.open(image_path) as img:

            sizes[img.size] += 1
            modes[img.mode] += 1

        valid_images += 1

    except Exception as e:

        corrupted_images.append(
            (str(image_path), str(e))
        )


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("IMAGE INSPECTION RESULTS")
print("========================================")

print("\nValid images:", valid_images)
print("Corrupted images:", len(corrupted_images))

print("\nImage dimensions:")
for size, count in sizes.most_common():
    print(f"{size}: {count}")

print("\nImage modes:")
for mode, count in modes.items():
    print(f"{mode}: {count}")


# ============================================================
# CORRUPTED FILES
# ============================================================

if corrupted_images:

    print("\nCorrupted files:")

    for path, error in corrupted_images:
        print(path)
        print("Error:", error)

else:

    print("\nNo corrupted images found.")


# ============================================================
# CLASS × SPLIT
# ============================================================

print("\nClass × Split:")
print(
    pd.crosstab(
        df["class"],
        df["split"]
    )
)