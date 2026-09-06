from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"E:\Oral_cancer")

CSV_PATH = ROOT / "splits" / "final_dataset.csv"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(CSV_PATH)


# ============================================================
# CONFIGURATION
# ============================================================

classes = [
    "OSCC",
    "Leukoplakia_with_dysplasia",
    "Leukoplakia_without_dysplasia",
    "Normal"
]

SAMPLES_PER_CLASS = 5
RANDOM_SEED = 42


# ============================================================
# SAMPLE IMAGES
# ============================================================

samples = {}

for class_name in classes:

    class_df = df[df["class"] == class_name]

    sampled = class_df.sample(
        n=SAMPLES_PER_CLASS,
        random_state=RANDOM_SEED
    )

    samples[class_name] = sampled


# ============================================================
# PLOT GRID: 4 rows (classes) × 5 columns (samples)
# ============================================================

fig, axes = plt.subplots(
    len(classes),
    SAMPLES_PER_CLASS,
    figsize=(20, 16)
)

for row, class_name in enumerate(classes):

    class_samples = samples[class_name]

    for col, (_, sample) in enumerate(class_samples.iterrows()):

        ax = axes[row, col]

        image_path = Path(sample["image_path"])
        image = Image.open(image_path)

        ax.imshow(image)
        ax.axis("off")

        if col == 0:
            ax.set_ylabel(
                class_name.replace("_", "\n"),
                fontsize=12,
                fontweight="bold",
                rotation=0,
                labelpad=80,
                va="center"
            )

        if row == 0:
            ax.set_title(
                f"Sample {col + 1}",
                fontsize=11
            )


plt.suptitle(
    "Representative Samples — All 4 Classes",
    fontsize=16,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(rect=[0.08, 0, 1, 0.96])

output_path = OUTPUT_DIR / "class_samples_grid.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nSaved grid to:")
print(output_path)


# ============================================================
# PRINT IMAGE METADATA
# ============================================================

print("\n========================================")
print("SAMPLE IMAGE DETAILS")
print("========================================")

for class_name in classes:

    print(f"\n--- {class_name} ---")

    class_samples = samples[class_name]

    for _, sample in class_samples.iterrows():

        image_path = Path(sample["image_path"])

        with Image.open(image_path) as img:
            w, h = img.size
            mode = img.mode

        print(f"  {image_path.name}  |  {w}×{h}  |  {mode}  |  {sample['source']}")
