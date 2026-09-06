from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# PATH
# ============================================================

ROOT = Path(r"E:\Oral_cancer")

CSV_PATH = ROOT / "splits" / "final_dataset.csv"


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(CSV_PATH)


# ============================================================
# CLASSES
# ============================================================

classes = [
    "OSCC",
    "Leukoplakia_with_dysplasia",
    "Leukoplakia_without_dysplasia",
    "Normal"
]


# ============================================================
# DISPLAY ONE RANDOM IMAGE PER CLASS
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)

axes = axes.ravel()


for ax, class_name in zip(axes, classes):

    class_df = df[df["class"] == class_name]

    sample = class_df.sample(
        n=1,
        random_state=42
    ).iloc[0]

    image_path = Path(sample["image_path"])

    image = Image.open(image_path)

    ax.imshow(image)

    ax.set_title(
        class_name.replace("_", " "),
        fontsize=14
    )

    ax.axis("off")


plt.tight_layout()

output_path = ROOT / "dataset_samples.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nSaved visualization to:")
print(output_path)