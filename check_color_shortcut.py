from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import DATA_ROOT, resolve


# ============================================================
# PATHS
# ============================================================

DATASETS = {
    "Before normalization": DATA_ROOT / "splits" / "preprocessed_dataset.csv",
    "After normalization": DATA_ROOT / "splits" / "normalized_dataset.csv",
}


# ============================================================
# CONFIGURATION
# ============================================================

# Sanity check for the colour shortcut: a linear model that sees ONLY
# colour histograms (no shapes, no cells) tries to predict the class.
# If it can find Normal easily, a CNN can too, and the Normal results
# would reflect staining differences instead of tissue biology.
BINS = 16
THUMBNAIL = 128


# ============================================================
# FEATURES
# ============================================================

def colour_histogram(path):
    with Image.open(resolve(path)) as img:
        img = img.convert("RGB").resize((THUMBNAIL, THUMBNAIL))
        rgb = np.asarray(img)
        hsv = np.asarray(img.convert("HSV"))

    features = []
    for arr in (rgb, hsv):
        for c in range(3):
            hist, _ = np.histogram(arr[:, :, c], bins=BINS, range=(0, 256))
            features.append(hist / hist.sum())

    return np.concatenate(features)


# ============================================================
# EVALUATE
# ============================================================

results = []

for name, csv_path in DATASETS.items():

    if not csv_path.exists():
        print(f"Skipping {name}: {csv_path} not found")
        continue

    df = pd.read_csv(csv_path)
    df = df[df["split"].isin(["train", "test"])]

    X = np.stack([
        colour_histogram(p)
        for p in tqdm(df["image_path"], desc=name)
    ])
    y = df["class"].values

    train = (df["split"] == "train").values
    test = (df["split"] == "test").values

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced")
    )
    model.fit(X[train], y[train])

    pred = model.predict(X[test])
    normal_col = list(model.classes_).index("Normal")
    normal_prob = model.predict_proba(X[test])[:, normal_col]

    results.append({
        "dataset": name,
        "4-class balanced accuracy": balanced_accuracy_score(y[test], pred),
        "Normal-vs-rest AUC": roc_auc_score(y[test] == "Normal", normal_prob),
    })


# ============================================================
# REPORT
# ============================================================

print("\n========================================")
print("COLOUR-ONLY CLASSIFIER (test split)")
print("========================================\n")

print(pd.DataFrame(results).round(3).to_string(index=False))

print("\nChance level: balanced accuracy 0.25, AUC 0.5.")
print("A Normal-vs-rest AUC near 1.0 means colour alone identifies Normal.")
