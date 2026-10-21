"""
Compare whole-image classification with and without segmentation.

Reads the 10 runs written by train_classifier.py for the ROI experiment
(runs/roi_<setting>_round<k>, setting = all | roi, k = 1..5) and scores
each WHOLE image by averaging the class probabilities of its tiles.
Image-level scores are what matter clinically (one answer per image),
and they are comparable between the two settings because both cover
exactly the same images.

Reported per setting as mean +/- SD over the 5 patient-level rounds:
    image-level balanced accuracy and macro-F1 (4 classes)
    image-level balanced accuracy on NDB-UFES images only (3 classes)
    image-level balanced accuracy on 10x and 40x images
    tile-level balanced accuracy (from metrics.json)

Usage:
    python -m scripts.classification.summarize_roi
    python -m scripts.classification.summarize_roi --prefix smoke_roi   # smoke-test runs

Outputs (OUTPUT_ROOT/runs/):
    roi_image_predictions.csv, roi_per_round.csv, roi_summary.csv
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from oral_cancer.config import OUTPUT_ROOT, resolve
from oral_cancer.model import CLASSES_3, CLASSES_4


# ============================================================
# CONFIGURATION
# ============================================================

TILES_CSV = "splits/whole_image_tiles.csv"
SETTINGS = ("all", "roi")
N_FOLDS = 5


# ============================================================
# HELPERS
# ============================================================

def image_predictions(run_dir, tiles):
    """Average tile probabilities per whole image."""
    preds = pd.read_csv(run_dir / "test_predictions.csv")
    preds = preds.merge(tiles[["image_path", "parent_image", "zoom"]], on="image_path")
    prob_cols = [f"prob_{c}" for c in CLASSES_4]
    images = preds.groupby(["parent_image", "class", "source", "zoom"], as_index=False)[prob_cols].mean()
    images["n_tiles"] = preds.groupby("parent_image").size().reindex(images["parent_image"]).values
    images["predicted"] = np.array(CLASSES_4)[images[prob_cols].values.argmax(axis=1)]
    return images


def scores(images):
    out = {
        "image_balanced_accuracy": balanced_accuracy_score(images["class"], images["predicted"]),
        "image_macro_f1": f1_score(images["class"], images["predicted"], labels=CLASSES_4,
                                   average="macro", zero_division=0),
    }
    ndb = images[images["source"] == "NDB-UFES"]
    if len(ndb):
        out["ndb_image_balanced_accuracy"] = balanced_accuracy_score(ndb["class"], ndb["predicted"])
        out["ndb_image_macro_f1"] = f1_score(ndb["class"], ndb["predicted"], labels=CLASSES_3,
                                             average="macro", zero_division=0)
    for zoom, part in images.groupby("zoom"):
        out[f"image_balanced_accuracy_{zoom}"] = balanced_accuracy_score(part["class"], part["predicted"])
    return out


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", default="roi", help="Run folder prefix: <prefix>_<setting>_round<k>")
    args = parser.parse_args()

    tiles = pd.read_csv(resolve(TILES_CSV))
    runs_root = OUTPUT_ROOT / "runs"

    rows, all_images = [], []
    for setting in SETTINGS:
        for k in range(1, N_FOLDS + 1):
            run_dir = runs_root / f"{args.prefix}_{setting}_round{k}"
            if not (run_dir / "test_predictions.csv").exists():
                print(f"missing: {run_dir.name}")
                continue
            images = image_predictions(run_dir, tiles).assign(setting=setting, round=k)
            all_images.append(images)

            metrics = json.loads((run_dir / "metrics.json").read_text())
            rows.append({"setting": setting, "round": k, "n_images": len(images),
                         "n_tiles": int(images["n_tiles"].sum()),
                         "tile_balanced_accuracy": metrics["test"]["balanced_accuracy"],
                         **scores(images)})

    if not rows:
        raise SystemExit("No finished runs found.")

    per_round = pd.DataFrame(rows)
    per_round.to_csv(runs_root / f"{args.prefix}_per_round.csv", index=False)
    pd.concat(all_images).to_csv(runs_root / f"{args.prefix}_image_predictions.csv", index=False)

    metric_cols = [c for c in per_round.columns if c not in ("setting", "round", "n_images", "n_tiles")]
    summary = per_round.groupby("setting")[metric_cols].agg(["mean", "std"])
    summary.to_csv(runs_root / f"{args.prefix}_summary.csv")

    print("\nImage-level results, mean +/- SD over patient-level rounds")
    print(f"{'metric':36s}" + "".join(f"{s:>20s}" for s in SETTINGS if s in summary.index))
    for col in metric_cols:
        line = f"{col:36s}"
        for s in SETTINGS:
            if s in summary.index:
                m, sd = summary.loc[s, (col, "mean")], summary.loc[s, (col, "std")]
                line += f"{100 * m:13.1f} +/- {100 * (0 if pd.isna(sd) else sd):4.1f}"
        print(line)

    print("\nTiles per round:")
    print(per_round.pivot(index="round", columns="setting", values="n_tiles").to_string())


if __name__ == "__main__":
    main()
