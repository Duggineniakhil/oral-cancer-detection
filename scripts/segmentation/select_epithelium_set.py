"""
Choose the whole images to annotate with epithelium masks.

The segmenter is trained on a small set of hand-checked masks. The set
must cover every class, both zoom levels and every cross-validation
fold, so the segmenter can later be evaluated on patients (folds) it
was not trained on. Images are drawn evenly from each
(source, class, zoom) group, round-robin over folds, with a fixed seed.

Usage:
    python -m scripts.segmentation.select_epithelium_set
    python -m scripts.segmentation.select_epithelium_set --per-group 6

Output:
    annotations/epithelium/annotation_set.csv
"""

import argparse

import pandas as pd

from oral_cancer.config import DATA_ROOT, OUTPUT_ROOT


# ============================================================
# CONFIGURATION
# ============================================================

FOLDS_CSV = DATA_ROOT / "splits" / "whole_image_folds.csv"
OUTPUT_CSV = OUTPUT_ROOT / "annotations" / "epithelium" / "annotation_set.csv"

RANDOM_SEED = 42


# ============================================================
# MAIN
# ============================================================

def main(args):
    df = pd.read_csv(FOLDS_CSV)
    df = df.sample(frac=1, random_state=RANDOM_SEED)

    picks = []
    for _, g in df.groupby(["source", "class", "estimated_zoom"]):
        # Round-robin over folds so each group is spread over all folds
        g = g.assign(rank=g.groupby("fold").cumcount()).sort_values(["rank", "fold"])
        picks.append(g.head(args.per_group))

    sel = pd.concat(picks).drop(columns="rank")
    sel = sel.sort_values(["fold", "source", "class", "estimated_zoom"]).reset_index(drop=True)
    sel.insert(0, "order", range(1, len(sel) + 1))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUTPUT_CSV, index=False)

    print(f"Selected {len(sel)} images for epithelium annotation\n")
    print(pd.crosstab([sel["source"], sel["class"]], sel["estimated_zoom"]).to_string())
    print("\nImages per fold:", sel["fold"].value_counts().sort_index().to_dict())
    print(f"\nSaved to: {OUTPUT_CSV}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Select whole images for epithelium annotation")
    parser.add_argument("--per-group", type=int, default=5,
                        help="Images per (source, class, zoom) group")

    main(parser.parse_args())
