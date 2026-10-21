"""
Patient-level cross-validation folds for the WHOLE-IMAGE pipeline.

The NDB-UFES whole images (237, 2048x1536) come from 69 patients, and
each patient has several images at both 10x and 40x. A random split
would put images of the same patient in training and test. Here every
patient is assigned to exactly one of 5 folds (stratified by diagnosis),
so a model is always tested on patients it never saw. Patients of each
class are spread evenly over the folds (by patient and image counts).

With only 17 patients in the smallest class, a single train/test split
would leave ~3 of them in the test set; 5-fold cross-validation tests
every image once instead.

The Rahman et al. dataset adds Normal (290) and OSCC (934) whole images.
It has no patient IDs, so each Rahman image is its own group (a stated
limitation). Its OSCC images mean OSCC comes from both sources, so the
source no longer identifies a class on its own.

How to use the folds, round k = 1..5:
    test = fold k, validation = fold (k mod 5) + 1, train = the other 3

Usage:
    python -m scripts.data.create_whole_image_splits

Output:
    splits/whole_image_folds.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from oral_cancer.config import DATA_ROOT, OUTPUT_ROOT, relative


# ============================================================
# CONFIGURATION
# ============================================================

NDB_WHOLE_DIR = DATA_ROOT / "dataset_1_ndb_ufes" / "whole_images" / "extracted" / "images"
NDB_META_CSV = DATA_ROOT / "dataset_1_ndb_ufes" / "metadata" / "ndb-ufes.csv"
NORMAL_DIR = DATA_ROOT / "dataset_2_normal" / "images"
RAHMAN_OSCC_DIR = DATA_ROOT / "dataset_3_rahman_oscc" / "images"
ZOOM_CSV = OUTPUT_ROOT / "outputs" / "magnification_estimates.csv"

OUTPUT_CSV = OUTPUT_ROOT / "splits" / "whole_image_folds.csv"

N_FOLDS = 5
RANDOM_SEED = 42

# NDB-UFES diagnosis text -> class names used everywhere else
CLASS_MAP = {
    "OSCC": "OSCC",
    "Leukoplakia with dysplasia": "Leukoplakia_with_dysplasia",
    "Leukoplakia without dysplasia": "Leukoplakia_without_dysplasia",
}


# ============================================================
# MAIN
# ============================================================

def main(args):

    # --------------------------------------------------------
    # NDB-UFES whole images (grouped by patient)
    # --------------------------------------------------------
    meta = pd.read_csv(NDB_META_CSV)
    ndb = pd.DataFrame({
        "image_path": [relative(NDB_WHOLE_DIR / p) for p in meta["path"]],
        "class": meta["diagnosis"].map(CLASS_MAP),
        "source": "NDB-UFES",
        "patient_id": meta["patient_id"].astype(str),
        "lesion_id": meta["lesion_id"].astype(str),
        "group": "ndb-patient-" + meta["patient_id"].astype(str),
    })
    assert ndb["class"].notna().all(), "Unknown diagnosis in ndb-ufes.csv"

    if ZOOM_CSV.exists():
        zoom = pd.read_csv(ZOOM_CSV)
        zoom = zoom[zoom["source"] == "NDB-UFES"].set_index("path")["estimated_zoom"]
        ndb["estimated_zoom"] = meta["path"].map(zoom).values

    # --------------------------------------------------------
    # Rahman whole images: Normal + OSCC (no patient IDs: one group per image)
    # --------------------------------------------------------
    def rahman(folder, cls, source):
        paths = sorted(folder.glob("*.jpg"))
        return pd.DataFrame({
            "image_path": [relative(p) for p in paths],
            "class": cls,
            "source": source,
            "patient_id": "",
            "lesion_id": "",
            "group": [f"{source.lower()}-image-{p.stem}" for p in paths],
            "estimated_zoom": ["10x" if "_100x_" in p.name else "40x" for p in paths],
        })

    normal = rahman(NORMAL_DIR, "Normal", "Dataset2_Normal")
    rahman_oscc = rahman(RAHMAN_OSCC_DIR, "OSCC", "Dataset3_Rahman_OSCC")
    assert len(rahman_oscc) > 0, f"No Rahman OSCC images found in {RAHMAN_OSCC_DIR}"

    df = pd.concat([ndb, normal, rahman_oscc], ignore_index=True)

    # --------------------------------------------------------
    # Folds
    # --------------------------------------------------------
    # NDB-UFES: class by class, patients are dealt to folds greedily (largest
    # patient first, into the fold with the fewest images of that class), so
    # patients AND images of every class are spread evenly. A generic
    # stratified-group split is dominated by the 290 single-image Normal
    # groups and left one fold with a single "without dysplasia" patient.
    rng = np.random.default_rng(RANDOM_SEED)
    df["fold"] = 0
    for cls in sorted(ndb["class"].unique()):
        sizes = df[(df["source"] == "NDB-UFES") & (df["class"] == cls)].groupby("group").size()
        order = sizes.sample(frac=1, random_state=int(rng.integers(1e9))).sort_values(ascending=False, kind="stable")
        images = np.zeros(N_FOLDS, dtype=int)
        patients = np.zeros(N_FOLDS, dtype=int)
        for group, n in order.items():
            f = int(np.lexsort((patients, images))[0])   # fewest images, then fewest patients
            df.loc[df["group"] == group, "fold"] = f + 1
            images[f] += n
            patients[f] += 1

    # Rahman (Normal + OSCC): one image per group, stratified by class and zoom
    rm = df["source"] != "NDB-UFES"
    strata = df.loc[rm, "class"] + "-" + df.loc[rm, "estimated_zoom"]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rahman_idx = df.index[rm]
    for fold, (_, test_idx) in enumerate(skf.split(rahman_idx, strata), start=1):
        df.loc[rahman_idx[test_idx], "fold"] = fold

    # No group may span two folds
    assert (df.groupby("group")["fold"].nunique() == 1).all(), "A patient spans several folds"

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------
    print("========================================")
    print(f"WHOLE-IMAGE FOLDS ({N_FOLDS}, grouped by patient)")
    print("========================================")

    print("\nImages per fold and class:")
    print(pd.crosstab(df["class"], df["fold"], margins=True).to_string())

    print("\nImages per fold and source:")
    print(pd.crosstab(df["source"], df["fold"], margins=True).to_string())

    ndb_part = df[df["source"] == "NDB-UFES"]
    print("\nNDB-UFES patients per fold and class:")
    print(pd.crosstab(ndb_part["class"], ndb_part["fold"], values=ndb_part["patient_id"],
                      aggfunc="nunique").fillna(0).astype(int).to_string())

    if "estimated_zoom" in df:
        print("\nZoom per fold (NDB-UFES):")
        print(pd.crosstab(ndb_part["estimated_zoom"], ndb_part["fold"]).to_string())

    print("\nRound k: test = fold k, validation = fold (k mod 5) + 1, train = the other 3")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Patient-level folds for the whole-image pipeline")
    main(parser.parse_args())
