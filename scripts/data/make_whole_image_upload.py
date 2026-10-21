"""
Package the whole images for a Kaggle Dataset (ROI experiment).

Contains every image in splits/whole_image_folds.csv (NDB-UFES whole
images, Rahman Normal and Rahman OSCC; ~4.6 GB) with its project-
relative path, plus the fold CSV and the annotation set. The
annotation images are among them, so this one dataset is enough for
notebooks/roi_classification_kaggle.ipynb (segmenter training, whole-
image segmentation, tiling and classification).

Images are already compressed (PNG/JPEG), so they are stored without
re-compression.

Usage:
    python -m scripts.data.make_whole_image_upload

Output:
    kaggle_upload/oral-cancer-whole-images.zip
"""

import sys
import zipfile

import pandas as pd

from oral_cancer.config import OUTPUT_ROOT, resolve


# ============================================================
# CONFIGURATION
# ============================================================

FOLDS_CSV = "splits/whole_image_folds.csv"
SET_CSV = "annotations/epithelium/annotation_set.csv"
OUTPUT_ZIP = OUTPUT_ROOT / "kaggle_upload" / "oral-cancer-whole-images.zip"


# ============================================================
# MAIN
# ============================================================

def main():
    rows = pd.read_csv(resolve(FOLDS_CSV))
    annotated = pd.read_csv(resolve(SET_CSV))

    missing_annotated = set(annotated["image_path"]) - set(rows["image_path"])
    if missing_annotated:
        sys.exit(f"Annotated images not in the folds: {sorted(missing_annotated)[:5]}")

    files = [resolve(p) for p in rows["image_path"]]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        sys.exit(f"Missing images: {missing[:5]}")

    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_STORED) as zf:
        for rel in (FOLDS_CSV, SET_CSV):
            zf.write(resolve(rel), rel, compress_type=zipfile.ZIP_DEFLATED)
        for i, (rel, f) in enumerate(zip(rows["image_path"], files), 1):
            zf.write(f, rel)
            if i % 200 == 0:
                print(f"  {i}/{len(files)}")

    print(f"Packaged {len(files)} images + {FOLDS_CSV} + {SET_CSV}")
    print(f"Created: {OUTPUT_ZIP} ({OUTPUT_ZIP.stat().st_size / 2**30:.2f} GB)")
    print("Upload it at https://www.kaggle.com/datasets -> New Dataset (private).")


if __name__ == "__main__":
    main()
