"""
Package the epithelium annotation images for a Kaggle Dataset.

Only the images listed in annotations/epithelium/annotation_set.csv are
included (about 50 whole images, ~300 MB), with their project-relative
folder structure, plus the CSV itself. On Kaggle,
notebooks/medsam_embeddings_kaggle.ipynb computes MedSAM embeddings for
them on a GPU.

Usage:
    python -m scripts.data.make_annotation_upload

Output:
    kaggle_upload/epithelium-annotation-images.zip
"""

import sys
import zipfile

import pandas as pd

from oral_cancer.config import DATA_ROOT, OUTPUT_ROOT, resolve


# ============================================================
# CONFIGURATION
# ============================================================

SET_CSV = "annotations/epithelium/annotation_set.csv"
OUTPUT_ZIP = OUTPUT_ROOT / "kaggle_upload" / "epithelium-annotation-images.zip"


# ============================================================
# MAIN
# ============================================================

def main():
    rows = pd.read_csv(resolve(SET_CSV))
    files = [resolve(p) for p in rows["image_path"]]

    missing = [str(f) for f in files if not f.exists()]
    if missing:
        sys.exit(f"Missing images: {missing[:5]}")

    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(resolve(SET_CSV), SET_CSV, compress_type=zipfile.ZIP_DEFLATED)
        for rel, f in zip(rows["image_path"], files):
            zf.write(f, rel)

    print(f"Packaged {len(files)} images + {SET_CSV}")
    print(f"Created: {OUTPUT_ZIP} ({OUTPUT_ZIP.stat().st_size / 2**20:.0f} MB)")
    print("Upload it at https://www.kaggle.com/datasets -> New Dataset (private).")


if __name__ == "__main__":
    main()
