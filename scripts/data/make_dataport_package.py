"""
Organise every dataset used or created in this project into one folder
for upload to IEEE DataPort.

Layout (one sub-folder per kind of data):

    01_source_NDB-UFES/        original NDB-UFES data (patches, whole images, metadata)
    02_source_Rahman/          original Rahman et al. images (normal, OSCC)
    03_processed_patches/      our 512x512 patches: tiled + blurred, and stain-normalised
    04_epithelium_annotations/ our 50 AI-assisted epithelium masks and logs
    05_epithelium_predicted_masks/  U-Net masks for all 1,461 whole images
    06_splits_and_metadata/    patch splits and patient-level whole-image folds
    README.md                  contents, credits, licences, caveats

The source data is CC BY 4.0, so it may be redistributed with credit;
the README credits both datasets. Paths inside the CSV files are
rewritten to this layout, so the CSVs work from the package root.

Usage:
    python -m scripts.data.make_dataport_package

Output:
    OUTPUT_ROOT/dataport_upload/
"""

import shutil
import sys

import pandas as pd

from oral_cancer.config import DATA_ROOT, OUTPUT_ROOT, PROJECT_DIR, resolve


# ============================================================
# CONFIGURATION
# ============================================================

OUT = OUTPUT_ROOT / "dataport_upload"

SRC_NDB = "01_source_NDB-UFES"
SRC_RAH = "02_source_Rahman"
PROC = "03_processed_patches"
ANN = "04_epithelium_annotations"
PRED = "05_epithelium_predicted_masks"
META = "06_splits_and_metadata"

# (old project-relative folder, new folder). Order matters: longest prefix first.
FOLDERS = [
    ("dataset_1_ndb_ufes/whole_images/extracted/images", f"{SRC_NDB}/whole_images"),
    ("dataset_1_ndb_ufes/images", f"{SRC_NDB}/patches_512"),
    ("dataset_1_ndb_ufes/metadata", f"{SRC_NDB}/metadata"),
    ("dataset_2_normal/images", f"{SRC_RAH}/normal_images"),
    ("dataset_3_rahman_oscc/images", f"{SRC_RAH}/oscc_images"),
    ("preprocessed_normalized", f"{PROC}/stain_normalized"),
    ("preprocessed", f"{PROC}/tiled_blurred"),
    ("segmentation/whole_images", PRED),
]

# (csv, destination). Tile CSVs are left out: the tiles were made on a GPU server and not kept.
CSVS = [
    ("splits/final_dataset.csv", META),
    ("splits/preprocessed_dataset.csv", META),
    ("splits/normalized_dataset.csv", META),
    ("splits/whole_image_folds.csv", META),
    ("splits/whole_image_masks.csv", META),
]

README = """\
# Oral histopathology data: sources, patches, epithelium masks and splits

Data used and created for the paper "Epithelium Segmentation and Four-Class
Classification of Oral Histopathology Images"
(code: https://github.com/Duggineniakhil/oral-cancer-detection).
Classes: OSCC, leukoplakia with dysplasia, leukoplakia without dysplasia, normal.

## Folders

| Folder | What it is | Who made it |
|---|---|---|
| `01_source_NDB-UFES/` | NDB-UFES: 3,763 labelled 512x512 patches, 237 whole images (2048x1536, 69 patients), metadata CSVs | Ribeiro-de-Assis et al. (2023), **unchanged** |
| `02_source_Rahman/` | 290 normal-mucosa and 934 OSCC images (100x / 400x, 2048x1536) | Rahman et al. (2020), **unchanged** |
| `03_processed_patches/` | 6,626 patches (4 classes) in train/validation/test folders. `tiled_blurred/`: normal images tiled into 512x512 (>= 60% tissue) and blurred (sigma 2.5). `stain_normalized/`: the same after Macenko stain normalisation (used for training) | this project |
| `04_epithelium_annotations/` | 50 epithelium masks (0/255, original image size), the annotation log, and the list of selected images | this project |
| `05_epithelium_predicted_masks/` | U-Net epithelium masks (0/255, half resolution) for all 1,461 whole images | this project |
| `06_splits_and_metadata/` | Patch splits (official NDB-UFES folds 1-4 train, 5 validation, 6 test; normal images split per image before tiling) and patient-level 5-fold splits of the 1,461 whole images | this project |

Paths in the CSV files are relative to this folder.

## Important notes

* **The 50 annotations are AI-assisted** (MedSAM box prompts with manual
  corrections) and have **not been verified by a pathologist**.
* **Predicted masks are model output**, not annotations. Each image was
  segmented by the cross-validation model that held out its fold.
* **Patch folds are patch-level, not patient-level** (NDB-UFES `folder` is a
  stratified fold over patches), so results on them are optimistic.
* **Whole-image folds are patient-level.** NDB-UFES images are grouped by
  patient; Rahman images have no patient IDs, so each image is its own group.
  Round k: test = fold k, validation = fold (k mod 5) + 1, training = the rest.
* `estimated_zoom` (10x / 40x) was estimated from nucleus size (about 95%
  accurate on held-out images), not read from metadata.
* **The normal class comes from a different laboratory** than the leukoplakia
  classes. Models can learn the source instead of the tissue.

## Licence and credit

All data is released under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Folders 01 and 02 contain the
original NDB-UFES and Rahman et al. data, **unchanged**, redistributed under
their CC BY 4.0 licences; folders 03-06 are derived data created by us. This
upload is not endorsed by the original authors. If you use it, please cite the
source datasets:

* Ribeiro-de-Assis, M. C. F., Soares, J. P., de Lima, L. M., de Barros, L. A. P.,
  Grao-Velloso, T. R., Krohling, R. A., & Camisasca, D. R. (2023). NDB-UFES: An
  oral cancer and leukoplakia dataset composed of histopathological images and
  patient data. Data in Brief, 48, 109128. https://doi.org/10.1016/j.dib.2023.109128
  (https://data.mendeley.com/datasets/bbmmm4wgr8/4)
* Rahman, T. Y., Mahanta, L. B., Das, A. K., & Sarma, J. D. (2020).
  Histopathological imaging database for oral cancer analysis. Data in Brief,
  29, 105114. https://doi.org/10.1016/j.dib.2020.105114
  (https://data.mendeley.com/datasets/ftmp4cvtmb/2)
"""


# ============================================================
# HELPERS
# ============================================================

def new_path(path):
    """Old project-relative path -> path inside the package (unchanged if unknown)."""
    if not isinstance(path, str):
        return path
    p = path.replace("\\", "/")
    for old, new in FOLDERS:
        if p.startswith(old + "/"):
            return new + p[len(old):]
    if p.startswith("annotations/epithelium/masks/"):
        return f"{ANN}/masks/" + p.rsplit("/", 1)[-1]
    return path


def copy_tree(src, dst):
    if not src.exists():
        sys.exit(f"Missing folder: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    n = sum(1 for f in dst.rglob("*") if f.is_file())
    print(f"  {dst.relative_to(OUT)}: {n} files")


def write_csv(df, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(new_path)
    df.to_csv(dst, index=False)


# ============================================================
# MAIN
# ============================================================

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    print("Copying images...")
    for old, new in FOLDERS:
        copy_tree(DATA_ROOT / old, OUT / new)

    ann = PROJECT_DIR / "annotations" / "epithelium"
    copy_tree(ann / "masks", OUT / ANN / "masks")
    write_csv(pd.read_csv(ann / "annotations.csv").drop(columns=["saved_at"]), OUT / ANN / "annotations.csv")
    write_csv(pd.read_csv(ann / "annotation_set.csv"), OUT / ANN / "annotation_set.csv")

    print("Writing CSVs...")
    for csv, folder in CSVS:
        src = resolve(csv)
        if src.exists():
            write_csv(pd.read_csv(src), OUT / folder / src.name)
        else:
            print(f"  skipped (not found): {csv}")

    (OUT / "README.md").write_text(README, encoding="utf-8")

    files = [f for f in OUT.rglob("*") if f.is_file()]
    size = sum(f.stat().st_size for f in files) / 2**30
    print(f"\n{len(files)} files, {size:.1f} GB -> {OUT}")


if __name__ == "__main__":
    main()
