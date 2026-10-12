"""
Package the data needed on Kaggle into one zip for upload as a
PRIVATE Kaggle Dataset.

Contents: splits/ and preprocessed_normalized/
Output:   kaggle_upload/oral-cancer-data.zip

Images are already compressed PNGs, so they are stored without
re-compression (much faster, same size).
"""

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_ROOT, OUTPUT_ROOT

FOLDERS = ["splits", "preprocessed_normalized"]

OUTPUT_ZIP = OUTPUT_ROOT / "kaggle_upload" / "oral-cancer-data.zip"
OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)

files = [
    path
    for folder in FOLDERS
    for path in sorted((DATA_ROOT / folder).rglob("*"))
    if path.is_file()
]

missing = [f for f in FOLDERS if not (DATA_ROOT / f).exists()]
if missing:
    sys.exit(f"Missing folders: {missing}. Run preprocess.py and stain_normalize.py first.")

print(f"Packaging {len(files)} files...")

with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_STORED) as zf:
    for i, path in enumerate(files, 1):
        compress = zipfile.ZIP_DEFLATED if path.suffix == ".csv" else zipfile.ZIP_STORED
        zf.write(path, path.relative_to(DATA_ROOT).as_posix(), compress_type=compress)
        if i % 1000 == 0:
            print(f"  {i}/{len(files)}")

print(f"\nCreated: {OUTPUT_ZIP} ({OUTPUT_ZIP.stat().st_size / 2**30:.2f} GB)")
print("Upload it at https://www.kaggle.com/datasets -> New Dataset (keep it PRIVATE).")
