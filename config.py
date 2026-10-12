"""
Shared paths so the same scripts run on a laptop and on Kaggle.

DATA_ROOT    where inputs are read from (datasets, splits/, preprocessed*/)
OUTPUT_ROOT  where outputs are written

Both default to this project folder. Override with environment
variables, e.g. on Kaggle:

    ORAL_CANCER_DATA_ROOT=/kaggle/input/oral-cancer-data
    ORAL_CANCER_OUTPUT_ROOT=/kaggle/working

Paths stored in CSV files are RELATIVE (POSIX style, e.g.
"preprocessed/train/OSCC/p0001.png") so the CSVs work on any machine.
Use resolve() to turn a CSV path into a real path and relative() to
store one.
"""

import os
from pathlib import Path, PurePosixPath

PROJECT_DIR = Path(__file__).resolve().parent

DATA_ROOT = Path(os.environ.get("ORAL_CANCER_DATA_ROOT", PROJECT_DIR))
OUTPUT_ROOT = Path(os.environ.get("ORAL_CANCER_OUTPUT_ROOT", PROJECT_DIR))


def resolve(path):
    """CSV path -> absolute Path.

    Relative paths are looked up in OUTPUT_ROOT first (files produced
    in this run/environment), then DATA_ROOT.
    """
    path = Path(str(path).replace("\\", "/"))

    if path.is_absolute():
        return path

    candidate = OUTPUT_ROOT / path
    if candidate.exists():
        return candidate

    return DATA_ROOT / path


def relative(path):
    """Absolute Path -> relative POSIX string for storing in a CSV."""
    path = Path(path).resolve()

    for root in (OUTPUT_ROOT, DATA_ROOT):
        try:
            return str(PurePosixPath(path.relative_to(root.resolve())))
        except ValueError:
            continue

    raise ValueError(f"{path} is not inside DATA_ROOT or OUTPUT_ROOT")
