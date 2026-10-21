"""
Is the microscope zoom (10x vs 40x objective) tied to the diagnosis in
the NDB-UFES whole images?

NDB-UFES whole images were captured at 10x and 40x, but the metadata
does not record which. If one class was mostly photographed at one zoom,
a classifier could learn the zoom instead of the tissue (a shortcut like
the Normal one). Zoom is estimated from cell-nucleus size: nuclei appear
much larger at 40x than at 10x (75th-percentile diameter ~36 vs ~16 px).

The estimator is calibrated and checked on the Rahman Normal images,
whose zoom is in the file name (Normal_100x_* = 10x objective,
Normal_400x_* = 40x objective), then applied to the 237 NDB-UFES images.

Usage:
    python -m scripts.analysis.check_magnification
    python -m scripts.analysis.check_magnification --normal-per-zoom 0   # all Normal images

Outputs (OUTPUT_ROOT/outputs/):
    magnification_estimates.csv   one row per image
"""

import argparse

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2hed
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from tqdm import tqdm

from oral_cancer.config import DATA_ROOT, OUTPUT_ROOT


# ============================================================
# CONFIGURATION
# ============================================================

NDB_WHOLE_DIR = DATA_ROOT / "dataset_1_ndb_ufes" / "whole_images" / "extracted" / "images"
NDB_META_CSV = DATA_ROOT / "dataset_1_ndb_ufes" / "metadata" / "ndb-ufes.csv"
NORMAL_DIR = DATA_ROOT / "dataset_2_normal" / "images"

OUTPUT_CSV = OUTPUT_ROOT / "outputs" / "magnification_estimates.csv"

DOWNSCALE = 2              # measure at 1024x768 for speed; sizes reported at full resolution
SMOOTH_SIGMA = 1           # merge chromatin specks into whole nuclei
MIN_AREA_PX = 10           # at the downscaled size: ignore specks
SIZE_PERCENTILE = 75       # upper range of nucleus sizes (median is dominated by small fragments)
MAX_AREA_FRACTION = 0.02   # ignore huge merged blobs (> 2% of the image)
RANDOM_SEED = 42


# ============================================================
# NUCLEUS SIZE
# ============================================================

def nucleus_diameter(path):
    """75th-percentile equivalent diameter (px, full resolution) of nuclei.

    Calibrated on the Normal images: smoothing + the 75th percentile
    separates 10x from 40x with ~98% accuracy, the plain median with ~73%.
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((img.width // DOWNSCALE, img.height // DOWNSCALE), Image.BILINEAR)
        rgb = np.asarray(img)

    # Haematoxylin (nuclear stain) channel by colour deconvolution
    h = gaussian(rgb2hed(rgb)[:, :, 0], sigma=SMOOTH_SIGMA)
    mask = h > threshold_otsu(h)
    mask = ndi.binary_fill_holes(ndi.binary_opening(mask, iterations=1))

    max_area = MAX_AREA_FRACTION * mask.size
    areas = np.array([r.area for r in regionprops(label(mask))])
    areas = areas[(areas >= MIN_AREA_PX) & (areas <= max_area)]

    if len(areas) < 20:
        return np.nan, len(areas)

    diameter = np.sqrt(4 * areas / np.pi) * DOWNSCALE
    return float(np.percentile(diameter, SIZE_PERCENTILE)), len(areas)


def measure(paths):
    rows = []
    for p in tqdm(paths, desc="Measuring nuclei"):
        d, n = nucleus_diameter(p)
        rows.append({"file": p.name, "nucleus_diameter_px": d, "nuclei_counted": n})
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main(args):

    if args.low_priority:
        try:
            import psutil
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            print("Running at below-normal priority")
        except Exception as e:
            print(f"Could not lower priority ({e}); continuing")

    # --------------------------------------------------------
    # 1. Calibrate on Normal images with known zoom
    # --------------------------------------------------------
    normal = pd.DataFrame({"path": sorted(NORMAL_DIR.glob("*.jpg"))})
    normal["zoom"] = normal["path"].map(lambda p: "10x" if "_100x_" in p.name else "40x")
    if args.normal_per_zoom:
        normal = normal.groupby("zoom").sample(
            n=args.normal_per_zoom, random_state=RANDOM_SEED, replace=False
        )

    cal = measure(list(normal["path"]))
    cal["zoom"] = normal["zoom"].values
    cal = cal.dropna(subset=["nucleus_diameter_px"])

    # Best single threshold between the two zooms (in log space, nuclei scale ~4x)
    log_d = np.log(cal["nucleus_diameter_px"])
    candidates = np.sort(log_d.values)
    is_40 = (cal["zoom"] == "40x").values
    accs = [((log_d.values > t) == is_40).mean() for t in candidates]
    threshold = float(np.exp(candidates[int(np.argmax(accs))]))
    calib_acc = max(accs)

    print("\n========================================")
    print("CALIBRATION ON NORMAL IMAGES (known zoom)")
    print("========================================")
    print(cal.groupby("zoom")["nucleus_diameter_px"].describe()[["count", "25%", "50%", "75%"]].round(1).to_string())
    print(f"\nThreshold: nucleus diameter > {threshold:.1f} px  ->  40x")
    print(f"Accuracy on the calibration images: {calib_acc:.1%}")

    if args.normal_per_zoom:
        all_normal = pd.DataFrame({"path": sorted(NORMAL_DIR.glob("*.jpg"))})
        held = all_normal[~all_normal["path"].isin(normal["path"])].copy()
        held["zoom"] = held["path"].map(lambda p: "10x" if "_100x_" in p.name else "40x")
        hm = measure(list(held["path"]))
        hm["zoom"] = held["zoom"].values
        hm = hm.dropna(subset=["nucleus_diameter_px"])
        held_acc = ((hm["nucleus_diameter_px"] > threshold) == (hm["zoom"] == "40x")).mean()
        print(f"Accuracy on {len(hm)} held-out Normal images (not used to set the threshold): {held_acc:.1%}")

    # --------------------------------------------------------
    # 2. Apply to NDB-UFES whole images
    # --------------------------------------------------------
    meta = pd.read_csv(NDB_META_CSV)
    ndb = measure([NDB_WHOLE_DIR / p for p in meta["path"]])
    ndb = meta[["path", "patient_id", "lesion_id", "diagnosis"]].join(ndb[["nucleus_diameter_px", "nuclei_counted"]])
    ndb["estimated_zoom"] = np.where(ndb["nucleus_diameter_px"] > threshold, "40x", "10x")
    ndb.loc[ndb["nucleus_diameter_px"].isna(), "estimated_zoom"] = "unknown"

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat([
        ndb.assign(source="NDB-UFES"),
        cal.assign(source="Normal (calibration)", estimated_zoom=np.where(
            cal["nucleus_diameter_px"] > threshold, "40x", "10x")),
    ], ignore_index=True)
    out.to_csv(OUTPUT_CSV, index=False)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------
    print("\n========================================")
    print("NDB-UFES WHOLE IMAGES: ESTIMATED ZOOM vs DIAGNOSIS")
    print("========================================")
    tab = pd.crosstab(ndb["diagnosis"], ndb["estimated_zoom"], margins=True)
    print(tab.to_string())

    share = pd.crosstab(ndb["diagnosis"], ndb["estimated_zoom"], normalize="index").mul(100).round(0)
    print("\nShare of each class (%):")
    print(share.to_string())

    print("\nPatients with images at both zooms:",
          int((ndb[ndb["estimated_zoom"] != "unknown"].groupby("patient_id")["estimated_zoom"].nunique() > 1).sum()),
          "of", ndb["patient_id"].nunique())

    print(f"\nSaved to: {OUTPUT_CSV}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Estimate 10x vs 40x zoom and relate it to the diagnosis")
    parser.add_argument("--normal-per-zoom", type=int, default=60,
                        help="Normal images per zoom for calibration (0 = all)")
    parser.add_argument("--no-low-priority", dest="low_priority", action="store_false",
                        help="Run at normal process priority")

    main(parser.parse_args())
