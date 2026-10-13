"""
Is the Normal class identifiable from a single global image property?

Grad-CAM showed that Normal predictions rely on patch borders rather
than localised tissue, which suggests a global cue (sharpness, texture,
contrast). Here each property is reduced to ONE number per image and
used alone to separate Normal from the NDB-UFES classes (ROC AUC).

An AUC near 1.0 for a property means it identifies the source dataset
by itself, i.e. it is a candidate shortcut. 0.5 = no information.

Properties are measured on the stain-normalized images, both at full
512 px and at the 224 px the classifier actually sees.

Usage:
    python -m scripts.analysis.check_sharpness                   # 250 images per class
    python -m scripts.analysis.check_sharpness --per-class 0     # all images
    python -m scripts.analysis.check_sharpness --calibrate-blur  # find blur that removes the cue
"""

import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from oral_cancer.config import DATA_ROOT, OUTPUT_ROOT, resolve


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = DATA_ROOT / "splits" / "normalized_dataset.csv"
OUTPUT_DIR = OUTPUT_ROOT / "outputs"

MODEL_INPUT_SIZE = 224
RANDOM_SEED = 42

# 3x3 Laplacian (edge / focus detector)
LAPLACIAN = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128)


# ============================================================
# FEATURES
# ============================================================

def laplacian_variance(gray):
    """Classic focus measure: variance of the Laplacian (higher = sharper)."""
    lap = np.asarray(gray.filter(LAPLACIAN), dtype=np.float32) - 128
    return float(lap.var())


def high_frequency_ratio(gray):
    """Share of spectral energy outside the central low-frequency band."""
    arr = np.asarray(gray, dtype=np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(arr - arr.mean()))) ** 2
    h, w = spectrum.shape
    cy, cx, r = h // 2, w // 2, min(h, w) // 8
    low = spectrum[cy - r:cy + r, cx - r:cx + r].sum()
    return float(1 - low / (spectrum.sum() + 1e-8))


def measure(path, blur_sigma=0.0):
    with Image.open(resolve(path)) as img:
        if blur_sigma > 0:
            img = img.convert("RGB").filter(ImageFilter.GaussianBlur(blur_sigma))
        gray = img.convert("L")
        small = gray.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.BILINEAR)

    arr = np.asarray(gray, dtype=np.float32)

    return {
        "sharpness_512": laplacian_variance(gray),
        "sharpness_224": laplacian_variance(small),
        "high_freq_224": high_frequency_ratio(small),
        "brightness": float(arr.mean()),
        "contrast": float(arr.std()),
    }


# ============================================================
# BLUR CALIBRATION
# ============================================================

def separability(normal_values, other_values):
    """Normal-vs-rest AUC folded to [0.5, 1]."""
    labels = np.r_[np.ones(len(normal_values)), np.zeros(len(other_values))]
    auc = roc_auc_score(labels, np.r_[normal_values, other_values])
    return max(auc, 1 - auc)


def calibrate_blur(df, features, sigmas):
    """
    Blur the Normal images with increasing Gaussian sigma (at 512 px)
    and re-measure them against the unchanged NDB-UFES images.
    The best sigma brings the sharpness AUC closest to 0.5.
    """
    is_normal = (df["class"] == "Normal").values
    others = features[~is_normal]
    normal_paths = df.loc[is_normal, "image_path"].tolist()

    rows = []
    for sigma in sigmas:
        if sigma == 0:
            blurred = features[is_normal]
        else:
            blurred = pd.DataFrame([measure(p, sigma) for p in tqdm(normal_paths, desc=f"sigma {sigma}")])

        rows.append({
            "sigma": sigma,
            **{f"AUC {name}": separability(blurred[name].values, others[name].values)
               for name in features.columns},
            "median Normal sharpness_224": blurred["sharpness_224"].median(),
        })

    sweep = pd.DataFrame(rows)
    sweep["median NDB-UFES sharpness_224"] = others["sharpness_224"].median()

    best = sweep.loc[(sweep["AUC sharpness_224"] - 0.5).abs().idxmin()]

    sweep.to_csv(OUTPUT_DIR / "blur_calibration.csv", index=False)

    print("\n========================================")
    print("BLUR CALIBRATION (Normal blurred, NDB-UFES unchanged)")
    print("========================================\n")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(sweep.set_index("sigma").to_string())
    print(f"\nBest Gaussian sigma (sharpness_224 AUC closest to 0.5): {best['sigma']}")
    print(f"Saved to: {OUTPUT_DIR / 'blur_calibration.csv'}")


# ============================================================
# MAIN
# ============================================================

def main(args):

    if args.low_priority:
        try:
            import psutil
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            print("Running at below-normal priority")
        except Exception as e:  # non-Windows or psutil missing
            print(f"Could not lower priority ({e}); continuing")

    df = pd.read_csv(CSV_PATH)

    if args.per_class:
        df = df.groupby("class").sample(n=args.per_class, random_state=RANDOM_SEED)

    print(f"Images: {len(df)}")
    print(df["class"].value_counts().to_string())

    features = pd.DataFrame([measure(p) for p in tqdm(df["image_path"], desc="Measuring")])
    features.index = df.index
    df = pd.concat([df, features], axis=1)

    # --------------------------------------------------------
    # Single-feature separability: Normal vs NDB-UFES classes
    # --------------------------------------------------------
    is_normal = (df["class"] == "Normal").values

    rows = []
    for name in features.columns:
        auc = roc_auc_score(is_normal, df[name])
        rows.append({
            "property": name,
            "Normal-vs-rest AUC": max(auc, 1 - auc),
            "Normal is": "higher" if auc >= 0.5 else "lower",
            **{
                f"median {c}": df.loc[df["class"] == c, name].median()
                for c in ["OSCC", "Leukoplakia_with_dysplasia", "Leukoplakia_without_dysplasia", "Normal"]
            },
        })

    summary = pd.DataFrame(rows).sort_values("Normal-vs-rest AUC", ascending=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "sharpness_per_image.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "sharpness_summary.csv", index=False)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------
    print("\n========================================")
    print("SINGLE-PROPERTY SEPARABILITY: Normal vs rest")
    print("========================================\n")

    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(summary.set_index("property").to_string())

    print("\nAUC 0.5 = no information, 1.0 = the property alone identifies Normal.")
    print(f"Saved to: {OUTPUT_DIR}")

    if args.calibrate_blur:
        calibrate_blur(df, features, args.sigmas)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Global image properties as Normal-class shortcuts")
    parser.add_argument("--per-class", type=int, default=250, help="Images per class (0 = all)")
    parser.add_argument("--no-low-priority", dest="low_priority", action="store_false",
                        help="Run at normal process priority")
    parser.add_argument("--calibrate-blur", action="store_true",
                        help="Find the Gaussian blur that makes Normal as sharp as NDB-UFES")
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
                        help="Gaussian sigmas (pixels at 512 px) to try")

    main(parser.parse_args())
