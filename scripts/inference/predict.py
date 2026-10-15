"""
Predict the class of histopathology images with a trained model and
save a Grad-CAM overlay for each.

Accepts 512x512 patches or larger microscope images (tiled into
512x512 patches, never resized). RESEARCH TOOL ONLY, not for diagnosis.

Usage:
    python -m scripts.inference.predict path/to/image.png
    python -m scripts.inference.predict img1.png img2.jpg --model 3class
    python -m scripts.inference.predict image.png --run convnext_tiny_4class_224px

Outputs (OUTPUT_ROOT/outputs/predictions/):
    <name>_prediction.png    original | normalized + Grad-CAM, with probabilities
    <name>_prediction.json   probabilities, per-tile results, warnings
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from oral_cancer.config import OUTPUT_ROOT
from oral_cancer.predictor import CLASS_DISPLAY, Predictor


# ============================================================
# CONFIGURATION
# ============================================================

MODELS = {
    "4class": "convnext_tiny_4class_224px",
    "3class": "convnext_tiny_3class_224px",
    "4class_gray": "convnext_tiny_4class_gray_224px",
}

OUT_DIR = OUTPUT_ROOT / "outputs" / "predictions"


# ============================================================
# OUTPUT
# ============================================================

def save_figure(image, result, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))

    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[1].imshow(result.overlay)
    axes[1].set_title(f"Grad-CAM ({len(result.tiles)} tile{'s' if len(result.tiles) > 1 else ''})")
    for ax in axes:
        ax.axis("off")

    lines = [f"{CLASS_DISPLAY[c]}: {p:.1%}" for c, p in
             sorted(result.probabilities.items(), key=lambda kv: -kv[1])]
    fig.suptitle(f"Prediction: {CLASS_DISPLAY[result.prediction]} ({result.confidence:.1%})",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.02, "   ·   ".join(lines) + "\nResearch prototype, not for clinical use.",
             ha="center", fontsize=9, color="#52514e")

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main(args):
    run = args.run or MODELS[args.model]
    run_dir = OUTPUT_ROOT / "runs" / run

    if not (run_dir / "best_model.pt").exists():
        raise SystemExit(f"Model not found: {run_dir / 'best_model.pt'} (download runs.zip from Kaggle)")

    print(f"Model: {run}")
    predictor = Predictor(run_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in args.images:
        image_path = Path(image_path)
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        result = predictor.predict(image)

        stem = image_path.stem
        save_figure(image, result, OUT_DIR / f"{stem}_prediction.png")
        with open(OUT_DIR / f"{stem}_prediction.json", "w") as f:
            json.dump({"image": str(image_path), "model": run, **result.to_dict()}, f, indent=2)

        print(f"\n{image_path.name}  ({image.size[0]}x{image.size[1]}, {len(result.tiles)} tile(s))")
        print(f"  Prediction: {CLASS_DISPLAY[result.prediction]}  ({result.confidence:.1%})")
        for cls, p in sorted(result.probabilities.items(), key=lambda kv: -kv[1]):
            print(f"    {CLASS_DISPLAY[cls]:<38s} {p:6.1%}")
        for w in result.warnings:
            print(f"  Note: {w}")

    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Predict oral histopathology images (research tool)")
    parser.add_argument("images", nargs="+", help="Image file(s): PNG, JPG or TIFF")
    parser.add_argument("--model", choices=list(MODELS), default="4class", help="Which trained model to use")
    parser.add_argument("--run", help="Run folder name under runs/ (overrides --model)")

    main(parser.parse_args())
