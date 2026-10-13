"""
Grad-CAM heatmaps for a trained ConvNeXt-Tiny run.

Shows which regions of each patch drove the prediction. Images are
sampled at random (fixed seed) from the test predictions, so the
examples are not cherry-picked:
    - N correctly classified images per class
    - up to M misclassified images

Usage:
    python gradcam.py                                  # 4-class run
    python gradcam.py --run convnext_tiny_4class_gray_224px
    python gradcam.py --per-class 8 --errors 16
    python gradcam.py --border-check                   # do heatmaps favour patch edges?

Outputs (OUTPUT_ROOT/runs/<run>/gradcam/):
    correct_<class>.png   original + heatmap for correct predictions
    errors.png            misclassified examples
    gradcam_samples.csv   which images were used
    border_check_*.csv    (with --border-check)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from config import OUTPUT_ROOT, resolve
from train_classifier import CLASSES_3, CLASSES_4, build_model, build_transforms


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
HEATMAP_ALPHA = 0.45


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) on a convolutional layer."""

    def __init__(self, model, layer):
        self.model = model
        self.activations = None
        self.gradients = None
        layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def __call__(self, image, target_class=None):
        """Returns (heatmap in [0, 1] at input size, class probabilities)."""
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = self.model(image)
            probs = torch.softmax(logits, dim=1)[0].detach()

            if target_class is None:
                target_class = int(probs.argmax())

            logits[0, target_class].backward()

        # Channel weights = spatially averaged gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam.cpu().numpy(), probs.cpu().numpy()

    def raw_batch(self, images):
        """Un-upsampled CAMs (B, h, w) for the predicted class of each image.

        ConvNeXt uses LayerNorm (per-sample), so backpropagating the sum
        of each image's own target logit gives per-image gradients.
        """
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = self.model(images)
            preds = logits.argmax(dim=1)
            logits.gather(1, preds[:, None]).sum().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cams = F.relu((weights * self.activations).sum(dim=1)).detach()

        return cams.cpu().numpy(), preds.cpu().numpy()


# ============================================================
# BORDER CHECK
# ============================================================

def border_check(cam_model, eval_tf, class_names, run_dir, device, batch_size=16):
    """
    Share of Grad-CAM mass in the outer ring of the CAM grid, for every
    test image. A heatmap spread evenly over the patch puts
    (outer cells / all cells) of its mass on the border (24/49 = 49% for
    a 7x7 grid). Much higher values mean the model looks at patch edges.

    Normal tiles are also split by whether the tile lay on the edge of
    its original microscope image (where lighting fall-off/artefacts
    are likely) or in its interior.
    """
    preds = pd.read_csv(run_dir / "test_predictions.csv")

    # Tile positions come from the dataset CSV
    meta = pd.read_csv(resolve("splits/normalized_dataset.csv"))
    meta = meta[["image_path", "tile_row", "tile_col", "original_width", "original_height"]]
    preds = preds.merge(meta, on="image_path", how="left")

    border_share, peak_on_border = [], []

    for start in range(0, len(preds), batch_size):
        rows = preds.iloc[start:start + batch_size]
        batch = []
        for p in rows["image_path"]:
            with Image.open(resolve(p)) as img:
                batch.append(eval_tf(img.convert("RGB")))

        cams, _ = cam_model.raw_batch(torch.stack(batch).to(device))

        for cam in cams:
            ring = np.ones_like(cam, dtype=bool)
            ring[1:-1, 1:-1] = False
            total = cam.sum() + 1e-8
            border_share.append(cam[ring].sum() / total)
            peak_on_border.append(bool(ring.ravel()[cam.argmax()]))

        print(f"  {min(start + batch_size, len(preds))}/{len(preds)}", end="\r")

    h, w = cams.shape[1:]
    uniform = 1 - (h - 2) * (w - 2) / (h * w)

    preds["cam_border_share"] = border_share
    preds["cam_peak_on_border"] = peak_on_border

    # Normal tiles on the edge of their source image vs interior
    is_normal = preds["class"] == "Normal"
    n_rows = preds["original_height"] // 512
    n_cols = preds["original_width"] // 512
    on_edge = (
        (preds["tile_row"] == 0) | (preds["tile_col"] == 0)
        | (preds["tile_row"] == n_rows - 1) | (preds["tile_col"] == n_cols - 1)
    )
    preds["group"] = preds["class"]
    preds.loc[is_normal & on_edge, "group"] = "Normal (edge of source image)"
    preds.loc[is_normal & ~on_edge, "group"] = "Normal (interior of source image)"

    summary = (
        preds.groupby("group")
        .agg(
            images=("cam_border_share", "size"),
            border_share_mean=("cam_border_share", "mean"),
            border_share_median=("cam_border_share", "median"),
            peak_on_border=("cam_peak_on_border", "mean"),
        )
        .round(3)
    )

    out_dir = run_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_dir / "border_check_per_image.csv", index=False)
    summary.to_csv(out_dir / "border_check_summary.csv")

    print("\n\n========================================")
    print("GRAD-CAM BORDER CHECK (all test images)")
    print("========================================")
    print(f"CAM grid: {h}x{w}. Expected if heatmaps were spread evenly: "
          f"border share {uniform:.3f}, peak on border {uniform:.3f}\n")
    print(summary.to_string())
    print(f"\nSaved to: {out_dir}")


# ============================================================
# PLOTTING
# ============================================================

def short(name):
    return name.replace("Leukoplakia_", "Leuk. ").replace("_", " ")


def plot_grid(samples, title, out_path):
    """Two columns per sample: image, image + heatmap."""
    n = len(samples)
    cols = 4  # samples per row
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 2.6, rows * 3.0))
    axes = np.atleast_2d(axes)

    for ax in axes.ravel():
        ax.axis("off")

    for k, s in enumerate(samples):
        r, c = divmod(k, cols)
        ax_img, ax_cam = axes[r, 2 * c], axes[r, 2 * c + 1]

        ax_img.imshow(s["image"])
        ax_img.set_title(f"true: {short(s['true'])}", fontsize=8)

        ax_cam.imshow(s["image"])
        ax_cam.imshow(s["cam"], cmap="jet", alpha=HEATMAP_ALPHA, vmin=0, vmax=1)
        ax_cam.set_title(f"pred: {short(s['pred'])} ({s['prob']:.2f})", fontsize=8,
                         color="black" if s["true"] == s["pred"] else "red")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main(args):

    run_dir = OUTPUT_ROOT / "runs" / args.run
    cfg = json.load(open(run_dir / "config.json"))

    class_names = CLASSES_4 if cfg["classes"] == 4 else CLASSES_3
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run: {args.run} | device: {device}")

    model = build_model(len(class_names), pretrained=False)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.to(device).eval()

    # Last ConvNeXt stage: 768 channels at 1/32 resolution (7x7 for 224px)
    cam_model = GradCAM(model, model.features[-1])

    _, eval_tf = build_transforms(cfg["img_size"], cfg["grayscale"])

    if args.border_check:
        border_check(cam_model, eval_tf, class_names, run_dir, device)
        return

    # --------------------------------------------------------
    # Select images (random, fixed seed)
    # --------------------------------------------------------
    preds = pd.read_csv(run_dir / "test_predictions.csv")
    preds["correct"] = preds["class"] == preds["predicted"]

    correct = (
        preds[preds["correct"]]
        .groupby("class")
        .sample(n=args.per_class, random_state=RANDOM_SEED)
    )
    errors = preds[~preds["correct"]]
    errors = errors.sample(n=min(args.errors, len(errors)), random_state=RANDOM_SEED)

    print(f"Test errors in this run: {(~preds['correct']).sum()} / {len(preds)}")

    out_dir = run_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Compute heatmaps
    # --------------------------------------------------------
    def compute(rows):
        samples = []
        for _, row in rows.iterrows():
            with Image.open(resolve(row["image_path"])) as img:
                img = img.convert("RGB")
                tensor = eval_tf(img).unsqueeze(0).to(device)
                shown = img.resize((cfg["img_size"], cfg["img_size"]))
                if cfg["grayscale"]:
                    shown = shown.convert("L").convert("RGB")

            cam, probs = cam_model(tensor)
            pred_idx = int(probs.argmax())

            samples.append({
                "image": np.asarray(shown),
                "cam": cam,
                "true": row["class"],
                "pred": class_names[pred_idx],
                "prob": float(probs[pred_idx]),
                "image_path": row["image_path"],
            })
        return samples

    records = []

    for class_name in class_names:
        rows = correct[correct["class"] == class_name]
        samples = compute(rows)
        path = out_dir / f"correct_{class_name}.png"
        plot_grid(samples, f"Grad-CAM — correctly classified: {short(class_name)}", path)
        records += [{"kind": "correct", **{k: s[k] for k in ("image_path", "true", "pred", "prob")}} for s in samples]
        print(f"  saved {path.name}")

    if len(errors):
        samples = compute(errors)
        path = out_dir / "errors.png"
        plot_grid(samples, "Grad-CAM — misclassified examples (red = wrong prediction)", path)
        records += [{"kind": "error", **{k: s[k] for k in ("image_path", "true", "pred", "prob")}} for s in samples]
        print(f"  saved {path.name}")

    pd.DataFrame(records).to_csv(out_dir / "gradcam_samples.csv", index=False)
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Grad-CAM heatmaps for a trained run")
    parser.add_argument("--run", default="convnext_tiny_4class_224px", help="Folder name under runs/")
    parser.add_argument("--per-class", type=int, default=8, help="Correct examples per class")
    parser.add_argument("--errors", type=int, default=12, help="Misclassified examples")
    parser.add_argument("--border-check", action="store_true",
                        help="Measure how much of each heatmap lies on the patch border (all test images)")

    main(parser.parse_args())
