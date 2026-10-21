"""
Train and evaluate an epithelium segmenter on the annotated whole images.

The 50 masks in annotations/epithelium/ (MedSAM-assisted, NOT expert-
verified) are the ground truth. A U-Net with an ImageNet ResNet-34
encoder is trained with the patient-level folds of the whole-image
pipeline, so results are cross-validated and no patient is in both
training and test:

    round k: test = fold k, validation = fold (k mod 5) + 1, train = rest

The best epoch is chosen on validation Dice; the test fold is evaluated
once with that checkpoint. Each round's model is kept, so later every
whole image can be segmented by the model that never saw its fold.

Images are processed at half resolution (2048x1536 -> 1024x768): the
segmenter needs tissue architecture (epithelium vs stroma), not single
nuclei, and the larger field of view helps. Dice and IoU are computed at
the working resolution.

Baselines on the same test images:
    tissue  every tissue pixel counts as epithelium (no segmentation)
    whole   the whole image (what the classifier sees without ROI)

Usage:
    python -m scripts.segmentation.train_segmenter
    python -m scripts.segmentation.train_segmenter --smoke-test   # tiny CPU run

Outputs (OUTPUT_ROOT/runs/segmenter/):
    round<k>/best_model.pt, round<k>/history.csv
    test_per_image.csv, summary.json, examples.png, config.json
"""

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from oral_cancer.config import OUTPUT_ROOT, PROJECT_DIR, resolve
from oral_cancer.tissue import tissue_mask


# ============================================================
# CONFIGURATION
# ============================================================

SET_CSV = "annotations/epithelium/annotation_set.csv"
LOG_CSV = "annotations/epithelium/annotations.csv"

N_FOLDS = 5
SCALE = 0.5            # working resolution relative to the original image
CROP = 512             # training crop at working resolution
CROPS_PER_IMAGE = 8    # crops per training image per epoch
EPOCHS = 60
BATCH_SIZE = 8
LR = 3e-4
THRESHOLD = 0.5
SEED = 42

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ============================================================
# DATA
# ============================================================

def annotation_file(rel):
    """Masks and the log are committed to the repo; images come from the data root."""
    in_repo = PROJECT_DIR / rel
    return in_repo if in_repo.exists() else resolve(rel)


def load_items(scale):
    """All annotated images and masks at working resolution, kept in memory."""
    rows = pd.read_csv(annotation_file(SET_CSV))
    log = pd.read_csv(annotation_file(LOG_CSV))
    rows = rows.merge(log[["image_path", "mask_path", "status"]], on="image_path")
    rows = rows[rows["status"] == "done"].reset_index(drop=True)

    items = []
    for row in rows.to_dict("records"):
        with Image.open(resolve(row["image_path"])) as im:
            im = im.convert("RGB")
            size = (round(im.width * scale), round(im.height * scale))
            image = np.asarray(im.resize(size, Image.BILINEAR))
        with Image.open(annotation_file(row["mask_path"])) as m:
            mask = np.asarray(m.convert("L").resize(size, Image.NEAREST)) > 127

        items.append({"name": row["image_path"], "class": row["class"], "source": row["source"],
                      "zoom": row["estimated_zoom"], "fold": int(row["fold"]),
                      "image": image, "mask": mask})
    return items


def to_tensor(image):
    x = (image.astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class CropDataset(Dataset):
    """Random crops with flips, 90-degree rotations and colour jitter."""

    def __init__(self, items, crop, per_image):
        self.items, self.crop, self.per_image = items, crop, per_image

    def __len__(self):
        return len(self.items) * self.per_image

    def __getitem__(self, i):
        item = self.items[i // self.per_image]
        image, mask = item["image"], item["mask"]
        h, w = mask.shape
        c = min(self.crop, h, w)
        y, x = random.randint(0, h - c), random.randint(0, w - c)
        image, mask = image[y:y + c, x:x + c], mask[y:y + c, x:x + c]

        k = random.randint(0, 3)
        image, mask = np.rot90(image, k), np.rot90(mask, k)
        if random.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]

        # Stain / brightness variation between the three sources
        image = image.astype(np.float32)
        image = image * np.random.uniform(0.85, 1.15, 3) + np.random.uniform(-15, 15, 3)
        image = np.clip(image, 0, 255).astype(np.uint8)

        return to_tensor(image), torch.from_numpy(mask.copy()).float()[None]


# ============================================================
# MODEL / EVALUATION
# ============================================================

def build_segmenter(pretrained=True):
    import segmentation_models_pytorch as smp
    return smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None, classes=1)


def dice_loss(logits, target):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return 1 - ((2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean()


@torch.no_grad()
def predict(model, image, device):
    """Probability map for a whole image (padded to a multiple of 32)."""
    model.eval()
    x = to_tensor(image)[None].to(device)
    h, w = x.shape[-2:]
    x = F.pad(x, (0, -w % 32, 0, -h % 32), mode="reflect")
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        logits = model(x)
    return torch.sigmoid(logits.float())[0, 0, :h, :w].cpu().numpy()


def scores(pred, truth):
    inter = (pred & truth).sum()
    total = pred.sum() + truth.sum()
    union = (pred | truth).sum()
    return {"dice": 2 * inter / total if total else 1.0,
            "iou": inter / union if union else 1.0}


def tissue_baseline(image):
    small = Image.fromarray(image).reduce(2)
    t = Image.fromarray(tissue_mask(small).astype(np.uint8) * 255)
    return np.asarray(t.resize((image.shape[1], image.shape[0]), Image.NEAREST)) > 127


def mean_dice(model, items, device):
    return float(np.mean([scores(predict(model, it["image"], device) > THRESHOLD, it["mask"])["dice"]
                          for it in items]))


# ============================================================
# TRAINING
# ============================================================

def train_round(k, items, args, device, out_dir):
    val_fold = k % N_FOLDS + 1
    train = [it for it in items if it["fold"] not in (k, val_fold)]
    val = [it for it in items if it["fold"] == val_fold]
    test = [it for it in items if it["fold"] == k]
    print(f"\n=== Round {k}: train {len(train)}, validation {len(val)} (fold {val_fold}), test {len(test)} ===")

    loader = DataLoader(CropDataset(train, args.crop, args.crops_per_image), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, drop_last=True)
    model = build_segmenter(pretrained=not args.smoke_test).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, LR, total_steps=args.epochs * len(loader))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    round_dir = out_dir / f"round{k}"
    round_dir.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for step, (x, y) in enumerate(loader):
            if args.smoke_test and step == 2:
                break
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(x)
            logits = logits.float()
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item()

        val_dice = mean_dice(model, val, device)
        history.append({"epoch": epoch, "train_loss": total / len(loader), "val_dice": val_dice})
        if val_dice > best:
            best = val_dice
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_dice": val_dice,
                        "scale": args.scale, "encoder": "resnet34"}, round_dir / "best_model.pt")
        print(f"epoch {epoch:3d}  loss {total / len(loader):.4f}  val Dice {val_dice:.4f}"
              f"{'  *' if val_dice == best else ''}  ({time.time() - t0:.0f}s)")

    pd.DataFrame(history).to_csv(round_dir / "history.csv", index=False)

    # Test once with the best checkpoint
    model.load_state_dict(torch.load(round_dir / "best_model.pt", map_location=device)["state_dict"])
    rows = []
    for it in test:
        prob = predict(model, it["image"], device)
        truth = it["mask"]
        row = {"image_path": it["name"], "class": it["class"], "source": it["source"],
               "zoom": it["zoom"], "fold": k, "epithelium_fraction": float(truth.mean())}
        for method, pred in [("segmenter", prob > THRESHOLD), ("tissue", tissue_baseline(it["image"])),
                             ("whole", np.ones_like(truth))]:
            for metric, v in scores(pred, truth).items():
                row[f"{method}_{metric}"] = float(v)
        rows.append(row)
        it["prediction"] = prob > THRESHOLD
    return rows


# ============================================================
# REPORTING
# ============================================================

def summarize(df):
    out = {}
    for method in ("segmenter", "tissue", "whole"):
        per_fold = df.groupby("fold")[f"{method}_dice"].mean()
        out[method] = {
            "dice_mean": round(df[f"{method}_dice"].mean(), 4),
            "iou_mean": round(df[f"{method}_iou"].mean(), 4),
            "dice_fold_mean": round(per_fold.mean(), 4),
            "dice_fold_sd": round(per_fold.std(ddof=1), 4) if len(per_fold) > 1 else 0.0,
        }
        # 40x images are mostly epithelium, so "whole" is already close there;
        # 10x images (stroma, glass) are where segmentation has to earn its keep
        out[method]["dice_by_zoom"] = df.groupby("zoom")[f"{method}_dice"].mean().round(4).to_dict()
        out[method]["dice_by_source"] = df.groupby("source")[f"{method}_dice"].mean().round(4).to_dict()
    out["n_images"] = len(df)
    out["note"] = "Ground truth is MedSAM-assisted (Claude), not expert-verified."
    return out


def save_examples(items, path, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [it for it in items if "prediction" in it]
    # Prefer 10x images: that is where stroma and glass are present
    shown = sorted(shown, key=lambda it: (it["zoom"] != "10x", it["fold"]))[:n]
    fig, axes = plt.subplots(len(shown), 3, figsize=(10, 2.6 * len(shown)))
    for ax_row, it in zip(np.atleast_2d(axes), shown):
        d = scores(it["prediction"], it["mask"])["dice"]
        for ax, img, title in zip(ax_row, [it["image"], it["mask"], it["prediction"]],
                                  [f"{it['class']} ({it['zoom']})", "annotation", f"segmenter, Dice {d:.2f}"]):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--crop", type=int, default=CROP)
    parser.add_argument("--crops-per-image", type=int, default=CROPS_PER_IMAGE)
    parser.add_argument("--scale", type=float, default=SCALE)
    parser.add_argument("--rounds", type=int, nargs="*", default=list(range(1, N_FOLDS + 1)))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true", help="1 round, 2 epochs, tiny crops, CPU-friendly")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs, args.crop, args.crops_per_image, args.batch_size = 2, 128, 1, 2
        args.scale, args.rounds, args.workers = 0.125, [1], 0

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    out_dir = OUTPUT_ROOT / "runs" / ("segmenter_smoke" if args.smoke_test else "segmenter")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    items = load_items(args.scale)
    print(f"Loaded {len(items)} annotated images at scale {args.scale}")

    rows = []
    for k in args.rounds:
        rows += train_round(k, items, args, device, out_dir)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test_per_image.csv", index=False)
    summary = summarize(df)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    save_examples(items, out_dir / "examples.png")

    print("\n=== Test Dice (cross-validated, per image) ===")
    for method in ("segmenter", "tissue", "whole"):
        s = summary[method]
        print(f"{method:10s} Dice {s['dice_mean']:.3f}  IoU {s['iou_mean']:.3f}  "
              f"(fold mean {s['dice_fold_mean']:.3f} +/- {s['dice_fold_sd']:.3f})  by zoom {s['dice_by_zoom']}")
    print("Outputs in", out_dir)


if __name__ == "__main__":
    main()
