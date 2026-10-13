"""
Train and evaluate ConvNeXt-Tiny on whole (stain-normalized) patches.

Splits come from the CSV: the model trains on "train", the best epoch
is chosen on "validation" (macro-F1), and "test" is evaluated ONCE
with that checkpoint at the end.

Usage:
    python -m scripts.classification.train_classifier                 # 4 classes
    python -m scripts.classification.train_classifier --classes 3     # NDB-UFES only
    python -m scripts.classification.train_classifier --grayscale     # colour-shortcut check
    python -m scripts.classification.train_classifier --smoke-test    # tiny CPU run

Outputs (OUTPUT_ROOT/runs/<run name>/):
    best_model.pt, history.csv, curves.png, test_predictions.csv,
    metrics.json, confusion_matrix.png, config.json
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from oral_cancer.config import OUTPUT_ROOT, resolve


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CSV = "splits/normalized_dataset.csv"

CLASSES_4 = [
    "OSCC",
    "Leukoplakia_with_dysplasia",
    "Leukoplakia_without_dysplasia",
    "Normal",
]
CLASSES_3 = CLASSES_4[:3]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# DATA
# ============================================================

class PatchDataset(Dataset):

    def __init__(self, df, class_to_idx, transform):
        self.paths = [resolve(p) for p in df["image_path"]]
        self.labels = [class_to_idx[c] for c in df["class"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            image = self.transform(img.convert("RGB"))
        return image, self.labels[i]


class RandomRotate90:
    """Histology has no canonical orientation."""

    def __call__(self, img):
        return img.rotate(90 * random.randint(0, 3))


def build_transforms(img_size, grayscale):
    gray = [transforms.Grayscale(num_output_channels=3)] if grayscale else []

    train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRotate90(),
        # Strong colour augmentation so the model cannot rely on the
        # residual stain differences between the two source datasets
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        *gray,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    evaluate = transforms.Compose([
        transforms.Resize(img_size),
        *gray,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train, evaluate


# ============================================================
# MODEL
# ============================================================

def build_model(num_classes, pretrained):
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


# ============================================================
# TRAIN / EVALUATE
# ============================================================

def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None, scaler=None):
    training = optimizer is not None
    model.train(training)

    total_loss, all_probs, all_labels = 0.0, [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training), \
             torch.autocast(device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        total_loss += loss.item() * len(labels)
        all_probs.append(torch.softmax(logits.float(), dim=1).detach().cpu())
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    return total_loss / len(labels), probs, labels


def summarize(labels, probs, class_names):
    preds = probs.argmax(axis=1)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

    try:
        metrics["macro_auc_ovr"] = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        metrics["macro_auc_ovr"] = float("nan")

    metrics["per_class"] = classification_report(
        labels, preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return metrics


def save_plots(history, cm, class_names, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["val_macro_f1"], label="macro-F1")
    axes[1].plot(history["epoch"], history["val_balanced_accuracy"], label="balanced accuracy")
    axes[1].set_title("Validation")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(run_dir / "curves.png", dpi=120)
    plt.close(fig)

    short = [c.replace("Leukoplakia_", "Leuk. ").replace("_", " ") for c in class_names]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(short)), short, rotation=30, ha="right")
    ax.set_yticks(range(len(short)), short)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(short)):
        for j in range(len(short)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Test confusion matrix")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_names = CLASSES_4 if args.classes == 4 else CLASSES_3
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    run_name = args.run_name or (
        f"convnext_tiny_{args.classes}class"
        + ("_gray" if args.grayscale else "")
        + f"_{args.img_size}px"
    )
    run_dir = OUTPUT_ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    df = pd.read_csv(resolve(args.csv))
    df = df[df["class"].isin(class_names)]

    if args.smoke_test:
        df = df.groupby(["split", "class"]).head(4)

    splits = {s: df[df["split"] == s].reset_index(drop=True) for s in ["train", "validation", "test"]}

    print(f"\nRun: {run_name}")
    print(pd.crosstab(df["class"], df["split"]).reindex(class_names).to_string())

    train_tf, eval_tf = build_transforms(args.img_size, args.grayscale)

    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=(device.type == "cuda"))
    loaders = {
        "train": DataLoader(PatchDataset(splits["train"], class_to_idx, train_tf), shuffle=True, drop_last=True, **loader_args),
        "validation": DataLoader(PatchDataset(splits["validation"], class_to_idx, eval_tf), **loader_args),
        "test": DataLoader(PatchDataset(splits["test"], class_to_idx, eval_tf), **loader_args),
    }

    # Inverse-frequency class weights (training split only)
    counts = splits["train"]["class"].value_counts().reindex(class_names).values.astype(float)
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=device)
    print("\nClass weights:", dict(zip(class_names, weights.cpu().numpy().round(2))))

    # --------------------------------------------------------
    # Model / optimisation
    # --------------------------------------------------------
    model = build_model(len(class_names), pretrained=not args.smoke_test).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(loaders["train"]))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.epochs * steps_per_epoch,
        pct_start=min(0.3, 1 / max(args.epochs, 1)),
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # --------------------------------------------------------
    # Training loop (model selection on validation macro-F1)
    # --------------------------------------------------------
    history = []
    best_f1, best_epoch, epochs_without_improvement = -1.0, 0, 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, _, _ = run_epoch(model, loaders["train"], device, criterion, optimizer, scheduler, scaler)
        val_loss, val_probs, val_labels = run_epoch(model, loaders["validation"], device, criterion)
        val = summarize(val_labels, val_probs, class_names)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val["accuracy"],
            "val_balanced_accuracy": val["balanced_accuracy"],
            "val_macro_f1": val["macro_f1"],
        })

        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, epochs_without_improvement = val["macro_f1"], epoch, 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:2d}/{args.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
            f"val bal-acc {val['balanced_accuracy']:.4f} | val macro-F1 {val['macro_f1']:.4f}"
            f"{'  *best*' if improved else ''} | {time.time() - start:.0f}s"
        )

        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs")
            break

    # --------------------------------------------------------
    # Test (once, best checkpoint)
    # --------------------------------------------------------
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    _, test_probs, test_labels = run_epoch(model, loaders["test"], device, criterion)
    test = summarize(test_labels, test_probs, class_names)

    cm = confusion_matrix(test_labels, test_probs.argmax(axis=1), labels=list(range(len(class_names))))

    predictions = splits["test"][["image_path", "class", "source"]].copy()
    predictions["predicted"] = [class_names[i] for i in test_probs.argmax(axis=1)]
    for i, c in enumerate(class_names):
        predictions[f"prob_{c}"] = test_probs[:, i]
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)

    # For 4-class runs, also report the 3 NDB-UFES classes on their own
    # (comparable with published NDB-UFES results)
    ndb = predictions["source"] == "NDB-UFES"
    if args.classes == 4 and ndb.any():
        ndb_labels = test_labels[ndb.values]
        ndb_preds = test_probs[ndb.values].argmax(axis=1)
        test["ndb_ufes_subset"] = {
            "balanced_accuracy": balanced_accuracy_score(ndb_labels, ndb_preds),
            "macro_f1": f1_score(ndb_labels, ndb_preds, labels=[0, 1, 2], average="macro", zero_division=0),
            "predicted_as_normal": int((ndb_preds == class_to_idx["Normal"]).sum()),
        }

    metrics = {"run": run_name, "best_epoch": best_epoch, "best_val_macro_f1": best_f1, "test": test}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    save_plots(pd.DataFrame(history), cm, class_names, run_dir)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------
    print("\n========================================")
    print(f"TEST RESULTS — {run_name} (best epoch {best_epoch})")
    print("========================================")
    print(f"Accuracy:          {test['accuracy']:.4f}")
    print(f"Balanced accuracy: {test['balanced_accuracy']:.4f}")
    print(f"Macro-F1:          {test['macro_f1']:.4f}")
    print(f"Macro AUC (OvR):   {test['macro_auc_ovr']:.4f}")

    if "ndb_ufes_subset" in test:
        sub = test["ndb_ufes_subset"]
        print(f"\nNDB-UFES test patches only: balanced acc {sub['balanced_accuracy']:.4f}, "
              f"macro-F1 {sub['macro_f1']:.4f}, wrongly called Normal: {sub['predicted_as_normal']}")

    print("\nPer class:")
    print(classification_report(test_labels, test_probs.argmax(axis=1),
                                labels=list(range(len(class_names))),
                                target_names=class_names, digits=4, zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())
    print(f"\nSaved to: {run_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train ConvNeXt-Tiny on oral histopathology patches")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Input CSV (relative to the data root)")
    parser.add_argument("--classes", type=int, choices=[3, 4], default=4,
                        help="4 = with Normal; 3 = NDB-UFES classes only")
    parser.add_argument("--grayscale", action="store_true", help="Train/evaluate on greyscale images")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6, help="Early-stopping patience (epochs)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", help="Output folder name (default derived from settings)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny run (4 images per class/split, no pretrained weights) to check the code")

    args = parser.parse_args()

    if args.smoke_test:
        args.epochs, args.img_size, args.batch_size, args.workers = 1, 64, 8, 0
        args.run_name = args.run_name or "smoke_test"

    main(args)
