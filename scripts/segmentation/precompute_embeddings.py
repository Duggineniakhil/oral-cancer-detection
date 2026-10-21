"""
Precompute MedSAM image embeddings for the epithelium annotation set.

MedSAM's image encoder is the slow part (~4.5 min per image on a laptop
CPU, about a second on a Kaggle GPU). Its output, the image embedding,
does not depend on the boxes drawn, so it is computed once here and the
annotator (app/epithelium_annotator.py) loads it instead of recomputing.
Box prompts then take only a few seconds on the CPU.

The local medsam_hf/ weights and the Hugging Face copy are identical
(same SHA-256), so embeddings made on Kaggle match the laptop's decoder.

Usage (on Kaggle, see notebooks/medsam_embeddings_kaggle.ipynb):
    python -m scripts.segmentation.precompute_embeddings

Outputs (OUTPUT_ROOT/annotations/epithelium/embeddings/):
    <image stem>.pt   float16 tensor (1, 256, 64, 64)
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from oral_cancer.config import DATA_ROOT, OUTPUT_ROOT, resolve


# ============================================================
# CONFIGURATION
# ============================================================

SET_CSV = "annotations/epithelium/annotation_set.csv"
OUT_DIR = OUTPUT_ROOT / "annotations" / "epithelium" / "embeddings"

LOCAL_MODEL = DATA_ROOT / "medsam_hf"
HUB_MODEL = "flaviagiammarino/medsam-vit-base"


# ============================================================
# MAIN
# ============================================================

def main(args):
    from transformers import SamModel, SamProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: no GPU found; this takes several minutes per image.")

    source = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else HUB_MODEL
    print(f"Loading MedSAM from: {source}")
    processor = SamProcessor.from_pretrained(source)
    model = SamModel.from_pretrained(source).to(device).eval()

    rows = pd.read_csv(resolve(args.csv))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done = 0
    for p in tqdm(rows["image_path"], desc="Embedding"):
        out = OUT_DIR / f"{Path(p).stem}.pt"
        if out.exists() and not args.overwrite:
            continue
        with Image.open(resolve(p)) as img:
            image = img.convert("RGB")
        with torch.no_grad():
            pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
            embedding = model.get_image_embeddings(pixel_values)
        torch.save(embedding.half().cpu(), out)
        done += 1

    print(f"\nComputed {done} embeddings ({len(rows) - done} already present)")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Precompute MedSAM embeddings for the annotation set")
    parser.add_argument("--csv", default=SET_CSV, help="Annotation set CSV (relative to the data root)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing embeddings")

    main(parser.parse_args())
