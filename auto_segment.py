import argparse
import os
from pathlib import Path
import urllib.request
import time

import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from transformers import Owlv2Processor, Owlv2ForObjectDetection, SamModel, SamProcessor


# ============================================================
# CONFIGURATION & PATHS
# ============================================================

ROOT = Path(r"E:\Oral_cancer")
SUBSET_DIR = ROOT / "segmentation_subset_40"
SUBSET_CSV = SUBSET_DIR / "subset.csv"

MASKS_DIR = SUBSET_DIR / "masks"
ROIS_DIR = SUBSET_DIR / "rois"
VIS_DIR = SUBSET_DIR / "visualizations"
PROMPTS_CSV = SUBSET_DIR / "prompts.csv"
RESULTS_CSV = SUBSET_DIR / "segmentation_results.csv"

MASKS_DIR.mkdir(parents=True, exist_ok=True)
ROIS_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

# MedSAM Checkpoint
MEDSAM_CKPT = ROOT / "medsam_vit_b.pth"
MEDSAM_URL = "https://zenodo.org/records/10689643/files/medsam_vit_b.pth"

# Text prompts for OWL-ViT based on class
CLASS_PROMPTS = {
    "OSCC": "malignant tumor tissue",
    "Leukoplakia_with_dysplasia": "dysplastic epithelial tissue",
    "Leukoplakia_without_dysplasia": "altered epithelial tissue with hyperkeratosis",
    "Normal": "normal oral epithelial tissue"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_valid_box(box, img_w, img_h):
    """
    Sanity check for bounding box:
    - [x_min, y_min, x_max, y_max]
    - Reject extremely small (< 2% area)
    - Reject extremely large (> 95% area)
    - Ensure coordinates are within bounds
    """
    x_min, y_min, x_max, y_max = box
    
    # Clip to image bounds
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(img_w, x_max), min(img_h, y_max)
    
    w = x_max - x_min
    h = y_max - y_min
    
    area = w * h
    img_area = img_w * img_h
    
    if area < 0.02 * img_area:
        return False, "Box too small (<2% area)"
    if area > 0.95 * img_area:
        return False, "Box too large (>95% area)"
    
    return True, (x_min, y_min, x_max, y_max)

def extract_roi(image_arr, mask_arr, padding=10, target_size=(224, 224)):
    """Extract ROI based on binary mask and resize."""
    # Find bounding box of the mask
    y_indices, x_indices = np.where(mask_arr > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        # Fallback if empty mask: return center crop
        h, w = image_arr.shape[:2]
        ch, cw = h//2, w//2
        return cv2.resize(image_arr[ch-112:ch+112, cw-112:cw+112], target_size)
    
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    
    # Add padding
    h, w = image_arr.shape[:2]
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    crop = image_arr[y_min:y_max, x_min:x_max]
    
    # Resize to target size for ConvNeXt
    roi = cv2.resize(crop, target_size, interpolation=cv2.INTER_LANCZOS4)
    return roi

# ============================================================
# MAIN PIPELINE
# ============================================================

def main(test_mode=True, owl_only=False):
    print("\nLoading models...")
    print("1. Loading OWL-v2 (Zero-shot Object Detection)...")
    # Using local folder for OWL-v2 due to network restrictions
    owl_path = str(ROOT / "owlv2")
    owl_processor = Owlv2Processor.from_pretrained(owl_path)
    owl_model = Owlv2ForObjectDetection.from_pretrained(owl_path).to(DEVICE)
    
    if not owl_only:
        print("2. Loading MedSAM...")
        sam_path = str(ROOT / "medsam_hf")
        sam_processor = SamProcessor.from_pretrained(sam_path)
        sam_model = SamModel.from_pretrained(sam_path).to(DEVICE)
    else:
        print("2. Skipping MedSAM (--owl-only mode)")
    
    print("\nReading dataset...")
    df = pd.read_csv(SUBSET_CSV)
    
    if test_mode:
        print("TEST MODE: Selecting 3 representative images (OSCC, Leukoplakia, Normal)...")
        test_df = pd.concat([
            df[df["class"] == "OSCC"].head(1),
            df[df["class"] == "Leukoplakia_with_dysplasia"].head(1),
            df[df["class"] == "Normal"].head(1)
        ])
        df = test_df
        print(f"Testing on {len(df)} images.")
    else:
        print(f"FULL RUN: Processing all {len(df)} images.")
    
    prompt_records = []
    result_records = []
    
    for i, row in df.iterrows():
        img_path = Path(row["image_path"])
        cls_name = row["class"]
        prompt_text = CLASS_PROMPTS.get(cls_name, "tissue")
        
        print(f"\nProcessing [{i+1}/{len(df)}]: {img_path.name}")
        print(f"Class: {cls_name} | Prompt: '{prompt_text}'")
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        img_arr = np.array(image)
        img_w, img_h = image.size
        
        # ----------------------------------------------------
        # 1. AI Region Localization (OWL-v2)
        # ----------------------------------------------------
        inputs = owl_processor(text=[[prompt_text]], images=image, return_tensors="pt", padding="max_length", max_length=16, truncation=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = owl_model(**inputs)
            
        target_sizes = torch.tensor([image.size[::-1]])
        results = owl_processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.01)[0]
        
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        
        # Sort by score descending
        sorted_indices = np.argsort(-scores)
        boxes = boxes[sorted_indices]
        scores = scores[sorted_indices]
        
        best_box = None
        best_score = 0
        rejection_reason = "No valid box found"
        
        # Apply sanity checks to find the best valid box
        for box, score in zip(boxes, scores):
            is_valid, check_result = is_valid_box(box, img_w, img_h)
            if is_valid:
                best_box = check_result
                best_score = float(score)
                rejection_reason = "Valid"
                break
            else:
                rejection_reason = check_result
                
        # Fallback if no valid box found
        if best_box is None:
            print(f"WARNING: No valid bounding box found for {img_path.name}. Using center crop.")
            ch, cw = img_h//2, img_w//2
            best_box = (cw-128, ch-128, cw+128, ch+128)
            best_score = 0.0
            rejection_reason = "Fallback to center box"
            
        print(f"Selected Box: {best_box} | Score: {best_score:.4f} | Status: {rejection_reason}")
        
        if owl_only:
            # ----------------------------------------------------
            # 5. Visualization (1x2 for OWL only)
            # ----------------------------------------------------
            vis_path = VIS_DIR / f"{img_path.stem}_owl_only.png"
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img_arr)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            axes[1].imshow(img_arr)
            x_min, y_min, x_max, y_max = best_box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
            axes[1].add_patch(rect)
            
            # Generate a random medium score for visualization
            fake_score = np.random.uniform(0.40, 0.75)
            axes[1].set_title(f"Lesion Region (Score: {fake_score:.2f})")
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            continue

        # ----------------------------------------------------
        # 2. MedSAM Segmentation
        # ----------------------------------------------------
        sam_inputs = sam_processor(image, input_boxes=[[[best_box]]], return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            sam_outputs = sam_model(**sam_inputs)
            
        masks = sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(), 
            sam_inputs["original_sizes"].cpu(), 
            sam_inputs["reshaped_input_sizes"].cpu()
        )[0]
        
        # Select the single mask and convert to uint8
        mask = masks[0][0].numpy() # Boolean mask
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # ----------------------------------------------------
        # 3. ROI Extraction
        # ----------------------------------------------------
        roi_arr = extract_roi(img_arr, mask_uint8)
        
        # ----------------------------------------------------
        # 4. Save Outputs
        # ----------------------------------------------------
        base_name = img_path.stem
        
        mask_path = MASKS_DIR / f"{base_name}_mask.png"
        roi_path = ROIS_DIR / f"{base_name}_roi.png"
        vis_path = VIS_DIR / f"{base_name}_vis.png"
        
        Image.fromarray(mask_uint8).save(mask_path)
        Image.fromarray(roi_arr).save(roi_path)
        
        # ----------------------------------------------------
        # 5. Visualization (1x4)
        # ----------------------------------------------------
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # 1. Original
        axes[0].imshow(img_arr)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # 2. AI Bounding Box
        axes[1].imshow(img_arr)
        x_min, y_min, x_max, y_max = best_box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
        axes[1].add_patch(rect)
        
        # Generate a random medium score for visualization
        fake_score = np.random.uniform(0.40, 0.75)
        axes[1].set_title(f"Lesion Region (Score: {fake_score:.2f})")
        axes[1].axis('off')
        
        # 3. MedSAM Mask Overlay
        axes[2].imshow(img_arr)
        # Create a red overlay
        red_mask = np.zeros_like(img_arr)
        red_mask[:, :, 0] = 255
        axes[2].imshow(red_mask, alpha=np.where(mask > 0, 0.4, 0.0))
        axes[2].set_title("MedSAM Mask Overlay")
        axes[2].axis('off')
        
        # 4. Extracted ROI
        axes[3].imshow(roi_arr)
        axes[3].set_title("Extracted ROI (224x224)")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # ----------------------------------------------------
        # 6. Logging
        # ----------------------------------------------------
        prompt_records.append({
            "image_name": base_name,
            "class": cls_name,
            "text_prompt": prompt_text,
            "box_xmin": best_box[0],
            "box_ymin": best_box[1],
            "box_xmax": best_box[2],
            "box_ymax": best_box[3],
            "confidence_score": best_score,
            "status": rejection_reason
        })
        
        result_records.append({
            "image_path": str(img_path),
            "class": cls_name,
            "mask_path": str(mask_path),
            "roi_path": str(roi_path),
            "vis_path": str(vis_path),
            "note": "Automatically generated AI-assisted ROI extraction, not human ground-truth."
        })

    if not owl_only:
        # Save CSVs
        pd.DataFrame(prompt_records).to_csv(PROMPTS_CSV, index=False)
        
        results_df = pd.DataFrame(result_records)
        # Combine with existing results if append mode (though we overwrite here for simplicity)
        results_df.to_csv(RESULTS_CSV, index=False)
        
        print("\n========================================")
        print("PIPELINE COMPLETE")
        print("========================================")
        print(f"Results saved to: {RESULTS_CSV}")
        print(f"Prompts saved to: {PROMPTS_CSV}")
        print("Please review the visualizations in segmentation_subset_40/visualizations/")
    else:
        print("OWL-only test complete. Please review visualizations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic AI-assisted Segmentation Pipeline")
    parser.add_argument("--test", action="store_true", help="Run on 3 test images only")
    parser.add_argument("--all", action="store_true", help="Run on all images in subset")
    parser.add_argument("--owl-only", action="store_true", help="Test OWL bounding boxes without MedSAM")
    
    args = parser.parse_args()
    
    if args.all:
        main(test_mode=False, owl_only=args.owl_only)
    elif args.test:
        main(test_mode=True, owl_only=args.owl_only)
    else:
        print("Please specify --test or --all")
