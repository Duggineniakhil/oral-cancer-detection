"""
Prediction for a single histopathology image: class probabilities and
a Grad-CAM heatmap.

The image goes through the same steps as the training data:
    1. Images larger than one patch are cut into 512x512 tiles (never
       resized, so cell size matches training); tiles with <60% tissue
       are skipped. A patch-sized image is used as a single tile.
    2. Each tile is Macenko stain-normalized with the training
       reference.
    3. Each tile is resized to the model input (224 px) and classified;
       Grad-CAM shows which regions drove the prediction.
    4. Tile probabilities are averaged into one image-level prediction,
       and tile heatmaps are stitched into one overlay.

RESEARCH TOOL ONLY: not validated for diagnosis. See README limitations
(patch-level folds, cross-dataset Normal class).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image

from oral_cancer.model import CLASSES_3, CLASSES_4, GradCAM, build_model, build_transforms
from oral_cancer.stain import DEFAULT_REFERENCE, load_reference, normalize
from oral_cancer.tissue import MIN_TISSUE_FRACTION, TILE_SIZE, iter_tiles, tissue_mask


# ============================================================
# CONFIGURATION
# ============================================================

# Images up to this size are treated as one patch rather than tiled
SINGLE_PATCH_MAX = int(TILE_SIZE * 1.5)

HEATMAP_ALPHA = 0.45

CLASS_DISPLAY = {
    "OSCC": "OSCC (oral squamous cell carcinoma)",
    "Leukoplakia_with_dysplasia": "Leukoplakia with dysplasia",
    "Leukoplakia_without_dysplasia": "Leukoplakia without dysplasia",
    "Normal": "Normal mucosa",
}


# ============================================================
# RESULT
# ============================================================

@dataclass
class TileResult:
    row: int
    col: int
    box: tuple
    probabilities: dict
    prediction: str
    stain_normalized: bool


@dataclass
class PredictionResult:
    prediction: str
    confidence: float
    probabilities: dict
    tiles: list
    overlay: Image.Image
    normalized: Image.Image
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "tiles_used": len(self.tiles),
            "tiles": [
                {
                    "row": t.row,
                    "col": t.col,
                    "prediction": t.prediction,
                    "probabilities": t.probabilities,
                    "stain_normalized": t.stain_normalized,
                }
                for t in self.tiles
            ],
            "warnings": self.warnings,
        }


# ============================================================
# PREDICTOR
# ============================================================

class Predictor:
    """Loads one trained run (runs/<name>/) and predicts single images."""

    def __init__(self, run_dir, device=None, stain_reference=DEFAULT_REFERENCE):
        run_dir = Path(run_dir)
        self.config = json.load(open(run_dir / "config.json"))
        self.classes = CLASSES_4 if self.config["classes"] == 4 else CLASSES_3

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        model = build_model(len(self.classes), pretrained=False)
        state = torch.load(run_dir / "best_model.pt", map_location=self.device)
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()

        # Last ConvNeXt stage: 7x7 maps at 224 px input
        self.cam = GradCAM(self.model, self.model.features[-1])
        _, self.transform = build_transforms(self.config["img_size"], self.config["grayscale"])

        self.ref_stains, self.ref_max_c = load_reference(stain_reference)

    # --------------------------------------------------------
    # Tiling
    # --------------------------------------------------------
    def _tiles(self, img, warnings):
        w, h = img.size

        if w <= SINGLE_PATCH_MAX and h <= SINGLE_PATCH_MAX:
            if (w, h) != (TILE_SIZE, TILE_SIZE):
                warnings.append(
                    f"Image is {w}x{h}; the model was trained on 512x512 patches "
                    "(10x/40x objective). Other sizes or magnifications may reduce accuracy."
                )
            return [(0, 0, (0, 0, w, h))]

        mask = tissue_mask(img)
        tiles, fractions = [], []
        for r, c, box in iter_tiles(img):
            x0, y0, x1, y1 = box
            fraction = float(mask[y0:y1, x0:x1].mean())
            fractions.append((fraction, (r, c, box)))
            if fraction >= MIN_TISSUE_FRACTION:
                tiles.append((r, c, box))

        if not tiles:
            best = max(fractions, key=lambda f: f[0])
            warnings.append(
                f"No 512x512 tile has >= {MIN_TISSUE_FRACTION:.0%} tissue; "
                f"using the tile with the most tissue ({best[0]:.0%})."
            )
            tiles = [best[1]]

        skipped = (h // TILE_SIZE) * (w // TILE_SIZE) - len(tiles)
        if skipped:
            warnings.append(f"{skipped} tile(s) with too little tissue were skipped.")

        return tiles

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------
    def predict(self, image):
        img = image.convert("RGB")
        w, h = img.size
        warnings = []

        heat = np.full((h, w), np.nan, dtype=np.float32)
        normalized_canvas = np.asarray(img).copy()
        tile_results = []

        for r, c, box in self._tiles(img, warnings):
            x0, y0, x1, y1 = box
            rgb = np.asarray(img.crop(box))

            normalized = normalize(rgb, self.ref_stains, self.ref_max_c)
            stain_ok = normalized is not None
            if not stain_ok:
                normalized = rgb

            tensor = self.transform(Image.fromarray(normalized)).unsqueeze(0).to(self.device)
            cam, probs = self.cam(tensor)

            cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((x1 - x0, y1 - y0), Image.BILINEAR)
            heat[y0:y1, x0:x1] = np.asarray(cam_img, dtype=np.float32) / 255
            normalized_canvas[y0:y1, x0:x1] = normalized

            prob_dict = {cls: float(p) for cls, p in zip(self.classes, probs)}
            tile_results.append(TileResult(
                row=r, col=c, box=box,
                probabilities=prob_dict,
                prediction=self.classes[int(np.argmax(probs))],
                stain_normalized=stain_ok,
            ))

        if any(not t.stain_normalized for t in tile_results):
            warnings.append("Stain normalization failed on some tiles (too little tissue); originals were used.")

        mean_probs = np.mean([[t.probabilities[c] for c in self.classes] for t in tile_results], axis=0)
        best = int(np.argmax(mean_probs))

        return PredictionResult(
            prediction=self.classes[best],
            confidence=float(mean_probs[best]),
            probabilities={c: float(p) for c, p in zip(self.classes, mean_probs)},
            tiles=tile_results,
            overlay=self._overlay(np.asarray(normalized_canvas), heat, tile_results),
            normalized=Image.fromarray(normalized_canvas),
            warnings=warnings,
        )

    @staticmethod
    def _overlay(rgb, heat, tiles):
        """Heatmap over the analysed tiles; skipped areas are dimmed."""
        cmap = matplotlib.colormaps["jet"]
        out = rgb.astype(np.float32) / 255

        analysed = ~np.isnan(heat)
        colored = cmap(np.nan_to_num(heat))[..., :3]
        out[analysed] = (1 - HEATMAP_ALPHA) * out[analysed] + HEATMAP_ALPHA * colored[analysed]
        out[~analysed] = 0.35 * out[~analysed] + 0.65

        # Thin white outline around each analysed tile
        if len(tiles) > 1:
            for t in tiles:
                x0, y0, x1, y1 = t.box
                out[y0:y0 + 2, x0:x1] = 1
                out[y1 - 2:y1, x0:x1] = 1
                out[y0:y1, x0:x0 + 2] = 1
                out[y0:y1, x1 - 2:x1] = 1

        return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
