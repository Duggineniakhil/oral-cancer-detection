"""
Macenko H&E stain normalization (Macenko et al., ISBI 2009).

Shared by the data pipeline (scripts/data/stain_normalize.py) and the
prediction tool, so that images at inference time get exactly the
same normalization as the training data.

NDB-UFES (pink) and the Normal dataset (blue-purple) were stained and
imaged at different institutions. Every image is mapped to one
reference H&E appearance, fitted on NDB-UFES training patches only.
"""

from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

LIGHT_INTENSITY = 240      # Io: brightness of the light source
OD_BETA = 0.15             # ignore near-transparent pixels
ANGLE_PERCENTILE = 1       # alpha: robust extreme stain angles
MAX_C_PERCENTILE = 99      # robust maximum stain concentration
MIN_TISSUE_PIXELS = 500    # below this, the fit is unreliable

# Reference fitted by scripts/data/stain_normalize.py on 200 NDB-UFES
# training patches; a copy is versioned with the code for inference.
DEFAULT_REFERENCE = Path(__file__).resolve().parent / "assets" / "macenko_reference.npz"


# ============================================================
# MACENKO
# ============================================================

def to_od(rgb):
    """RGB uint8 (H, W, 3) -> optical density (N, 3)."""
    rgb = rgb.reshape(-1, 3).astype(np.float64)
    return -np.log((rgb + 1) / LIGHT_INTENSITY)


def fit_stains(rgb):
    """
    Estimate the H&E stain matrix (3x2, columns = H, E) and the
    robust maximum concentration of each stain.

    Returns None if the image has too little tissue to fit.
    """
    od = to_od(rgb)

    tissue = od[np.all(od > OD_BETA, axis=1)]

    if len(tissue) < MIN_TISSUE_PIXELS:
        return None

    # Plane spanned by the two largest eigenvectors of the OD cloud
    _, eigvecs = np.linalg.eigh(np.cov(tissue.T))
    plane = eigvecs[:, 1:3]

    projected = tissue @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])

    min_angle = np.percentile(angles, ANGLE_PERCENTILE)
    max_angle = np.percentile(angles, 100 - ANGLE_PERCENTILE)

    v_min = plane @ np.array([np.cos(min_angle), np.sin(min_angle)])
    v_max = plane @ np.array([np.cos(max_angle), np.sin(max_angle)])

    # Haematoxylin absorbs more red light than eosin -> larger OD_R
    if v_min[0] > v_max[0]:
        stains = np.array([v_min, v_max]).T
    else:
        stains = np.array([v_max, v_min]).T

    # Eigenvector signs are arbitrary; stain OD vectors are positive
    stains *= np.sign(stains.sum(axis=0))
    stains /= np.linalg.norm(stains, axis=0)

    concentrations = np.linalg.lstsq(stains, od.T, rcond=None)[0]
    max_c = np.percentile(concentrations, MAX_C_PERCENTILE, axis=1)

    return stains, max_c, concentrations


def normalize(rgb, ref_stains, ref_max_c):
    """Map an RGB image onto the reference stain appearance (None if no tissue)."""
    fit = fit_stains(rgb)

    if fit is None:
        return None

    _, max_c, concentrations = fit

    concentrations = concentrations * (ref_max_c / max_c)[:, None]

    od = ref_stains @ concentrations
    out = LIGHT_INTENSITY * np.exp(-od)

    return np.clip(out.T, 0, 255).reshape(rgb.shape).astype(np.uint8)


def load_reference(path=DEFAULT_REFERENCE):
    """Return (stains, max_c) saved by scripts/data/stain_normalize.py."""
    ref = np.load(path)
    return ref["stains"], ref["max_c"]
