"""
Tiling and tissue detection, shared by preprocessing and prediction.

NDB-UFES patches are 512x512 crops of 2048x1536 microscope images
(Leica ICC50 HD, 10x/40x objectives). Larger images are therefore
TILED into 512x512 crops rather than resized; resizing would change
the apparent cell size.
"""

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

TILE_SIZE = 512

# Fraction of a tile that must be tissue. NDB-UFES patches were
# curated to exclude blank areas (~99% of them are >= 0.6 tissue).
MIN_TISSUE_FRACTION = 0.6

# Tissue detection uses optical density (OD) relative to each
# image's own background (glass) colour. The Normal slides have
# strong, varying colour casts (cyan/green glass), so fixed
# colour or brightness thresholds fail on them.
BACKGROUND_PERCENTILE = 99
OD_THRESHOLD = 0.2


# ============================================================
# HELPERS
# ============================================================

def tissue_mask(img):
    """Boolean mask of pixels that absorb light relative to the glass."""
    rgb = np.asarray(img).astype(np.float32)

    # Glass is the brightest region in brightfield microscopy
    background = np.percentile(
        rgb.reshape(-1, 3),
        BACKGROUND_PERCENTILE,
        axis=0
    )

    od = -np.log(np.clip((rgb + 1) / (background + 1), 1e-6, 1.0))

    return od.max(axis=2) > OD_THRESHOLD


def iter_tiles(img, tile_size=TILE_SIZE):
    """Yield (row, col, box) for a non-overlapping tile grid over a PIL image."""
    w, h = img.size
    for r in range(h // tile_size):
        for c in range(w // tile_size):
            box = (
                c * tile_size,
                r * tile_size,
                (c + 1) * tile_size,
                (r + 1) * tile_size
            )
            yield r, c, box
