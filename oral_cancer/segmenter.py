"""
Epithelium segmenter (U-Net, ResNet-34 encoder), shared by training
(scripts/segmentation/train_segmenter.py) and whole-image segmentation
(scripts/segmentation/segment_whole_images.py).

The model works at half resolution (SCALE): it needs tissue
architecture (epithelium vs stroma), not single nuclei.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

SCALE = 0.5
THRESHOLD = 0.5
ENCODER = "resnet34"

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ============================================================
# MODEL
# ============================================================

def build_segmenter(pretrained=True):
    import segmentation_models_pytorch as smp
    return smp.Unet(ENCODER, encoder_weights="imagenet" if pretrained else None, classes=1)


def load_segmenter(path, device):
    """Return (model, scale) from a checkpoint written by train_segmenter.py."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_segmenter(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint.get("scale", SCALE)


def to_tensor(image):
    x = (image.astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def downscale(img, scale=SCALE):
    """PIL image -> uint8 array at working resolution."""
    size = (round(img.width * scale), round(img.height * scale))
    return np.asarray(img.convert("RGB").resize(size, Image.BILINEAR))


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
