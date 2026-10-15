"""
Model, transforms and Grad-CAM shared by training, analysis and the
prediction tool, so all of them build exactly the same network and
input pipeline.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# ============================================================
# CLASSES
# ============================================================

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
# TRANSFORMS
# ============================================================

class RandomRotate90:
    """Histology has no canonical orientation."""

    def __call__(self, img):
        return img.rotate(90 * random.randint(0, 3))


def build_transforms(img_size, grayscale):
    """Returns (train, evaluate) transforms; images are PIL RGB."""
    gray = [transforms.Grayscale(num_output_channels=3)] if grayscale else []

    train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRotate90(),
        # Strong colour augmentation so the model cannot rely on the
        # residual stain differences between the two source datasets
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        # Random blur on all classes, so residual sharpness differences
        # between the two source datasets are not a reliable cue
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 1.5))], p=0.5),
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
