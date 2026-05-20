"""
AstroVink transforms: arcsinh flux preprocessing and torchvision augmentation pipelines.
"""

import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2, InterpolationMode


def arcsinh_preprocess(img):
    """
    Apply arcsinh stretch to a single 2-D flux array, as expected by AstroVink models
    Flow: arcsinh(flux * Q) -> clip negatives -> percentile clip -> normalise to [0, 1].
    """
    img = np.arcsinh(img * 500)
    img = np.clip(img, 0, None)

    clip_val = np.percentile(img, 99.85)
    img = np.clip(img, 0, clip_val)

    data_min = img.min()
    data_max = img.max()
    denom = data_max + data_min
    if denom > 0:
        img = (img + data_min) / denom
    img = np.clip(img, 0, 1)

    return img


class RandomRotate90(nn.Module):
    """Randomly rotate the input tensor by a multiple of 90 degrees."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() > self.p:
            return img
        k = int(torch.randint(0, 4, (1,)).item())
        if k == 0:
            return img
        return v2.functional.rotate(img, angle=90 * k)


def build_transforms(train, num_channels, img_size=256):
    # Apply ImageNet normalisation (as expected by DINO) to nr of channels
    mean = tuple([0.485] * num_channels)
    std = tuple([0.229] * num_channels)

    if not train:
        return v2.Compose([
            v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.ToDtype(torch.float32, scale=False),
            v2.Normalize(mean=mean, std=std),
        ])

    return v2.Compose([
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.5),
        RandomRotate90(0.7),
        v2.ToDtype(torch.float32, scale=False),
        v2.Normalize(mean=mean, std=std),
    ])