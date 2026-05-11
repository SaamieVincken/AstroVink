import torch
from torch import nn
from torchvision.transforms import v2, InterpolationMode


def build_aug_transforms(img_size, train: bool, num_channels: int):
    mean = tuple([0.485] * num_channels)
    std = tuple([0.229] * num_channels)

    if not train:
        return v2.Compose([
            v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.ToDtype(torch.float32, scale=False),  # Already in [0,1] from arcsinh
            v2.Normalize(mean=mean, std=std),
        ])

    return v2.Compose([
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.5),
        RandomRotate90(0.7),
        v2.ToDtype(torch.float32, scale=False),  # Already in [0,1] from arcsinh
        v2.Normalize(mean=mean, std=std),
    ])


class RandomRotate90(nn.Module):
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
