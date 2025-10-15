from torchvision import transforms


def get_transforms(mode: str = "inference"):
    """
    Returns the image transformation pipeline for AstroVink-Q1.

    Parameters
    ----------
    mode : str, optional
        'train', 'val', or 'inference'. Determines whether augmentations are applied.

    Returns
    -------
    torchvision.transforms.Compose
        A transformation pipeline ready for PIL images.
    """

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if mode == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

    else:  # inference (default)
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
