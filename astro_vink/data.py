import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from astro_vink.transforms import get_transforms


def prepare_dataloaders(data_dir: str, batch_size: int = 32):
    """
    Creates training and validation dataloaders from a directory structure:
        data_dir/
            Training/
                Lens/
                NoLens/
            Validation/
                Lens/
                NoLens/

    Returns
    -------
    tuple[DataLoader, DataLoader, list[str]]
        train_loader, val_loader, class_order
    """
    class_order = ["Lens", "NoLens"]

    class CustomImageFolder(datasets.ImageFolder):
        def __getitem__(self, index):
            image, label = super().__getitem__(index)
            return image, label

    train_dataset = CustomImageFolder(
        root=os.path.join(data_dir, "Training"),
        transform=get_transforms("train"),
    )

    val_dataset = CustomImageFolder(
        root=os.path.join(data_dir, "Validation"),
        transform=get_transforms("val"),
    )

    assert train_dataset.classes == class_order, (
        f"Expected class order {class_order}, got {train_dataset.classes}"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, class_order
