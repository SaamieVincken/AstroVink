import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from astropy.io import fits as fits_io


# Arcsinh preprocessing constants
Q = 500
CLIP = 99.85


class FITSDataset(Dataset):
    def __init__(self, root_dir, bands_to_use, transform=None):
        self.root_dir = Path(root_dir)
        self.bands_to_use = bands_to_use
        self.transform = transform

        self.samples = []
        for class_name in ['Lens', 'NoLens']:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                fits_files = list(class_dir.rglob("*.fits"))
                label = 0 if class_name == 'Lens' else 1
                self.samples.extend([(f, label) for f in fits_files])

        self.class_to_idx = {'Lens': 0, 'NoLens': 1}
        print(
            f"Found {len([s for s in self.samples if s[1] == 0])} Lens and {len([s for s in self.samples if s[1] == 1])} NoLens samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fits_path, label = self.samples[idx]

        with fits_io.open(fits_path) as hdul:
            bands_data = []
            for band_name in self.bands_to_use:
                band_found = False
                for hdu in hdul[1:]:
                    if hdu.name == band_name:
                        flux = hdu.data.astype(np.float32)
                        bands_data.append(flux)
                        band_found = True
                        break

                if not band_found:
                    bands_data.append(np.zeros((100, 100), dtype=np.float32))

            # Process all bands together to preserve relative scaling
            bands_processed = arcsinh_preprocess_multiband(bands_data, q=Q, clip=CLIP)
            image = np.stack(bands_processed, axis=0)

        image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)

        return image, label


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def prepare_dataloaders(data_dir, train_aug, val_aug, bands_to_use=4, batch_size=32, num_workers=0):
    train_ds = FITSDataset(
        root_dir=os.path.join(data_dir, "Train"),
        bands_to_use=bands_to_use,
        transform=train_aug,
    )

    val_ds = FITSDataset(
        root_dir=os.path.join(data_dir, "Val"),
        bands_to_use=bands_to_use,
        transform=val_aug,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, train_ds.class_to_idx


def arcsinh_preprocess_multiband(bands_data, q=Q, clip=CLIP):
    """
    Apply arcsinh preprocessing to multiple bands while preserving relative flux scaling.
    bands_data: list of numpy arrays, one per band
    """
    processed_bands = []

    # Apply arcsinh and clipping to each band
    for band in bands_data:
        band = np.arcsinh(band * q)
        band = np.clip(band, 0, None)

        if clip < 100.0:
            clip_val = np.percentile(band, clip)
            band = np.clip(band, 0, clip_val)

        processed_bands.append(band)

    # Stack all bands to find global min/max
    stacked = np.stack(processed_bands, axis=0)
    global_min = stacked.min()
    global_max = stacked.max()

    # Normalize all bands using the same global scale
    denom = global_max + global_min
    if denom > 0:
        normalized_bands = [(band + global_min) / denom for band in processed_bands]
    else:
        normalized_bands = processed_bands

    # Clip to [0,1] and return
    return [np.clip(band, 0, 1) for band in normalized_bands]
