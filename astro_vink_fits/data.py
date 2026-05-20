"""
AstroVink data: PyTorch Dataset for multi-band FITS cutouts.

Expects::

    root_dir/
        Lens/
            *.fits
        NoLens/
            *.fits
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from astropy.io import fits as fits_io
from transforms import arcsinh_preprocess

DEFAULT_IMG_SIZE = 100


class FITSDataset(Dataset):
    """
    Dataset for Lens/NoLens classified FITS cutouts.

    Each FITS file should contain one HDU per band,
    bands that are not found in a file are filled with zeros.
    """

    CLASS_TO_IDX = {"Lens": 0, "NoLens": 1}

    def __init__(self, root_dir, bands, transform=None):
        self.root_dir = Path(root_dir)
        self.bands = bands
        self.transform = transform
        self.q = 500
        self.clip = 99.85

        self.samples = []
        for class_name, label in self.CLASS_TO_IDX.items():
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                fits_files = list(class_dir.rglob("*.fits"))
                self.samples.extend([(f, label) for f in fits_files])

        self.class_to_idx = dict(self.CLASS_TO_IDX)
        n_lens = sum(1 for _, l in self.samples if l == 0)
        n_nolens = len(self.samples) - n_lens
        print(f"Found {n_lens} Lens and {n_nolens} NoLens samples in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fits_path, label = self.samples[idx]

        with fits_io.open(fits_path) as hdul:
            bands_data = []
            for band_name in self.bands:
                band_found = False
                for hdu in hdul[1:]:
                    if hdu.name == band_name:
                        flux = hdu.data.astype(np.float32)
                        flux_processed = arcsinh_preprocess(flux, q=self.q, clip=self.clip)
                        bands_data.append(flux_processed)
                        band_found = True
                        break
                if not band_found:
                    bands_data.append(np.zeros((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE), dtype=np.float32))

            image = np.stack(bands_data, axis=0)

        image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)

        return image, label
