#!/usr/bin/env python3
"""
Example: train AstroVink from scratch.

Directory layout expected:

    DATA_DIR/
        Train/
            Lens/*.fits
            NoLens/*.fits
        Val/
            Lens/*.fits
            NoLens/*.fits
"""

from astro_vink_fits.train import train

DATA_DIR = "/path/to/your/training-data"
OUTPUT_WEIGHTS = "/path/to/your/output.pth"

# These are Euclid bands, add as many as you need for other projects
# Make sure every FITS image has for each band an extension and define extension names below:
BANDS = ["VIS_BGSUB", "NIR_Y_BGSUB", "NIR_J_BGSUB", "NIR_H_BGSUB"]

if __name__ == "__main__":
    train(
        data_dir=DATA_DIR,
        output_weights=OUTPUT_WEIGHTS,
        bands=BANDS,
        img_size=256, # image size expected by dinoV3 MUST be dividable by 16
        batch_size=64, # batch size depending on what memory allows
        num_workers=4, # num of workers you can run on CPU cores
        epochs=100, # total epochs
        patience=15, # how many epochs until automatic stopping is triggered
    )
