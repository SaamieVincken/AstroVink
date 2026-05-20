#!/usr/bin/env python3
"""
Example: retrain (fine-tune) an existing AstroVink checkpoint.

To add or remove input bands, change the BANDS list below.  The patch
embedding is automatically re-adapted to the new channel count.
"""

from astro_vink_fits.retrain import retrain

CHECKPOINT = "/path/to/your/AstroVink_best_auc.pth"
DATA_DIR = "/path/to/your/training-data"
OUTPUT_WEIGHTS = "/path/to/your/output/AstroVink-retrained.pth"

# These are Euclid bands, add as many as you need for other projects
# Make sure every FITS image has for each band an extension and define extension names below:
BANDS = ["VIS_BGSUB", "NIR_Y_BGSUB", "NIR_J_BGSUB", "NIR_H_BGSUB"]

if __name__ == "__main__":
    retrain(
        checkpoint_path=CHECKPOINT,
        data_dir=DATA_DIR,
        output_weights=OUTPUT_WEIGHTS,
        bands=BANDS,
        img_size=256,
        batch_size=64,
        num_workers=4,
        epochs=50,
        patience=10,
    )