#!/usr/bin/env python3
"""
Example: run inference on a Lens/NoLens test set.

Directory layout expected::

    TEST_DIR/
        Lens/*.fits
        NoLens/*.fits

Outputs a CSV sorted by score (highest first) with columns:
id, score, label.
"""

from astro_vink_fits.inference import run_inference

WEIGHTS = "/path/to/your/AstroVink.pth"
TEST_DIR = "/path/to/your/test-data"
OUTPUT_CSV = "/path/to/your/output/scores.csv"

BANDS = ["VIS_BGSUB", "NIR_Y_BGSUB", "NIR_J_BGSUB", "NIR_H_BGSUB"]

if __name__ == "__main__":
    run_inference(
        weights_path=WEIGHTS,
        test_dir=TEST_DIR,
        output_csv=OUTPUT_CSV,
        bands=BANDS,
        img_size=256,
        batch_size=64,
        num_workers=4,
    )