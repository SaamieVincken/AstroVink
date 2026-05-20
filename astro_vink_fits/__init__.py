"""
AstroVink
------------

A Vision Transformer (ViT-S/14, DINOv2 backbone) model fine-tuned
for automated strong gravitational lens detection in Euclid images.

Modules
--------
model.py       – Model architecture, DINOv2 encoder, and Focal Loss.
train.py       – Training loop for AstroVink-base (simulated data).
retrain.py     – Fine-tuning loop for AstroVink-Q1 (real data).
inference.py   – Single-image or batch inference interface.
metrics.py     – Evaluation metrics (AUC, F1, etc.).
transforms.py  – Input preprocessing and normalization.
utils.py       – Hardware and device utilities.
data.py        – Dataloader creation and dataset management.
"""

__version__ = "AstroVink-Q1"
__author__ = "Saamie Helena Vincken"
__license__ = "MIT"
__citation__ = (
    "Vincken, S. H. (2025). "
    "Euclid Quick Data Release (Q1): AstroVink – A vision transformer approach "
    "to find strong gravitational lens systems. Astronomy & Astrophysics (in prep.)."
)

# --- Public imports ---
# --- Public imports ---
from .model import (
    AstroVink,
    adapt_encoder_channels,
    build_model,
    load_model,
)
from .train import train
from .retrain import retrain
from .inference import run_inference
from .metrics import compute_metrics
from .transforms import build_transforms, arcsinh_preprocess
from .data import FITSDataset
from .metrics import compute_metrics
from .utils import get_device


import sys

_BANNER_TEXT = (
    "\n"
    "AstroVink-Q1 loaded — Vision Transformer for strong lens detection\n"
    "© 2025 Saamie Helena Vincken  |  License: MIT\n"
    "Paper: Vincken et al. (2025, A&A, in prep.)\n"
)

if not hasattr(sys.modules[__name__], "banner_shown"):
    print(_BANNER_TEXT)
    setattr(sys.modules[__name__], "banner_shown", True)
