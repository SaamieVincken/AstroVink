"""
AstroVink
------------

A Vision Transformer (ViT-S/14, DINOv2 backbone) model fine-tuned
for automated strong gravitational lens detection in Euclid images.

Modules
--------
model.py       – Model architecture and weight loading.
inference.py   – Simple single-image prediction interface.
metrics.py     – Evaluation metrics (AUC, F1, etc.).
transforms.py  – Input preprocessing and normalization.
utils.py       – Device handling and helper functions.
"""

__version__ = "AstroVink-Q1"
__owner__ = "Saamie Vincken"

# Public imports
from .inference import predict
from .model import load_astrovink_q1, DINOv2ForClassification
from .metrics import compute_metrics
from .utils import get_device
