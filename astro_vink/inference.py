import torch
import torch.nn.functional as F
from PIL import Image
from .model import load_astrovink_q1
from .transforms import get_transforms
from .utils import get_device


def predict(image_path: str, weights_path: str):
    """
    Runs inference on a single image using the AstroVink-Q1 model.

    Parameters
    ----------
    image_path : str
        Path to the input image (RGB, 224×224 or larger).
    weights_path : str
        Path to the fine-tuned AstroVink-Q1 weights (.pth).

    Returns
    -------
    dict
        Dictionary containing probabilities for each class:
        {'Lens': float, 'NoLens': float}
    """
    device = get_device()
    model = load_astrovink_q1(weights_path, device)

    transform = get_transforms("inference")
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    return {
        "Lens": float(probs[0]),
        "NoLens": float(probs[1])
    }
