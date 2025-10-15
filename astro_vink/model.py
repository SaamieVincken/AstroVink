import os

import torch
from torch import nn


class DINOv2ForClassification(nn.Module):
    """
    AstroVink-Q1 model: DINOv2 ViT-S/14 backbone fine-tuned for
    strong gravitational lens classification in Euclid VIS images.
    """

    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.LayerNorm(encoder.embed_dim),
            nn.Linear(encoder.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )

        # Initialize all Linear layers
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """Forward pass through the DINOv2 encoder and classification head."""
        features = self.encoder(x)
        if isinstance(features, tuple):  # handle DINOv2 returning (features, cls_tokens)
            features = features[0]
        logits = self.classifier(features)
        return logits


def load_pretrained_dinov2(device):
    """
    Loads the pretrained DINOv2 ViT-S/14 encoder from Facebook Research hub.
    """
    encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    return encoder


def load_astrovink_q1(weights_path, device):
    """
    Loads the full AstroVink-Q1 model with the DINOv2 encoder and classifier weights.

    Parameters
    ----------
    weights_path : str
        Path to the .pth file containing the fine-tuned weights.
    device : torch.device
        Target device ('cpu', 'cuda', or 'mps').

    Returns
    -------
    model : nn.Module
        Fully loaded AstroVink-Q1 model ready for inference.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    encoder = load_pretrained_dinov2(device)
    model = DINOv2ForClassification(encoder, num_classes=2).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model
