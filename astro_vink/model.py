import os
import torch
from torch import nn
import torch.nn.functional as F


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


def load_astrovink(weights_path, device):
    """
    Loads the full AstroVink model with the DINOv2 encoder and classifier weights.

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


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced binary classification,
    as used for retraining on imbalanced or harder data.
    Alpha balances class weights, and gamma focuses on harder examples.

    Parameters
    ----------
    alpha : list[float]
        Weighting factors for each class [Lens, NoLens].
    gamma : float
        Focusing parameter that reduces the loss for easy examples.
    reduction : str
        Specifies reduction ('mean' or 'sum').
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is None:
            alpha = [1.0, 1.0]
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.ndim > 1 and inputs.size(1) > 1:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            ce_loss = F.binary_cross_entropy_with_logits(inputs.squeeze(), targets.float(), reduction='none')

        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha.to(inputs.device)[targets]
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
