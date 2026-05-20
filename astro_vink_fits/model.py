"""
AstroVink model: DINOv3 ViT encoder with a binary classification head
for Lens/NoLens detection on multi-band FITS data
"""

import torch
from torch import nn
from transformers import AutoModel


DEFAULT_BACKBONE = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def adapt_encoder_channels(encoder, num_channels):
    """
    Replace the patch-embedding Conv2d so the encoder accepts num_channels
    instead of the pretrained 3.

    For num_channels <= 3 the first num_channels slices of the original
    kernel are copied.  For num_channels > 3 the first 3 slices keep the
    original weights and every additional slice is initialised to the
    channel-wise mean of the original kernel.
    """
    original_proj = encoder.embeddings.patch_embeddings
    original_weight = original_proj.weight.data
    out_channels = original_weight.shape[0]
    patch_size = original_weight.shape[2:]

    new_proj = nn.Conv2d(
        in_channels=num_channels,
        out_channels=out_channels,
        kernel_size=patch_size,
        stride=patch_size,
        bias=original_proj.bias is not None,
    )

    if num_channels <= 3:
        new_proj.weight.data[:, :num_channels, :, :] = original_weight[:, :num_channels, :, :]
    else:
        new_proj.weight.data[:, :3, :, :] = original_weight
        avg_weight = original_weight.mean(dim=1, keepdim=True)
        for i in range(3, num_channels):
            new_proj.weight.data[:, i : i + 1, :, :] = avg_weight

    if original_proj.bias is not None:
        new_proj.bias.data = original_proj.bias.data

    encoder.embeddings.patch_embeddings = new_proj
    return encoder


class AstroVink(nn.Module):
    """
    Binary classifier on top of a DINOv3 ViT encoder.
    """

    def __init__(self, encoder, hidden_dim=768):
        super().__init__()
        self.encoder = encoder
        dim = int(self.encoder.config.hidden_size)

        self.head = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, pixel_values, return_cls=False):
        outputs = self.encoder(pixel_values=pixel_values, return_dict=True)
        tokens = outputs.last_hidden_state
        cls = tokens[:, 0]
        patches = tokens[:, 1:]
        pooled = patches.mean(dim=1)
        feat = torch.cat([cls, pooled], dim=1)
        logits = self.head(feat)

        if return_cls:
            return logits, cls
        return logits


def build_model(backbone, num_channels, device):
    """Build a fresh DinoV3Classifier from a HuggingFace backbone name."""
    encoder = AutoModel.from_pretrained(backbone, trust_remote_code=True)
    encoder = adapt_encoder_channels(encoder, num_channels)
    encoder = encoder.to(device)
    model = AstroVink(encoder).to(device)
    return model


def load_model(weights_path, device, num_channels=None):
    """
    Load a trained checkpoint.  Returns (model, checkpoint_dict).

    If num_channels is None the value stored in the checkpoint is used.
    Passing a different value re-adapts the patch embedding for retraining
    with more or fewer bands.
    """
    checkpoint = torch.load(weights_path, map_location=device)
    nc = num_channels if num_channels is not None else checkpoint.get("num_channels", 4)
    backbone = checkpoint.get("backbone", DEFAULT_BACKBONE)

    encoder = AutoModel.from_pretrained(backbone, trust_remote_code=True)
    encoder = adapt_encoder_channels(encoder, nc)
    encoder = encoder.to(device)
    model = AstroVink(encoder).to(device)

    strict = (num_channels is None) or (num_channels == checkpoint.get("num_channels", 4))
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.eval()
    return model, checkpoint