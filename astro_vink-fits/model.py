import torch
from torch import nn
from transformers import AutoModel



def load_dinov3(DINO_BACKBONE):
    return AutoModel.from_pretrained(DINO_BACKBONE, trust_remote_code=True)


def adapt_dinov3_for_n_channels(model, num_channels):
    original_proj = model.embeddings.patch_embeddings
    original_weight = original_proj.weight.data
    out_channels = original_weight.shape[0]
    patch_size = original_weight.shape[2:]

    new_proj = nn.Conv2d(
        in_channels=num_channels,
        out_channels=out_channels,
        kernel_size=patch_size,
        stride=patch_size,
        bias=original_proj.bias is not None
    )

    if num_channels <= 3:
        new_proj.weight.data[:, :num_channels, :, :] = original_weight[:, :num_channels, :, :]
    else:
        new_proj.weight.data[:, :3, :, :] = original_weight
        avg_weight = original_weight.mean(dim=1, keepdim=True)
        for i in range(3, num_channels):
            new_proj.weight.data[:, i:i + 1, :, :] = avg_weight

    if original_proj.bias is not None:
        new_proj.bias.data = original_proj.bias.data

    model.embeddings.patch_embeddings = new_proj

    return model


class DinoV3Classifier(nn.Module):
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

