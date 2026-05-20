"""
AstroVink retrain: resume training from a saved checkpoint, optionally with
a different number of input channels.
"""
from astro_vink_fits.utils import get_device
from model import load_model
from train import train as _train


def retrain(
    checkpoint_path,
    data_dir,
    output_weights,
    bands,
    num_channels=None,
    **train_kwargs,
):
    """
    Load a checkpoint to finetune on any number of input channels
    """
    device = get_device()
    nc = num_channels if num_channels is not None else len(bands)

    model, ckpt = load_model(checkpoint_path, device=device, num_channels=nc)
    backbone = ckpt.get("backbone")

    _train(
        data_dir=data_dir,
        output_weights=output_weights,
        bands=bands,
        backbone=backbone,
        model=model,
        **train_kwargs,
    )