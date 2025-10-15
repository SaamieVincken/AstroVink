import torch


def get_device():
    """
    Returns the best available device for PyTorch execution.

    Priority order:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon)
    3. CPU

    Returns
    -------
    torch.device
        Selected computation device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model):
    """
    Counts the total number of trainable parameters in a model.

    Parameters
    ----------
    model : torch.nn.Module

    Returns
    -------
    int
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
