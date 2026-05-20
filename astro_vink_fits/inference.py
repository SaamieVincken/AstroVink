"""
AstroVink inference: score a test set and save predictions to CSV.

Output CSV columns: id, score (probability of Lens), label (0=Lens, 1=NoLens).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from astro_vink_fits.utils import get_device
from model import load_model
from data import FITSDataset
from transforms import build_transforms


def run_inference(
    weights_path,
    test_dir,
    output_csv,
    bands,
    img_size=256,
    batch_size=64,
    num_workers=4,
    num_channels=None,
):
    """
    Score every sample in test_dir and write a ranked CSV.
    """
    device = get_device()
    nc = num_channels if num_channels is not None else len(bands)

    model, _ = load_model(weights_path, device=device, num_channels=nc)
    model.eval()

    transform = build_transforms(train=False, num_channels=nc, img_size=img_size)

    test_ds = FITSDataset(
        root_dir=test_dir, bands=bands,
        transform=transform,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    results = []
    batch_start_idx = 0

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()

            for i, score in enumerate(probs):
                file_path = test_ds.samples[batch_start_idx + i][0]
                file_id = file_path.stem
                label = int(labels[i])
                results.append({"id": file_id, "score": float(score), "label": label})

            batch_start_idx += len(imgs)

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(results)} predictions to: {output_csv}")