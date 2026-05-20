"""
AstroVink train: training loop with cosine-annealed dual LR, grad clipping,
early stopping, and best-checkpoint saving (best_loss, best_auc, best_recall).
"""

import os
import random
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from astro_vink_fits.utils import get_device
from model import build_model, DEFAULT_BACKBONE
from data import FITSDataset
from transforms import build_transforms
from metrics import compute_metrics


def _save_checkpoint(model, backbone, class_to_idx, num_channels, bands, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backbone": backbone,
            "class_to_idx": class_to_idx,
            "num_channels": num_channels,
            "bands": bands,
        },
        path,
    )


def train(
    data_dir,
    output_weights,
    bands,
    backbone=DEFAULT_BACKBONE,
    img_size=256,
    batch_size=64,
    num_workers=4,
    epochs=100,
    patience=15,
    encoder_lr=2e-5,
    head_lr=5e-4,
    weight_decay=0.03,
    grad_clip=3.0,
    seed=9999,
    model=None,
):
    """
    Train AstroVink on a Lens/NoLens FITS dataset
    """
    set_seed(seed)
    num_channels = len(bands)
    device = get_device()

    # Transforms
    train_aug = build_transforms(train=True, num_channels=num_channels, img_size=img_size)
    val_aug = build_transforms(train=False, num_channels=num_channels, img_size=img_size)

    # Datasets
    train_ds = FITSDataset(
        root_dir=os.path.join(data_dir, "Train"),
        bands=bands, transform=train_aug,
    )
    val_ds = FITSDataset(
        root_dir=os.path.join(data_dir, "Val"),
        bands=bands, transform=val_aug,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    class_to_idx = train_ds.class_to_idx

    # Model
    if model is None:
        model = build_model(backbone=backbone, num_channels=num_channels, device=device)
    else:
        model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": encoder_lr, "weight_decay": weight_decay},
        {"params": model.head.parameters(), "lr": head_lr, "weight_decay": 0.0},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Metrics
    best_val_loss = float("inf")
    best_val_auc = -float("inf")
    best_val_recall = -float("inf")
    patience_ctr = 0

    ckpt_kwargs = dict(
        backbone=backbone, class_to_idx=class_to_idx,
        num_channels=num_channels, bands=bands,
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        y_true, y_pred = [], []

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())

        scheduler.step()
        train_loss /= len(train_loader)
        train_m = compute_metrics(y_true, y_pred)

        print(
            f"Epoch {epoch + 1}/{epochs} [train] "
            f"loss={train_loss:.4f} acc={train_m['accuracy']:.4f} "
            f"prec={train_m['precision']:.4f} rec={train_m['recall']:.4f} "
            f"f1={train_m['f1']:.4f}"
        )

        model.eval()
        val_loss = 0.0
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(imgs)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)[:, 0]

                val_loss += loss.item() * labels.size(0)
                preds = torch.argmax(logits, dim=1)
                y_true.extend(labels.detach().cpu().numpy())
                y_pred.extend(preds.detach().cpu().numpy())
                y_prob.extend(probs.detach().cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_m = compute_metrics(y_true, y_pred, y_prob=np.array(y_prob))

        print(
            f"Epoch {epoch + 1}/{epochs} [val]   "
            f"loss={val_loss:.4f} acc={val_m['accuracy']:.4f} "
            f"prec={val_m['precision']:.4f} rec={val_m['recall']:.4f} "
            f"f1={val_m['f1']:.4f} auc={val_m['auc']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            _save_checkpoint(model=model, path=output_weights, **ckpt_kwargs)
            print(f"Saved best model (val_loss={val_loss:.6f})")
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            print("Early stopping.")
            break


def set_seed(seed=9999):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)