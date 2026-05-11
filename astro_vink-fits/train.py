import random
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve
from torch.amp import autocast, GradScaler
from transforms import build_aug_transforms
from dataloader import prepare_dataloaders
from model import load_dinov3, adapt_dinov3_for_n_channels, DinoV3Classifier

base_weights = ""
output_weights = ""
data_dir = ""

BANDS_TO_USE = ['VIS_BGSUB', 'NIR_Y_BGSUB', 'NIR_J_BGSUB', 'NIR_H_BGSUB']
NUM_CHANNELS = len(BANDS_TO_USE)

IMG_SIZE = 256
BATCH_SIZE = 64
NUM_WORKERS = 12
EPOCHS = 100
PATIENCE = 5

ENCODER_LR = 5e-6
HEAD_LR = 5e-4
WEIGHT_DECAY = 0.03

USE_AMP = True
AMP_DTYPE = torch.bfloat16

SEED = 9999

DINO_BACKBONE = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def grad_norm(module):
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def train():
    device = torch.device("cuda")

    train_aug = build_aug_transforms(IMG_SIZE, train=True, num_channels=NUM_CHANNELS)
    val_aug = build_aug_transforms(IMG_SIZE, train=False, num_channels=NUM_CHANNELS)

    train_loader, val_loader, class_to_idx = prepare_dataloaders(data_dir, train_aug, val_aug, BANDS_TO_USE,
                                                                 batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    encoder = load_dinov3(DINO_BACKBONE)
    encoder = adapt_dinov3_for_n_channels(encoder, NUM_CHANNELS)
    encoder = encoder.to(device)
    model = DinoV3Classifier(encoder).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": ENCODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": model.head.parameters(), "lr": HEAD_LR, "weight_decay": 0.0},
        ]
    )
    scaler = GradScaler(enabled=USE_AMP)

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_loss = float("inf")
    patience_ctr = 0
    prev_cls_feats = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        y_true, y_pred = [], []
        enc_gn_epoch = []
        head_gn_epoch = []

        for batch_idx, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)):
            labels = labels.to(device, non_blocking=True)
            pixel_values = imgs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
                logits = model(pixel_values)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            enc_gn_epoch.append(grad_norm(model.encoder))
            head_gn_epoch.append(grad_norm(model.head))

            if batch_idx == 0:
                total_preclip = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_preclip += p.grad.data.norm(2).item() ** 2

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())

        scheduler.step()
        train_loss /= len(train_loader)

        train_acc = accuracy_score(y_true, y_pred)
        train_precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        train_recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        train_f1 = f1_score(y_true, y_pred, average="macro")

        model.eval()
        cls_feats_epoch = []
        val_loss = 0.0
        y_true, y_pred = [], []
        y_prob = []
        val_ce_lens_sum = 0.0
        val_ce_nolens_sum = 0.0
        val_ce_lens_n = 0
        val_ce_nolens_n = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
                    logits, feats = model(imgs, return_cls=True)
                    cls_feats_epoch.append(feats.detach().cpu())

                    ce = nn.functional.cross_entropy(logits, labels, reduction="none")
                    lens_mask = labels == 0
                    nolens_mask = labels == 1
                    if lens_mask.any():
                        val_ce_lens_sum += ce[lens_mask].sum().item()
                        val_ce_lens_n += int(lens_mask.sum().item())
                    if nolens_mask.any():
                        val_ce_nolens_sum += ce[nolens_mask].sum().item()
                        val_ce_nolens_n += int(nolens_mask.sum().item())

                    probs = torch.softmax(logits, dim=1)[:, 0]
                    loss = criterion(logits, labels)

                y_prob.extend(probs.detach().cpu().numpy())
                val_loss += loss.item() * labels.size(0)
                preds = torch.argmax(logits, dim=1)
                y_true.extend(labels.detach().cpu().numpy())
                y_pred.extend(preds.detach().cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average="macro")

        y_true_np = np.array(y_true)
        y_prob_np = np.array(y_prob)

        fpr, tpr, _ = roc_curve((y_true_np == 0).astype(int), y_prob_np)

        cls_feats_epoch = torch.cat(cls_feats_epoch, dim=0)
        if prev_cls_feats is not None:
            cos = torch.nn.CosineSimilarity(dim=1)
            drift = 1 - cos(prev_cls_feats, cls_feats_epoch).mean().item()
            print(f"[DRIFT] cls_cosine_drift={drift:.4e}")

        prev_cls_feats = cls_feats_epoch.clone()

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"train_prec={train_precision:.4f} train_rec={train_recall:.4f} "
            f"train_f1={train_f1:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            ckpt = {
                "model_state_dict": model.state_dict(),
                "backbone": DINO_BACKBONE,
                "class_to_idx": class_to_idx,
                "num_channels": NUM_CHANNELS,
                "bands": BANDS_TO_USE,
            }
            torch.save(ckpt, output_weights)
            print(
                f"Saved new best model to: {output_weights} | epoch={epoch + 1} | val_loss={val_loss:.6f} | val_acc={val_acc:.6f} | val_f1={val_f1:.6f}")
        else:
            patience_ctr += 1

        if patience_ctr >= PATIENCE:
            print("Early stopping.")
            break


set_seed()
train()
