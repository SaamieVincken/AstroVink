import os
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from astro_vink.model import load_pretrained_dinov2, DINOv2ForClassification, FocalLoss
from astro_vink.utils import get_device
from astro_vink.data import prepare_dataloaders


def retrain_astrovink_q1(
    base_weights: str,
    data_dir: str,
    output_path: str = "weights/astrovink_retrained_checkpoint.pth",
    epochs: int = 200,
    batch_size: int = 32,
    patience: int = 20,
):
    """
    Fine-tunes the pretrained AstroVink-base model on real Euclid data
    using FocalLoss as described in the paper.

    Encoder blocks 9–12 are unfrozen and retrained together with the classifier.
    A full training checkpoint (model + optimizer + scheduler) is saved in weights/.
    """
    device = get_device()
    print(f"Retraining on device: {device}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load pretrained DINOv2 encoder and base weights
    encoder = load_pretrained_dinov2(device)
    model = DINOv2ForClassification(encoder, num_classes=2).to(device)

    checkpoint = torch.load(base_weights, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    print(f"Loaded base model from {base_weights}")

    # Freeze encoder then unfreeze blocks 9–12 and classifier
    for p in model.encoder.parameters():
        p.requires_grad = False
    for block in model.encoder.blocks[8:12]:  # 0-indexed: blocks 9–12
        for p in block.parameters():
            p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    train_loader, val_loader, _ = prepare_dataloaders(data_dir, batch_size)

    # Configuration
    criterion = FocalLoss(alpha=[2.0, 1.0], gamma=1.0)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": 1e-5, "weight_decay": 1e-4},
            {"params": model.classifier.parameters(), "lr": 5e-5, "weight_decay": 1e-3},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss, preds_all, labels_all = 0.0, [], []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds_all.extend(logits.argmax(dim=1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(labels_all, preds_all)
        train_f1 = f1_score(labels_all, preds_all, average="macro")

        # Validation
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}, F1: {val_f1:.4f}"
        )

        scheduler.step(epoch)

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, output_path)
            print(f" → Saved full checkpoint at '{output_path}' (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
