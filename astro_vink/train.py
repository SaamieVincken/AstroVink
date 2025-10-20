import os
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from astro_vink.model import load_pretrained_dinov2, DINOv2ForClassification
from astro_vink.utils import get_device
from astro_vink.data import prepare_dataloaders


def train_base_model(
    data_dir: str,
    output_path: str = "weights/astrovink_base_checkpoint.pth",
    epochs: int = 200,
    batch_size: int = 32,
    patience: int = 20,
):
    """
    Trains the AstroVink-base model on simulated data using CrossEntropyLoss,
    as described in the paper. All encoder blocks are unfrozen and trained jointly.
    The full training checkpoint is saved in the weights/ directory.
    """
    device = get_device()
    print(f"Training on device: {device}")

    # Ensure weights directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Model setup
    encoder = load_pretrained_dinov2(device)
    model = DINOv2ForClassification(encoder, num_classes=2).to(device)

    train_loader, val_loader, _ = prepare_dataloaders(data_dir, batch_size)

    # Training configuration
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 5e-6, "weight_decay": 1e-2},
            {"params": model.classifier.parameters(), "lr": 5e-6, "weight_decay": 1e-2},
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

        # Save full checkpoint when validation loss improves
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
