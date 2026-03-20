import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset_loader import DatasetLoader
from models.hrnet_w18 import LightweightHRNet
from utils.checkpoint import load_checkpoint, save_checkpoint
import torch_directml

import torch


def compute_pck(model, loader, device, threshold=0.05):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, heatmaps in loader:
            images = images.to(device)
            outputs = torch.sigmoid(model(images))
            B, K, H, W = outputs.shape
            flat_pred = outputs.view(B, K, -1).argmax(dim=-1)
            pred_x = (flat_pred % W).float()
            pred_y = (flat_pred // W).float()
            flat_gt = heatmaps.to(device).view(B, K, -1).argmax(dim=-1)
            gt_x = (flat_gt % W).float()
            gt_y = (flat_gt // W).float()
            dist = torch.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            correct += (dist < threshold * W).sum().item()
            total += B * K
    return correct / total


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, heatmaps in tqdm(loader, desc="Training"):
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, heatmaps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0

    with torch.no_grad():
        for images, heatmaps in tqdm(loader, desc="Validation"):
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            outputs = model(images)

            loss = criterion(outputs, heatmaps)

            running_loss += loss.item()

    return running_loss / len(loader)


def main():
    device = torch_directml.device()

    dataset = DatasetLoader(
        image_folder="dataset/FreiHAND/training/rgb",
        xyz_json="dataset/FreiHAND/training_xyz.json",
        k_json="dataset/FreiHAND/training_K.json",
        heatmap_size=64,
        image_size=224,
        augment=True
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    model = LightweightHRNet(num_keypoints=21).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.ones(21, 1, 1).to(device) * 100.0
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    EPOCHS = 100

    model, optimizer, start_epoch, best_loss = load_checkpoint(
        model,
        optimizer,
        os.path.join(checkpoint_dir, "latest.pth")
    )

    for epoch in range(start_epoch, EPOCHS):

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_loss)

        pck = compute_pck(model, val_loader, device)
        print(
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
            f"PCK@0.05: {pck:.4f}"
        )

        is_best = val_loss < best_loss

        if is_best:
            best_loss = val_loss

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": best_loss
            },
            is_best,
            checkpoint_dir
        )

    print("Training complete")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
