"""
train.py — Full fine-tuning loop for the game screenshot classifier.

Trains ResNet50 with Adam, tracks the best checkpoint by validation accuracy,
and saves a JSON history file for later plotting.

Run:
    python train.py
"""
import json
import os
import sys
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

# Scripts are run from the project root; add it to sys.path so the shared
# `config` and the sibling `data` / `model` packages resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data.data import get_dataloaders
from model.model import get_model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Run one full pass over the training set.

    Returns:
        (avg_loss, accuracy) for this epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on a validation or test loader.

    Returns:
        (avg_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


def main() -> None:
    torch.manual_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Ensure output folders exist (checkpoints/ for the model, results/ for history)
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(config.HISTORY_SAVE_PATH) or ".", exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders()

    model = get_model(num_classes=config.NUM_CLASSES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    best_val_acc = 0.0
    start = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Train  loss={train_loss:.4f}  acc={train_acc * 100:.2f}%")
        print(f"  Val    loss={val_loss:.4f}  acc={val_acc * 100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  >> New best — saved to {config.MODEL_SAVE_PATH} "
                  f"(val acc {val_acc * 100:.2f}%)")

    elapsed = time.time() - start
    print(
        f"\nTraining complete in {elapsed / 60:.1f} min. "
        f"Best val acc: {best_val_acc * 100:.2f}%"
    )

    with open(config.HISTORY_SAVE_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {config.HISTORY_SAVE_PATH}")


if __name__ == "__main__":
    main()
