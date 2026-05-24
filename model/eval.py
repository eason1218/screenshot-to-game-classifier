"""
eval.py — Evaluation, confusion matrix, and training curves for the classifier.

Loads the best saved checkpoint, runs on the held-out test set, and writes:
  • confusion_matrix.png
  • classification_report.txt
  • training_curves.png

Run:
    python eval.py
"""
import json
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Scripts are run from the project root; add it to sys.path so the shared
# `config` and the sibling `data` / `model` packages resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data.data import get_dataloaders
from model.model import get_model


@torch.no_grad()
def get_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect ground-truth labels and model predictions for an entire DataLoader.

    Returns:
        (labels, preds) — both as 1-D NumPy integer arrays.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list,
    save_path: str = "results/confusion_matrix.png",
) -> None:
    """Save a seaborn annotated heatmap of the confusion matrix."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix — Test Set")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")


def plot_training_curves(
    history_path: str = config.HISTORY_SAVE_PATH,
    save_path: str = "results/training_curves.png",
) -> None:
    """Load the JSON training history and save loss + accuracy curves."""
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    ax_loss.plot(epochs, history["train_loss"], marker="o", label="Train")
    ax_loss.plot(epochs, history["val_loss"],   marker="o", label="Val")
    ax_loss.set_title("Cross-Entropy Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, [a * 100 for a in history["train_acc"]], marker="o", label="Train")
    ax_acc.plot(epochs, [a * 100 for a in history["val_acc"]],   marker="o", label="Val")
    ax_acc.set_title("Accuracy (%)")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.suptitle("Training Curves", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {save_path}")


def main() -> None:
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    os.makedirs("results", exist_ok=True)

    _, _, test_loader = get_dataloaders()

    model = get_model(num_classes=config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=device)
    )
    model.to(device)
    print(f"Loaded checkpoint from {config.MODEL_SAVE_PATH}")

    labels, preds = get_predictions(model, test_loader, device)

    overall_acc = (labels == preds).mean()
    print(f"\nTest Accuracy: {overall_acc * 100:.2f}%")

    print("\nPer-class accuracy:")
    for i, cls in enumerate(config.CLASS_NAMES):
        mask = labels == i
        if mask.sum() > 0:
            cls_acc = (preds[mask] == labels[mask]).mean()
        else:
            cls_acc = 0.0
        print(f"  {cls:<20}  {cls_acc * 100:.2f}%")

    report = classification_report(labels, preds, target_names=config.CLASS_NAMES)
    print("\nClassification Report:\n", report)

    with open("results/classification_report.txt", "w") as f:
        f.write(report)
    print("Classification report saved to results/classification_report.txt")

    plot_confusion_matrix(labels, preds, config.CLASS_NAMES)
    plot_training_curves()


if __name__ == "__main__":
    main()
