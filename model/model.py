"""
model.py — ResNet50 model definition for game screenshot classification.
"""
import torch.nn as nn
from torchvision import models


def get_model(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Build a ResNet50 fine-tuned for game screenshot classification.

    Loads ImageNet weights (IMAGENET1K_V2) and replaces the final
    fully-connected layer to output `num_classes` logits.
    All backbone layers remain trainable (full fine-tuning).

    Args:
        num_classes: Number of output classes (default 10).
        pretrained:  Load ImageNet pretrained weights when True.

    Returns:
        A torch.nn.Module ready for training or inference.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    # Swap the 1000-class ImageNet head for our num_classes head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
