"""
data.py - Dataset loading, splitting, and preprocessing for the game classifier.

The project supports two data sources:
    1. HuggingFace baseline: Bingsu/Gameplay_Images, used for reproducible tests.
    2. Local collected data: dataset/<GameName>/*.png, produced by collect_data.py.

All sources use the same train/eval preprocessing so model training and demo
inference see identical image geometry.
"""
from __future__ import annotations

import os
import platform
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence, Tuple

# Scripts are run from the project root; add it to sys.path so the shared
# top-level `import config` resolves regardless of this file's subfolder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import config


HF_DATASET_NAME = getattr(config, "HF_DATASET_NAME", "Bingsu/Gameplay_Images")
DEFAULT_LOCAL_DATA_DIR = getattr(config, "DATASET_DIR", "dataset")
MIN_LOCAL_CLASSES = 2


class LetterboxResize:
    """
    Resize an image to a square `size` x `size` while preserving aspect ratio,
    padding the remaining area with a solid color (default black bars).

    Unlike Resize((s, s)) this does not distort 16:9 frames, and unlike
    Resize+CenterCrop it keeps the entire frame, so HUD/minimap cues remain.
    """

    def __init__(self, size: int, fill: int = 0):
        self.size = size
        self.fill = (fill, fill, fill)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas


class GameplayDataset(Dataset):
    """PyTorch Dataset wrapper around a HuggingFace Gameplay_Images split."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.dataset[idx]
        image: Image.Image = item["image"]
        label: int = item["label"]

        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """
    Return (train_transform, eval_transform).

    The augmentations intentionally mimic photo-of-screen artifacts: mild
    perspective distortion, rotation, blur, color shift, and occlusion.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        LetterboxResize(config.IMAGE_SIZE),
        transforms.RandomAffine(
            degrees=0,
            scale=(0.85, 1.0),
            translate=(0.05, 0.05),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.RandomGrayscale(p=0.03),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
    eval_tf = transforms.Compose([
        LetterboxResize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, eval_tf


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _effective_num_workers(num_workers: int) -> int:
    # Windows multiprocessing with fork-based workers often causes deadlocks.
    return 0 if platform.system() == "Windows" else num_workers


def _stratified_split_indices(
    labels: Sequence[int],
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split labels into train/val/test indices with class proportions preserved.
    """
    if not 0 < val_split < 0.5:
        raise ValueError("val_split must be between 0 and 0.5")

    indices = list(range(len(labels)))
    counts = Counter(labels)
    too_small = [label for label, count in counts.items() if count < 3]
    if too_small:
        raise ValueError(
            "Each class needs at least 3 images for train/val/test splitting. "
            f"Small labels: {too_small}"
        )

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=val_split * 2,
        stratify=labels,
        random_state=seed,
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=temp_labels,
        random_state=seed,
    )
    return train_idx, val_idx, test_idx


def _print_split_summary(
    labels: Sequence[int],
    split_indices: tuple[list[int], list[int], list[int]],
    class_names: Sequence[str],
) -> None:
    names = ("train", "val", "test")
    print("Split sizes:")
    for name, idxs in zip(names, split_indices):
        split_counts = Counter(labels[i] for i in idxs)
        parts = [
            f"{class_names[label]}={split_counts.get(label, 0)}"
            for label in sorted(set(labels))
        ]
        print(f"  {name:<5} {len(idxs):>5}  " + ", ".join(parts))


def _make_loaders(
    train_set: Dataset,
    val_set: Dataset,
    test_set: Dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    pin_memory = torch.cuda.is_available()
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn if num_workers else None,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn if num_workers else None,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn if num_workers else None,
    )
    return train_loader, val_loader, test_loader


def _balanced_indices(
    labels: Sequence[int],
    max_per_class: int | None,
    seed: int,
) -> list[int]:
    """
    Optionally cap each class to the same number of examples.

    If max_per_class is None, the cap is the smallest class size, which creates
    a fully balanced local dataset. This is useful after YouTube collection,
    where some games often have many more usable frames than others.
    """
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)

    cap = max_per_class or min(len(v) for v in by_label.values())
    selected: list[int] = []
    for label in sorted(by_label):
        candidates = by_label[label]
        rng.shuffle(candidates)
        selected.extend(candidates[:min(cap, len(candidates))])
    selected.sort()
    return selected


def get_hf_dataloaders(
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    print(f"Loading baseline dataset from HuggingFace Hub: {HF_DATASET_NAME}")
    raw = load_dataset(HF_DATASET_NAME, split="train")
    labels = raw["label"]
    hf_class_names = list(raw.features["label"].names)
    split_indices = _stratified_split_indices(labels, val_split, seed)
    _print_split_summary(labels, split_indices, hf_class_names)

    train_tf, eval_tf = get_transforms()
    train_idx, val_idx, test_idx = split_indices
    train_set = GameplayDataset(raw.select(train_idx), transform=train_tf)
    val_set = GameplayDataset(raw.select(val_idx), transform=eval_tf)
    test_set = GameplayDataset(raw.select(test_idx), transform=eval_tf)
    return _make_loaders(train_set, val_set, test_set, batch_size, num_workers, seed)


def get_local_dataloaders(
    data_dir: str = DEFAULT_LOCAL_DATA_DIR,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    val_split: float = 0.1,
    seed: int = config.SEED,
    balance: bool = True,
    max_per_class: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load a locally collected ImageFolder dataset.

    Expected layout:
        dataset/
          Minecraft/*.png
          Fortnite/*.png
          Valorant/*.png

    Labels follow ImageFolder's alphabetical class order. If these classes are
    used for training/evaluation, config.CLASS_NAMES and config.NUM_CLASSES
    must be updated to the same order before training the final model.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Local dataset folder not found: {data_dir}")

    train_tf, eval_tf = get_transforms()
    base_for_labels = ImageFolder(root=str(data_path))
    labels = [label for _, label in base_for_labels.samples]
    if len(base_for_labels.classes) < MIN_LOCAL_CLASSES:
        raise ValueError(
            f"Local dataset needs at least {MIN_LOCAL_CLASSES} classes; "
            f"found {len(base_for_labels.classes)} in {data_dir}"
        )

    selected_indices = (
        _balanced_indices(labels, max_per_class=max_per_class, seed=seed)
        if balance else list(range(len(labels)))
    )
    selected_labels = [labels[i] for i in selected_indices]
    split_relative = _stratified_split_indices(selected_labels, val_split, seed)
    split_indices = tuple(
        [selected_indices[i] for i in relative]
        for relative in split_relative
    )

    print(f"Loading local ImageFolder dataset: {data_path.resolve()}")
    print("Class mapping:")
    for idx, name in enumerate(base_for_labels.classes):
        print(f"  {idx}: {name}")
    if balance:
        cap = max_per_class or min(Counter(labels).values())
        print(f"Balancing enabled: using up to {cap} images per class")
    _print_split_summary(labels, split_indices, base_for_labels.classes)

    train_idx, val_idx, test_idx = split_indices
    train_base = ImageFolder(root=str(data_path), transform=train_tf)
    eval_base = ImageFolder(root=str(data_path), transform=eval_tf)
    train_set = Subset(train_base, train_idx)
    val_set = Subset(eval_base, val_idx)
    test_set = Subset(eval_base, test_idx)
    return _make_loaders(train_set, val_set, test_set, batch_size, num_workers, seed)


def get_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    val_split: float = 0.1,
    seed: int = config.SEED,
    source: str | None = None,
    data_dir: str = DEFAULT_LOCAL_DATA_DIR,
    balance_local: bool = True,
    max_per_class: int | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader).

    Args:
        source: "hf", "local", or "auto". If omitted, reads DATA_SOURCE env
            var and defaults to "hf" for backward compatibility.
        data_dir: Local ImageFolder root used by source="local".
        balance_local: Cap local classes to the same size before splitting.
        max_per_class: Optional local cap per class.
    """
    _seed_everything(seed)
    num_workers = _effective_num_workers(num_workers)

    source = (source or os.environ.get("DATA_SOURCE", "hf")).lower()
    if source not in {"hf", "local", "auto"}:
        raise ValueError("source must be one of: hf, local, auto")

    if source == "auto":
        try:
            return get_local_dataloaders(
                data_dir=data_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                val_split=val_split,
                seed=seed,
                balance=balance_local,
                max_per_class=max_per_class,
            )
        except Exception as exc:
            print(f"Local dataset unavailable ({exc}); falling back to HuggingFace.")
            source = "hf"

    if source == "local":
        return get_local_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            seed=seed,
            balance=balance_local,
            max_per_class=max_per_class,
        )

    return get_hf_dataloaders(batch_size, num_workers, val_split, seed)
