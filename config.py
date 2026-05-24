"""
config.py — Centralized hyperparameters for the game screenshot classifier.
"""
import torch

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5            # 17类合并集约4 epoch即饱和(val 99.65%)；每轮保存最佳checkpoint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 17-class merged set = classic HF 10-class (dataset_hf) + self-collected
# YouTube 10-class (dataset_youtube_hq), de-duplicated on the three overlapping
# games (Fortnite / Minecraft / Genshin Impact). Built by
# data/build_combined_dataset.py.
# Must match torchvision.datasets.ImageFolder alphabetical folder order.
CLASS_NAMES = [
    "ARCRaiders",
    "Among Us",
    "Apex Legends",
    "CounterStrike2",
    "Fortnite",
    "Forza Horizon",
    "Free Fire",
    "Genshin Impact",
    "God of War",
    "LeagueOfLegends",
    "MarvelRivals",
    "Minecraft",
    "Roblox",
    "RocketLeague",
    "Subnautica2",
    "Terraria",
    "Valorant",
]
NUM_CLASSES = len(CLASS_NAMES)

MODEL_SAVE_PATH = "checkpoints/best_model.pth"
HISTORY_SAVE_PATH = "results/training_history.json"
DATASET_DIR = "dataset_combined"
HF_DATASET_NAME = "Bingsu/Gameplay_Images"
SEED = 42
