"""
Configuration settings, system constants, and path directories.
"""

from pathlib import Path
import torch

# Dataset & Task
NUM_CLASSES = 5
IMAGE_SIZE = 384
VAL_SIZE = 0.15

# Training Hyperparameters
BATCH_SIZE = 16
MAX_EPOCHS = 15
PATIENCE = 4
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-5
LEARNING_RATE_BACKBONE = 2e-5
LEARNING_RATE_HEAD = 5e-4
DROPOUT = 0.4
LABEL_SMOOTHING = 0.05
SEEDS = [42, 123]
USE_TTA = True

# Directory Layout
BASE_OUTPUT = Path("/kaggle/working/retinol_engine")
CACHE_DIR = BASE_OUTPUT / "tensor_cache"
CHECKPOINT_DIR = BASE_OUTPUT / "checkpoints"
RESULT_DIR = BASE_OUTPUT / "results"

for d in [CACHE_DIR, CHECKPOINT_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Hardware Runtime
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

CLASS_NAMES = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]
