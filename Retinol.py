import os
import gc
import json
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    f1_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 5
IMAGE_SIZE = 384
BATCH_SIZE = 16
MAX_EPOCHS = 20
PATIENCE = 5
NUM_WORKERS = 4
VAL_SIZE = 0.15
TEST_SIZE = 0.15

WEIGHT_DECAY = 1e-5
LEARNING_RATE_BACKBONE = 2e-5
LEARNING_RATE_HEAD = 5e-4

DROPOUT = 0.4
LABEL_SMOOTHING = 0.05

USE_TTA = True
SEEDS = [42, 123, 777, 456, 999]

CACHE_VERSION = "effnet_v2_s_rgb_roi_edge_v5_fixed"

GRADIENT_ACCUMULATION_STEPS = 1


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

BASE_OUTPUT = Path("/kaggle/working/idrid_effnet_v2_s_fixed")
CACHE_DIR = BASE_OUTPUT / "cache"
CHECKPOINT_DIR = BASE_OUTPUT / "checkpoints"
RESULT_DIR = BASE_OUTPUT / "results"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("DEVICE CONFIGURATION")
print("=" * 80)
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# ============================================================
# DATASET DISCOVERY
# ============================================================

def find_idrid_root():
    candidates = [
        Path("/kaggle/input/datasets/aaryapatel98/indian-diabetic-retinopathy-image-dataset/B.%20Disease%20Grading/B. Disease Grading"),
        Path("/kaggle/input/indian-diabetic-retinopathy-image-dataset/B. Disease Grading"),
        Path("/kaggle/input/B. Disease Grading")
    ]

    for path in candidates:
        if path.exists():
            print(f"Dataset found: {path}")
            return path

    for path in Path("/kaggle/input").rglob("IDRiD_Disease Grading_Training Labels.csv"):
        print(f"Dataset discovered: {path.parent}")
        return path.parent

    raise FileNotFoundError("IDRiD Disease Grading dataset not found.")


DATA_ROOT = find_idrid_root()

TRAIN_IMAGE_DIR = DATA_ROOT / "1. Original Images" / "a. Training Set"
TEST_IMAGE_DIR = DATA_ROOT / "1. Original Images" / "b. Testing Set"
GROUNDTRUTH_DIR = DATA_ROOT / "2. Groundtruths"
TRAIN_CSV = GROUNDTRUTH_DIR / "a. IDRiD_Disease Grading_Training Labels.csv"
TEST_CSV = GROUNDTRUTH_DIR / "b. IDRiD_Disease Grading_Testing Labels.csv"

print()
print("=" * 80)
print("DATASET PATHS")
print("=" * 80)
print(f"Training: {TRAIN_IMAGE_DIR}")
print(f"Testing: {TEST_IMAGE_DIR}")


# ============================================================
# LOAD CSV LABELS
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df.columns = [str(c).strip() for c in train_df.columns]
test_df.columns = [str(c).strip() for c in test_df.columns]

IMAGE_COL = "Image name"
TARGET_COL = "Retinopathy grade"

train_df = train_df[[IMAGE_COL, TARGET_COL]].copy()
test_df = test_df[[IMAGE_COL, TARGET_COL]].copy()

train_df[TARGET_COL] = pd.to_numeric(train_df[TARGET_COL], errors="coerce")
test_df[TARGET_COL] = pd.to_numeric(test_df[TARGET_COL], errors="coerce")

train_df = train_df.dropna()
test_df = test_df.dropna()

train_df[TARGET_COL] = train_df[TARGET_COL].astype(int)
test_df[TARGET_COL] = test_df[TARGET_COL].astype(int)


# ============================================================
# BUILD IMAGE INDEX
# ============================================================

def build_image_index(folder):
    index = {}
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            index[path.stem.lower()] = str(path)
    return index


train_index = build_image_index(TRAIN_IMAGE_DIR)
test_index = build_image_index(TEST_IMAGE_DIR)


def resolve_image(name, index):
    stem = Path(str(name)).stem.lower()
    return index.get(stem, None)


train_df["path"] = train_df[IMAGE_COL].apply(lambda x: resolve_image(x, train_index))
test_df["path"] = test_df[IMAGE_COL].apply(lambda x: resolve_image(x, test_index))

train_df = train_df[train_df["path"].notna()].reset_index(drop=True)
test_df = test_df[test_df["path"].notna()].reset_index(drop=True)

print()
print("=" * 80)
print("IMAGE MATCHING")
print("=" * 80)
print(f"Training images: {len(train_df)}")
print(f"Testing images: {len(test_df)}")

# ============================================================
# PROPER DATA SPLIT (NO LEAKAGE)
# ============================================================

# Split training into train/val BEFORE preprocessing
train_part, val_part = train_test_split(
    train_df,
    test_size=VAL_SIZE,
    stratify=train_df[TARGET_COL],
    random_state=42
)

train_part = train_part.reset_index(drop=True)
val_part = val_part.reset_index(drop=True)

print()
print("=" * 80)
print("DATA SPLIT (PROPER)")
print("=" * 80)
print(f"Training: {len(train_part)}")
print(f"Validation: {len(val_part)}")
print(f"Official test: {len(test_df)}")

print()
print("=" * 80)
print("CLASS DISTRIBUTION")
print("=" * 80)
print("Training:")
print(train_part[TARGET_COL].value_counts().sort_index())
print("\nValidation:")
print(val_part[TARGET_COL].value_counts().sort_index())
print("\nTest:")
print(test_df[TARGET_COL].value_counts().sort_index())


# ============================================================
# IMPROVED PREPROCESSING
# ============================================================

def detect_retinal_roi(img):
    """Detect retinal region with better thresholding."""
    if img is None:
        raise ValueError("Image is None.")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use Otsu's thresholding for better automatic detection
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert for morphological operations
    thresh = cv2.bitwise_not(thresh)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)

    if area < (0.10 * h * w):
        return img

    x, y, cw, ch = cv2.boundingRect(contour)
    margin = int(0.08 * max(cw, ch))

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + cw + margin)
    y2 = min(h, y + ch + margin)

    crop = img[y1:y2, x1:x2]

    return crop if crop.size > 0 else img


def square_crop(img):
    """Extract square region from center."""
    h, w = img.shape[:2]
    size = min(h, w)
    
    cx, cy = w // 2, h // 2
    half = size // 2
    
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)

    crop = img[y1:y2, x1:x2]
    
    side = min(crop.shape[0], crop.shape[1])
    return crop[:side, :side]


def apply_clahe(img):
    """CLAHE enhancement."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def normalize_illumination(img):
    """Background illumination normalization."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    background = cv2.GaussianBlur(l, (0, 0), sigmaX=40)
    
    corrected = (l.astype(np.float32) - background.astype(np.float32) + 128.0)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    output = cv2.merge([corrected, a, b])
    return cv2.cvtColor(output, cv2.COLOR_LAB2BGR)


def extract_edges(img):
    """Extract edges with improved parameters."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.0)

    # Improved Canny parameters
    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    return edges


def preprocess_image(path):
    """Complete preprocessing pipeline."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Could not read image: {path}")

    original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    roi_bgr = detect_retinal_roi(img)
    roi_bgr = square_crop(roi_bgr)
    roi_bgr = cv2.resize(roi_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    
    roi_bgr = apply_clahe(roi_bgr)
    roi_bgr = normalize_illumination(roi_bgr)

    processed_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    edges = extract_edges(roi_bgr)

    rgb_float = processed_rgb.astype(np.float32) / 255.0
    edge_float = edges.astype(np.float32) / 255.0

    combined = np.concatenate([rgb_float, edge_float[..., None]], axis=2)
    combined = np.transpose(combined, (2, 0, 1))

    return (
        combined.astype(np.float32),
        original_rgb,
        processed_rgb,
        edges
    )


# ============================================================
# CACHE DATASET (WITH PROPER SPLIT)
# ============================================================

def cache_dataframe(df, split_name):
    """Cache preprocessed images."""
    cache_file = CACHE_DIR / f"{CACHE_VERSION}_{split_name}.pt"

    if cache_file.exists():
        print(f"Loading cache: {cache_file}")
        return torch.load(cache_file, map_location="cpu", weights_only=False)

    print(f"\nCreating cache: {split_name}")
    samples = []

    for i, row in df.iterrows():
        try:
            combined, _, _, _ = preprocess_image(row["path"])
            samples.append({
                "image": torch.from_numpy(combined),
                "label": int(row[TARGET_COL]),
                "name": str(row[IMAGE_COL]),
                "path": str(row["path"])
            })
        except Exception as e:
            print(f"Error: {row['path']} - {e}")

        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(df)}")

    torch.save(samples, cache_file)
    print(f"Saved: {cache_file}")
    return samples


train_cache = cache_dataframe(train_part, "train")
val_cache = cache_dataframe(val_part, "val")
test_cache = cache_dataframe(test_df, "test")


# ============================================================
# LIGHTWEIGHT OOD / IMAGE-SHIFT REFERENCE
# ============================================================
# These statistics are derived from the already available IDRiD
# training cache. No additional training data or model training is
# required. They are used only as a conservative review flag.
def _image_shift_features(sample_tensor):
    """Extract simple, model-independent distribution features."""
    # Accept either a PyTorch tensor or a NumPy array.
    # The frontend safety layer passes NumPy arrays, so calling
    # `.detach()` unconditionally causes:
    # AttributeError: 'numpy.ndarray' object has no attribute 'detach'
    if isinstance(sample_tensor, torch.Tensor):
        x = sample_tensor.detach().cpu().numpy().astype(np.float32)
    else:
        x = np.asarray(sample_tensor, dtype=np.float32)

    # Expected layout is C x H x W. Accept H x W x C defensively too.
    if x.ndim != 3:
        raise ValueError(f"Expected 3D feature tensor, got {x.shape}")

    if x.shape[0] == 4:
        rgb = x[:3]
        edge = x[3]
    elif x.shape[-1] == 4:
        x = np.transpose(x, (2, 0, 1))
        rgb = x[:3]
        edge = x[3]
    else:
        raise ValueError(f"Expected 4-channel RGB+edge tensor, got {x.shape}")

    gray = np.mean(rgb, axis=0)
    bright = (gray > 0.78).astype(np.uint8)
    # Remove tiny noise and merge nearby bright regions.
    kernel = np.ones((5, 5), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    blob_count = 0
    blob_area = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if 20 <= area <= 1800 and w > 0 and h > 0 and max(w, h) / max(1, min(w, h)) < 5.0:
            blob_count += 1
            blob_area += area

    return np.array([
        float(rgb.mean()),
        float(rgb.std()),
        float(rgb[0].mean()),
        float(rgb[1].mean()),
        float(rgb[2].mean()),
        float(edge.mean()),
        float((edge > 0.25).mean()),
        float(blob_count),
        float(blob_area) / float(rgb.shape[1] * rgb.shape[2]),
    ], dtype=np.float32)


try:
    _ood_feature_matrix = np.stack(
        [_image_shift_features(item["image"]) for item in train_cache], axis=0
    )
    OOD_REFERENCE_MEAN = _ood_feature_matrix.mean(axis=0)
    OOD_REFERENCE_STD = np.maximum(_ood_feature_matrix.std(axis=0), 1e-5)
except Exception as _ood_ref_error:
    print(f"OOD reference feature warning: {_ood_ref_error}")
    OOD_REFERENCE_MEAN = np.zeros(9, dtype=np.float32)
    OOD_REFERENCE_STD = np.ones(9, dtype=np.float32)



# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment_tensor(x):
    """Stronger augmentation strategy."""
    # Horizontal flip (high probability)
    if random.random() < 0.6:
        x = torch.flip(x, dims=[2])

    # Vertical flip
    if random.random() < 0.3:
        x = torch.flip(x, dims=[1])

    # Rotation
    if random.random() < 0.4:
        k = random.choice([1, 2, 3])
        x = torch.rot90(x, k=k, dims=[1, 2])

    # RGB brightness/contrast
    if random.random() < 0.5:
        scale = random.uniform(0.85, 1.15)
        bias = random.uniform(-0.05, 0.05)
        
        rgb = x[:3]
        rgb = (rgb * scale + bias).clamp(0, 1)
        x = torch.cat([rgb, x[3:4]], dim=0)

    # Edge enhancement (random)
    if random.random() < 0.3:
        edge = x[3:4]
        edge = (edge * random.uniform(0.8, 1.2)).clamp(0, 1)
        x = torch.cat([x[:3], edge], dim=0)

    return x


# ============================================================
# NORMALIZATION
# ============================================================

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def normalize_input(x):
    """Normalize RGB with ImageNet stats, edge separately."""
    rgb = x[:3]
    edge = x[3:4]

    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    edge = (edge - 0.5) / 0.5

    return torch.cat([rgb, edge], dim=0).float()


# ============================================================
# DATASET
# ============================================================

class CachedIDRiDDataset(Dataset):
    def __init__(self, samples, training=False):
        self.samples = samples
        self.training = training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        x = item["image"].clone()
        y = item["label"]

        if self.training:
            x = augment_tensor(x)

        x = normalize_input(x)

        return x, torch.tensor(y, dtype=torch.long)


# ============================================================
# IMPROVED CLASS BALANCING
# ============================================================

train_labels = np.array([item["label"] for item in train_cache])
class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)

print()
print("=" * 80)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 80)
print(class_counts)

# Better weighting strategy
class_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1))
class_weights = class_weights / class_weights.sum() * len(class_weights)

print(f"Class weights: {class_weights}")

sample_weights = np.array([class_weights[item["label"]] for item in train_cache])
sampler = WeightedRandomSampler(
    weights=torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(train_cache),
    replacement=True
)


# ============================================================
# DATALOADERS
# ============================================================

def create_loaders():
    train_dataset = CachedIDRiDDataset(train_cache, training=True)
    val_dataset = CachedIDRiDDataset(val_cache, training=False)
    test_dataset = CachedIDRiDDataset(test_cache, training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    return train_loader, val_loader, test_loader


# ============================================================
# IMPROVED MODEL ARCHITECTURE
# ============================================================

def build_model():
    """Build EfficientNet-V2-S with proper 4-channel initialization."""
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights)

    # CRITICAL FIX: Proper 4-channel convolution initialization
    old_conv = model.features[0][0]

    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=(old_conv.bias is not None)
    )

    with torch.no_grad():
        # Copy RGB weights
        new_conv.weight[:, :3] = old_conv.weight

        # Initialize edge channel as learned combination
        new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True) * 0.5

        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    model.features[0][0] = new_conv

    # Improved classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=DROPOUT * 0.8),
        nn.Linear(256, NUM_CLASSES)
    )

    return model


# ============================================================
# FOCAL LOSS + LABEL SMOOTHING
# ============================================================

class FocalLabelSmoothingCE(nn.Module):
    """Focal loss with label smoothing for imbalanced classes."""
    
    def __init__(self, weights, smoothing=0.05, alpha=0.25, gamma=2.0):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.smoothing = smoothing
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        n_classes = logits.size(1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        ce_loss = -(true_dist * log_probs).sum(dim=1)
        
        # Focal term
        probs = torch.softmax(logits, dim=1)
        p_t = (true_dist * probs).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma

        loss = self.alpha * focal_weight * ce_loss
        
        # Class weighting
        weights = self.weights[targets]
        return (loss * weights).mean()


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    balanced = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()

    losses = []
    all_targets = []
    all_preds = []
    all_probs = []

    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        logits = model(images)

        if criterion is not None:
            loss = criterion(logits, targets)
            losses.append(loss.item())

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    probs = np.concatenate(all_probs, axis=0)

    metrics = calculate_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0

    return metrics, y_true, y_pred, probs


# ============================================================
# TRAIN ONE SEED - IMPROVED
# ============================================================

def train_one_seed(seed):
    print()
    print("=" * 80)
    print(f"TRAINING SEED {seed}")
    print("=" * 80)

    seed_everything(seed)

    train_loader, val_loader, test_loader = create_loaders()
    model = build_model()
    model = model.to(DEVICE)

    # Differential learning rates
    early_params = []
    late_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("classifier"):
            head_params.append(param)
        elif name.startswith("features.0") or name.startswith("features.1"):
            early_params.append(param)
        else:
            late_params.append(param)

    # IMPROVED OPTIMIZER
    optimizer = optim.AdamW(
        [
            {"params": early_params, "lr": 1e-6},
            {"params": late_params, "lr": LEARNING_RATE_BACKBONE},
            {"params": head_params, "lr": LEARNING_RATE_HEAD}
        ],
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # COSINE ANNEALING SCHEDULER
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=1e-7
    )

    criterion = FocalLabelSmoothingCE(
        class_weights,
        smoothing=LABEL_SMOOTHING,
        alpha=0.25,
        gamma=2.0
    ).to(DEVICE)

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_macro_f1 = -1.0
    best_balanced = -1.0
    best_epoch = 0
    patience_counter = 0

    best_path = CHECKPOINT_DIR / f"effnet_v2_s_seed_{seed}_best.pt"
    history = []

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        train_targets = []
        train_preds = []

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            preds = logits.argmax(dim=1)

            train_targets.extend(targets.detach().cpu().numpy())
            train_preds.extend(preds.detach().cpu().numpy())

        scheduler.step()

        train_metrics = calculate_metrics(train_targets, train_preds)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion)

        current_lr = optimizer.param_groups[-1]["lr"]

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Loss {np.mean(train_losses):.4f} | "
            f"Train Acc {train_metrics['accuracy']:.4f} | "
            f"Val Acc {val_metrics['accuracy']:.4f} | "
            f"Val BalAcc {val_metrics['balanced_accuracy']:.4f} | "
            f"Val F1 {val_metrics['macro_f1']:.4f} | "
            f"LR {current_lr:.2e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "lr": current_lr
        })

        # Model selection (Macro-F1 primary)
        improved = (
            val_metrics["macro_f1"] > best_macro_f1 + 1e-4
            or (
                abs(val_metrics["macro_f1"] - best_macro_f1) < 1e-4
                and val_metrics["balanced_accuracy"] > best_balanced + 1e-4
            )
        )

        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
            best_balanced = val_metrics["balanced_accuracy"]
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "seed": seed,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "class_mapping": {i: f"Grade {i}" for i in range(NUM_CLASSES)}
            }, best_path)

            print("  -> BEST MODEL SAVED")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

    # Load best model
    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics, val_true, val_pred, val_probs = evaluate(model, val_loader, criterion)

    print()
    print(f"BEST MODEL - SEED {seed}")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
    print(f"Validation Macro-F1: {val_metrics['macro_f1']:.4f}")

    history_df = pd.DataFrame(history)
    history_df.to_csv(RESULT_DIR / f"history_seed_{seed}.csv", index=False)

    return {
        "seed": seed,
        "model": model,
        "val_metrics": val_metrics,
        "val_true": val_true,
        "val_pred": val_pred,
        "val_probs": val_probs,
        "best_epoch": best_epoch,
        "checkpoint": str(best_path)
    }


# ============================================================
# MULTI-SEED TRAINING
# ============================================================

results = []

for seed in SEEDS:
    result = train_one_seed(seed)
    results.append(result)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# MULTI-SEED SUMMARY
# ============================================================

seed_summary = pd.DataFrame([
    {
        "seed": result["seed"],
        "val_accuracy": result["val_metrics"]["accuracy"],
        "val_balanced_accuracy": result["val_metrics"]["balanced_accuracy"],
        "val_macro_f1": result["val_metrics"]["macro_f1"],
        "val_weighted_f1": result["val_metrics"]["weighted_f1"],
        "best_epoch": result["best_epoch"]
    }
    for result in results
])

print()
print("=" * 80)
print("MULTI-SEED VALIDATION RESULTS")
print("=" * 80)
print(seed_summary.to_string(index=False))
seed_summary.to_csv(RESULT_DIR / "multi_seed_validation.csv", index=False)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_result = sorted(
    results,
    key=lambda r: (
        r["val_metrics"]["macro_f1"],
        r["val_metrics"]["balanced_accuracy"],
        r["val_metrics"]["accuracy"]
    ),
    reverse=True
)[0]

BEST_SEED = best_result["seed"]
best_model = best_result["model"]

print()
print("=" * 80)
print("SELECTED BEST MODEL")
print("=" * 80)
print(f"Best seed: {BEST_SEED}")
print(f"Validation Accuracy: {best_result['val_metrics']['accuracy']:.4f}")
print(f"Validation Balanced Accuracy: {best_result['val_metrics']['balanced_accuracy']:.4f}")
print(f"Validation Macro-F1: {best_result['val_metrics']['macro_f1']:.4f}")


# ============================================================
# IMPROVED TEST-TIME AUGMENTATION
# ============================================================

@torch.no_grad()
def predict_tta(model, loader):
    """Better TTA with proper averaging."""
    model.eval()

    all_probs = []
    all_targets = []

    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)

        # Original
        probs1 = torch.softmax(model(images), dim=1)

        # Horizontal flip
        probs2 = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)

        # Vertical flip
        probs3 = torch.softmax(model(torch.flip(images, dims=[2])), dim=1)

        # 90-degree rotation
        probs4 = torch.softmax(model(torch.rot90(images, k=1, dims=[2, 3])), dim=1)

        # Weighted ensemble
        probs = (0.40 * probs1 + 0.25 * probs2 + 0.20 * probs3 + 0.15 * probs4)

        all_probs.append(probs.cpu().numpy())
        all_targets.extend(targets.numpy())

    probs = np.concatenate(all_probs, axis=0)
    targets = np.array(all_targets)
    preds = probs.argmax(axis=1)

    return targets, preds, probs


# ============================================================
# OFFICIAL TEST EVALUATION
# ============================================================

_, _, test_loader = create_loaders()

if USE_TTA:
    test_true, test_pred, test_probs = predict_tta(best_model, test_loader)
else:
    test_metrics, test_true, test_pred, test_probs = evaluate(best_model, test_loader)

test_metrics = calculate_metrics(test_true, test_pred)

precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
    test_true, test_pred, average="macro", zero_division=0
)

print()
print("=" * 80)
print("OFFICIAL IDRiD TEST RESULTS")
print("=" * 80)
print(f"Accuracy          : {test_metrics['accuracy']:.4f}")
print(f"Balanced Accuracy : {test_metrics['balanced_accuracy']:.4f}")
print(f"Macro Precision   : {precision_macro:.4f}")
print(f"Macro Recall      : {recall_macro:.4f}")
print(f"Macro F1          : {test_metrics['macro_f1']:.4f}")
print(f"Weighted F1       : {test_metrics['weighted_f1']:.4f}")


# ============================================================
# CLASSIFICATION REPORT & CONFUSION MATRIX
# ============================================================

CLASS_NAMES = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]

report = classification_report(
    test_true, test_pred,
    labels=[0, 1, 2, 3, 4],
    target_names=CLASS_NAMES,
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(report).T

print()
print("=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)
print(report_df.to_string())
report_df.to_csv(RESULT_DIR / "official_test_classification_report.csv")

cm = confusion_matrix(test_true, test_pred, labels=[0, 1, 2, 3, 4])

plt.figure(figsize=(8, 7))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("EfficientNet-V2-S IDRiD Test Confusion Matrix")
plt.colorbar()
plt.xticks(range(5), CLASS_NAMES, rotation=45)
plt.yticks(range(5), CLASS_NAMES)

for i in range(5):
    for j in range(5):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")

plt.xlabel("Predicted Grade")
plt.ylabel("True Grade")
plt.tight_layout()
plt.savefig(RESULT_DIR / "confusion_matrix.png", dpi=200)
plt.show()


# ============================================================
# SAVE FINAL MODEL
# ============================================================

FINAL_MODEL_PATH = BASE_OUTPUT / "EfficientNetV2S_IDRiD_BEST_FIXED.pt"

torch.save({
    "model_state_dict": best_model.state_dict(),
    "architecture": "EfficientNet-V2-S",
    "input_channels": 4,
    "image_size": IMAGE_SIZE,
    "class_mapping": {i: f"Grade {i}" for i in range(NUM_CLASSES)},
    "best_seed": BEST_SEED,
    "validation_metrics": best_result["val_metrics"],
    "test_metrics": test_metrics,
    "preprocessing": [
        "Retinal ROI detection (Otsu)",
        "Square crop",
        "CLAHE (clipLimit=3.0)",
        "Illumination normalization",
        "Canny edge detection (improved)"
    ]
}, FINAL_MODEL_PATH)

print()
print(f"Final model saved: {FINAL_MODEL_PATH}")


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "architecture": "EfficientNet-V2-S",
    "image_size": IMAGE_SIZE,
    "input": "RGB + Canny edge",
    "classes": {str(i): f"Grade {i}" for i in range(NUM_CLASSES)},
    "best_seed": int(BEST_SEED),
    "validation": {k: float(v) for k, v in best_result["val_metrics"].items()},
    "official_test": {k: float(v) for k, v in test_metrics.items()},
    "tta_enabled": bool(USE_TTA),
    "model_path": str(FINAL_MODEL_PATH),
    "key_improvements": [
        "Proper train/val/test split (no leakage)",
        "Focal loss + label smoothing",
        "Improved 4-channel initialization",
        "Cosine annealing scheduler",
        "Better class weighting",
        "Stronger data augmentation",
        "Improved edge detection",
        "Differential learning rates"
    ]
}

with open(RESULT_DIR / "final_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print()
print("=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)
print(f"Results directory: {RESULT_DIR}")
print(f"Summary: {RESULT_DIR / 'final_summary.json'}")


# ============================================================
# 46. FRONTEND PREPROCESSOR
# ============================================================

def preprocess_uploaded_image(image):
    """
    Robustly accepts:
        PIL.Image
        NumPy HxWx3
        NumPy HxWx4
        grayscale NumPy

    Returns:
        model tensor: 1 x 4 x 384 x 384
        original RGB: H x W x 3 uint8
        processed RGB: 384 x 384 x 3 uint8
        edges: 384 x 384 uint8
    """

    # Convert input to NumPy RGB
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        image = np.asarray(image)
    else:
        image = np.asarray(image)

    # Handle grayscale
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Handle RGBA
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    # Validate shape
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Invalid image shape: {image.shape}")

    # Convert to uint8
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            if image.max() <= 1.0:
                image = (image * 255.0)

        image = np.clip(image, 0, 255).astype(np.uint8)

    # Make contiguous
    image = np.ascontiguousarray(image)
    original_rgb = image.copy()

    # RGB -> BGR
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # ROI detection
    roi = detect_retinal_roi(bgr)

    # Square crop
    roi = square_crop(roi)

    # Resize
    roi = cv2.resize(roi, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    # CLAHE
    roi = apply_clahe(roi)

    # Illumination normalization
    roi = normalize_illumination(roi)

    # RGB processed image
    processed_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # CRITICAL: Output exactly (384, 384, 3) uint8
    processed_rgb = np.ascontiguousarray(processed_rgb).astype(np.uint8)

    # Edge detection
    edges = extract_edges(roi)
    edges = np.ascontiguousarray(edges).astype(np.uint8)

    # RGB float
    rgb_float = processed_rgb.astype(np.float32) / 255.0

    # Edge float
    edge_float = edges.astype(np.float32) / 255.0

    # RGB + EDGE (384 x 384 x 4)
    combined = np.concatenate([rgb_float, edge_float[..., None]], axis=2)

    # HWC -> CHW (4 x 384 x 384)
    combined = np.transpose(combined, (2, 0, 1))
    combined = np.ascontiguousarray(combined)

    # Convert to tensor
    tensor = torch.from_numpy(combined).float()
    tensor = normalize_input(tensor)

    # Add batch dimension (1 x 4 x 384 x 384)
    tensor = tensor.unsqueeze(0)

    return (tensor, original_rgb, processed_rgb, edges)


# ============================================================
# 47. FRONTEND PREDICTION
# ============================================================

def _clinical_review_flags(processed_rgb, edges, probs_stack, probs):
    """
    Conservative, training-free safety checks for uploaded images.

    These checks NEVER replace the trained model prediction. They only
    identify cases that deserve additional review, especially when an
    uploaded image differs from the IDRiD training distribution or when
    TTA predictions disagree.
    """
    # Distribution-shift score using features already computed from the
    # cached IDRiD training images.
    # `processed_rgb` is H x W x 3, while the cached model tensors
    # use C x H x W. Convert explicitly before concatenation.
    # The previous version attempted to concatenate HWC RGB with
    # CHW-style edge data, which caused:
    # "along dimension 2 ... size 3 ... size 384".
    rgb01 = processed_rgb.astype(np.float32) / 255.0
    if rgb01.ndim != 3 or rgb01.shape[2] != 3:
        raise ValueError(
            f"Unexpected processed RGB shape for review features: {rgb01.shape}"
        )
    rgb_chw = np.transpose(rgb01, (2, 0, 1))

    edge01 = edges.astype(np.float32) / 255.0
    if edge01.ndim == 3 and edge01.shape[2] == 1:
        edge01 = edge01[:, :, 0]
    elif edge01.ndim != 2:
        raise ValueError(
            f"Unexpected edge-map shape for review features: {edge01.shape}"
        )

    # Guarantee the spatial dimensions match before adding the edge channel.
    if rgb_chw.shape[1:] != edge01.shape:
        edge01 = cv2.resize(
            edge01,
            (rgb_chw.shape[2], rgb_chw.shape[1]),
            interpolation=cv2.INTER_NEAREST
        )

    feature_tensor = np.concatenate(
        [rgb_chw, edge01[None, ...]],
        axis=0
    )

    current_features = _image_shift_features(feature_tensor)
    z = np.abs(
        (current_features - OOD_REFERENCE_MEAN) / OOD_REFERENCE_STD
    )
    # Ignore the raw blob count/area from the single aggregate threshold
    # and require multiple independent features to be unusual.
    ood_feature_count = int(np.sum(z > 3.5))
    ood_flag = ood_feature_count >= 2

    # TTA stability: compare the probability assigned to each class across
    # the four existing augmentations. Large disagreement means the result
    # is less stable under harmless image transformations.
    tta_std = np.std(probs_stack, axis=0)
    tta_disagreement = float(np.max(tta_std))
    tta_flag = tta_disagreement >= 0.12

    # Conservative advanced-change review flag. This is deliberately NOT
    # a PDR detector. It combines a Grade-3/4 probability signal with
    # image-shift or TTA instability so unusual advanced cases are surfaced
    # without changing the model's medical label.
    high_grade_probability = float(probs[3] + probs[4])
    advanced_review_flag = (
        int(np.argmax(probs)) >= 3
        and high_grade_probability >= 0.55
        and (ood_flag or tta_flag)
    )

    return {
        "ood_flag": ood_flag,
        "ood_feature_count": ood_feature_count,
        "tta_flag": tta_flag,
        "tta_disagreement": tta_disagreement,
        "advanced_review_flag": advanced_review_flag,
        "high_grade_probability": high_grade_probability,
    }


@torch.no_grad()
def frontend_predict(image):
    """Make prediction on uploaded image with TTA."""

    if image is None:
        return (None, None, None, "No image uploaded", 0.0, "NORMAL", "Upload a retinal image to evaluate priority status.", pd.DataFrame())

    try:
        tensor, original, processed, edges = preprocess_uploaded_image(image)

    except Exception as e:
        print(f"Preprocessing error: {e}")
        return (None, None, None, f"Error: {str(e)[:100]}", 0.0, "ERROR", "⚠️ Image preprocessing failed.", pd.DataFrame())

    tensor = tensor.to(DEVICE)

    try:
        # Always force inference mode after training/evaluation.
        best_model.eval()

        # TTA 1: Original
        logits1 = best_model(tensor)
        probs1 = torch.softmax(logits1, dim=1)

        # TTA 2: Horizontal flip
        tensor_h = torch.flip(tensor, dims=[3])
        logits2 = best_model(tensor_h)
        probs2 = torch.softmax(logits2, dim=1)

        # TTA 3: Vertical flip
        tensor_v = torch.flip(tensor, dims=[2])
        logits3 = best_model(tensor_v)
        probs3 = torch.softmax(logits3, dim=1)

        # TTA 4: Rotation
        tensor_r = torch.rot90(tensor, k=1, dims=[2, 3])
        logits4 = best_model(tensor_r)
        probs4 = torch.softmax(logits4, dim=1)

        # Weighted ensemble
        probs_stack = torch.cat([probs1, probs2, probs3, probs4], dim=0)
        probs = (0.40 * probs1 + 0.25 * probs2 + 0.20 * probs3 + 0.15 * probs4)
        probs = probs[0].cpu().numpy()
        probs_stack_np = probs_stack.cpu().numpy()

    except Exception as e:
        print(f"Prediction error: {e}")
        return (None, None, None, f"Model error: {str(e)[:100]}", 0.0, "ERROR", "⚠️ Model inference failed.", pd.DataFrame())

    # Get the model's normal top prediction.
    model_pred = int(np.argmax(probs))
    model_confidence = float(probs[model_pred]) * 100.0

    # --------------------------------------------------------
    # PRIORITY COUNTER / SAFETY OVERRIDE
    # --------------------------------------------------------
    # If the second-highest grade is within <10 percentage points
    # of the highest probability AND that second grade is > 3,
    # prioritize the higher-severity grade.
    #
    # Example:
    #   Grade 3 = 46%, Grade 4 = 40%  -> Grade 4 is prioritized.
    #
    # This does NOT change the model probabilities. It only changes
    # the displayed priority classification and produces a warning.
    sorted_indices = np.argsort(probs)[::-1]
    top_idx = int(sorted_indices[0])
    second_idx = int(sorted_indices[1])

    top_prob = float(probs[top_idx]) * 100.0
    second_prob = float(probs[second_idx]) * 100.0
    probability_gap = top_prob - second_prob

    review_flags = _clinical_review_flags(
        processed, edges, probs_stack_np, probs
    )

    priority_override = (
        second_idx > 3
        and probability_gap < 10.0
    )

    if priority_override:
        pred = max(top_idx, second_idx)
        confidence = float(probs[pred]) * 100.0
        priority_level = "HIGH PRIORITY"
        warning_parts = [
            f"Grade {second_idx} has {second_prob:.2f}% probability and is "
            f"only {probability_gap:.2f} percentage points below Grade {top_idx} "
            f"({top_prob:.2f}%). Grade {pred} is prioritized for review."
        ]
    else:
        pred = model_pred
        confidence = model_confidence
        priority_level = "NORMAL"
        warning_parts = [
            f"No probability-based high-severity override. "
            f"Top prediction: Grade {pred} ({confidence:.2f}%)."
        ]

    # Training-free safety flags. These do not alter the probability table
    # and do not claim that PDR is present.
    if review_flags["advanced_review_flag"]:
        priority_level = "HIGH PRIORITY REVIEW"
        warning_parts.append(
            f"Advanced-change review flag: combined Grade 3/4 probability is "
            f"{review_flags['high_grade_probability'] * 100.0:.1f}%, while the "
            f"prediction is unstable or distribution-shifted. Grade 4/PDR "
            f"should not be excluded from specialist review."
        )

    if review_flags["tta_flag"]:
        warning_parts.append(
            f"TTA stability warning: maximum probability variation across "
            f"the four existing augmentations is {review_flags['tta_disagreement'] * 100.0:.1f} percentage points."
        )

    if review_flags["ood_flag"]:
        warning_parts.append(
            "Image-distribution warning: simple image characteristics differ "
            "substantially from the IDRiD training-image reference. "
            "Interpret the predicted grade cautiously."
        )

    priority_warning = "⚠️ " + " ".join(warning_parts) if (
        priority_override or review_flags["advanced_review_flag"] or
        review_flags["tta_flag"] or review_flags["ood_flag"]
    ) else " ".join(warning_parts)

    # Probability table
    probabilities_df = pd.DataFrame({
        "Grade": CLASS_NAMES,
        "Probability (%)": np.round(probs * 100.0, 2)
    })

    # Ensure output images are correct format
    original_output = np.ascontiguousarray(original).astype(np.uint8)
    processed_output = np.ascontiguousarray(processed).astype(np.uint8)
    edge_output = np.ascontiguousarray(edges).astype(np.uint8)

    return (
        original_output,
        processed_output,
        edge_output,
        f"Grade {pred}",
        confidence,
        priority_level,
        priority_warning,
        probabilities_df
    )



# ============================================================
# 48. PROFESSIONAL GRADIO FRONTEND
# ============================================================

print()
print("=" * 80)
print("STARTING PROFESSIONAL GRADIO INTERFACE")
print("=" * 80)

try:
    import gradio as gr
    import socket

    # --------------------------------------------------------
    # FIND AN AVAILABLE PORT
    # --------------------------------------------------------
    def find_free_port(start_port=7860, max_port=7999):
        """Return the first available TCP port."""
        for port in range(start_port, max_port + 1):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", port))
                sock.close()
                return port
            except OSError:
                sock.close()

        raise RuntimeError(
            f"No free port found in range {start_port}-{max_port}."
        )


    GRADIO_PORT = find_free_port(7860, 7999)

    print(f"Selected Gradio port: {GRADIO_PORT}")


    # --------------------------------------------------------
    # GRADIO CALLBACK
    # --------------------------------------------------------
    def analyze_image(image):
        """Run the existing frontend prediction pipeline."""
        if image is None:
            return (
                None,
                None,
                None,
                "Awaiting image",
                0.0,
                "NORMAL",
                "Upload a retinal image to evaluate priority status.",
                pd.DataFrame(
                    columns=["Grade", "Probability (%)"]
                )
            )

        return frontend_predict(image)


    # --------------------------------------------------------
    # PROFESSIONAL LIGHT BLUE / WHITE THEME
    # --------------------------------------------------------
    CUSTOM_CSS = """
    :root {
        --page-bg: #f7faff;
        --surface: #ffffff;
        --surface-soft: #f8fbfe;
        --border: #e3ebf3;
        --border-soft: #edf2f7;
        --text: #30465d;
        --text-soft: #607589;
        --text-muted: #718397;
        --blue: #2f80c9;
        --blue-dark: #276fae;
        --blue-soft: #edf6ff;
        --blue-border: #dcecf9;
        --shadow: rgba(40, 75, 110, 0.045);
    }

    body,
    .gradio-container {
        background: var(--page-bg) !important;
        color: var(--text) !important;
        font-family: Inter, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif !important;
    }

    /* Explicit text colors prevent Gradio's theme from producing
       low-contrast or white text on the light cards. */
    .gradio-container,
    .gradio-container p,
    .gradio-container span,
    .gradio-container div,
    .gradio-container label,
    .gradio-container .prose,
    .gradio-container .markdown-body {
        color: var(--text);
    }

    .gradio-container h1,
    .gradio-container h2,
    .gradio-container h3,
    .gradio-container h4,
    .gradio-container h5,
    .gradio-container h6 {
        color: #294159 !important;
    }

    .gradio-container p,
    .gradio-container li,
    .gradio-container .prose {
        color: #52677b !important;
    }

    .gradio-container .gr-form label,
    .gradio-container .gr-block-label,
    .gradio-container .wrap label,
    .gradio-container .label-wrap span {
        color: #50677c !important;
    }

    .gradio-container input,
    .gradio-container textarea,
    .gradio-container select,
    .gradio-container input::placeholder,
    .gradio-container textarea::placeholder {
        color: #425b70 !important;
        -webkit-text-fill-color: #425b70 !important;
    }

    .gradio-container input::placeholder,
    .gradio-container textarea::placeholder {
        color: #8796a5 !important;
        -webkit-text-fill-color: #8796a5 !important;
    }

    .gradio-container button:not(.primary),
    .gradio-container .secondary {
        color: #405a70 !important;
    }

    .gradio-container table,
    .gradio-container th,
    .gradio-container td {
        color: #52677b !important;
    }

    .gradio-container .tab-nav button {
        color: #52677b !important;
    }

    .gradio-container .tab-nav button.selected {
        color: #2f80c9 !important;
    }

    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto !important;
        padding: 28px 34px 44px !important;
    }

    .block,
    .form,
    .panel,
    .gr-box {
        border-color: var(--border) !important;
    }

    .app-header {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 3px 14px var(--shadow);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 15px;
    }

    .brand-icon {
        width: 46px;
        height: 46px;
        border-radius: 13px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--blue-soft);
        color: var(--blue);
        font-size: 23px;
        font-weight: 600;
    }

    .brand-title {
        margin: 0;
        color: #294159;
        font-size: 25px;
        line-height: 1.2;
        font-weight: 650;
        letter-spacing: -0.3px;
    }

    .brand-subtitle {
        margin-top: 5px;
        color: var(--text-soft);
        font-size: 13px;
    }

    .section-title {
        margin: 18px 0 7px 2px;
        color: #344d66;
        font-size: 17px;
        font-weight: 600;
    }

    .section-description {
        margin: 0 0 14px 2px;
        color: var(--text-soft);
        font-size: 13px;
        line-height: 1.6;
    }

    .soft-card {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 18px !important;
        box-shadow: 0 2px 12px var(--shadow) !important;
    }

    .result-card {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        min-height: 125px;
        box-shadow: 0 2px 10px var(--shadow) !important;
    }

    .priority-high textarea,
    .priority-high input {
        color: #a34b2c !important;
        -webkit-text-fill-color: #a34b2c !important;
        font-weight: 700 !important;
    }

    .priority-normal textarea,
    .priority-normal input {
        color: #456b82 !important;
        -webkit-text-fill-color: #456b82 !important;
        font-weight: 650 !important;
    }

    .warning-text {
        color: #8a5a2b !important;
        line-height: 1.55 !important;
    }

    .image-card {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 10px var(--shadow) !important;
    }

    .info-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 22px 25px;
        box-shadow: 0 2px 10px var(--shadow);
    }

    .metric-card {
        background: var(--surface-soft);
        border: 1px solid var(--border-soft);
        border-radius: 13px;
        padding: 15px 17px;
    }

    .metric-label {
        color: var(--text-muted);
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.7px;
        text-transform: uppercase;
    }

    .metric-value {
        color: #347bb4;
        font-size: 21px;
        font-weight: 650;
        margin-top: 5px;
    }

    .primary-action,
    button.primary {
        background: var(--blue) !important;
        border: none !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        min-height: 45px !important;
        box-shadow: none !important;
    }

    .primary-action:hover,
    button.primary:hover {
        background: var(--blue-dark) !important;
    }

    input,
    textarea,
    .gr-input {
        background: #ffffff !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
    }

    label {
        color: #64778a !important;
        font-weight: 500 !important;
    }

    .disclaimer {
        margin-top: 22px;
        padding: 14px 17px;
        background: #f8fafc;
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        color: #607589;
        font-size: 12px;
        line-height: 1.65;
    }

    .footer {
        text-align: center;
        color: #718397;
        font-size: 11px;
        padding: 22px 0 4px;
    }

    .reference-table {
        width: 100%;
        border-collapse: collapse;
        color: #637487;
        font-size: 13px;
    }

    .reference-table th {
        color: #728397;
        font-weight: 600;
        text-align: left;
        padding: 10px;
        border-bottom: 1px solid var(--border-soft);
    }

    .reference-table td {
        padding: 10px;
        border-bottom: 1px solid #f0f3f6;
        vertical-align: top;
    }

    .reference-table tr:last-child td {
        border-bottom: none;
    }

    @media (max-width: 800px) {
        .gradio-container {
            padding: 18px 14px 30px !important;
        }

        .app-header {
            padding: 20px;
        }

        .brand-title {
            font-size: 21px;
        }
    }
    """


    # ========================================================
    # BUILD INTERFACE
    # ========================================================

    with gr.Blocks(
        title="RetinaCare | Diabetic Retinopathy Analysis",
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=["Inter", "ui-sans-serif", "system-ui"]
        ),
        css=CUSTOM_CSS
    ) as demo:

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------
        gr.HTML(
            """
            <div class="app-header">
                <div class="brand">
                    <div class="brand-icon">◉</div>
                    <div>
                        <div class="brand-title">RetinaCare</div>
                        <div class="brand-subtitle">
                            AI-assisted diabetic retinopathy severity analysis
                        </div>
                    </div>
                </div>
            </div>
            """
        )

        # ----------------------------------------------------
        # UPLOAD SECTION
        # ----------------------------------------------------
        gr.HTML(
            """
            <div class="section-title">Retinal Image Assessment</div>
            <div class="section-description">
                Upload a retinal fundus image for severity classification.
                The analysis uses retinal ROI processing, contrast enhancement,
                illumination normalization and edge information.
            </div>
            """
        )

        with gr.Group(elem_classes="soft-card"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=4):
                    input_image = gr.Image(
                        type="numpy",
                        label="Retinal Fundus Image",
                        height=390
                    )

                with gr.Column(scale=2):
                    gr.Markdown(
                        """
                        **Analysis pipeline**

                        1. Retinal ROI detection
                        2. Square crop and resize
                        3. CLAHE enhancement
                        4. Illumination normalization
                        5. Canny edge extraction
                        6. RGB + edge model inference
                        """
                    )

                    analyze_button = gr.Button(
                        "Analyze Image",
                        variant="primary",
                        elem_classes="primary-action"
                    )

        # ----------------------------------------------------
        # IMAGE PROCESSING
        # ----------------------------------------------------
        gr.HTML(
            """
            <div class="section-title">Image Processing</div>
            <div class="section-description">
                Visual representation of the preprocessing stages used by the model.
            </div>
            """
        )

        with gr.Row():
            with gr.Column(elem_classes="image-card"):
                original_output = gr.Image(
                    label="Original Image",
                    type="numpy",
                    height=300
                )

            with gr.Column(elem_classes="image-card"):
                processed_output = gr.Image(
                    label="Processed Retinal ROI",
                    type="numpy",
                    height=300
                )

            with gr.Column(elem_classes="image-card"):
                edge_output = gr.Image(
                    label="Canny Edge Map",
                    type="numpy",
                    height=300
                )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------
        gr.HTML(
            """
            <div class="section-title">Analysis Result</div>
            <div class="section-description">
                Predicted diabetic retinopathy grade and model confidence.
                Safety flags can identify uncertain, unstable, or distribution-shifted
                images for additional review without retraining the model.
            </div>
            """
        )

        with gr.Row():
            with gr.Column(elem_classes="result-card"):
                gr.Markdown("**PREDICTED SEVERITY**")
                prediction_output = gr.Textbox(
                    label="",
                    show_label=False,
                    value="Awaiting image",
                    interactive=False
                )

            with gr.Column(elem_classes="result-card"):
                gr.Markdown("**MODEL CONFIDENCE**")
                confidence_output = gr.Number(
                    label="",
                    show_label=False,
                    value=0.0,
                    interactive=False
                )

        with gr.Row():
            with gr.Column(elem_classes="result-card"):
                gr.Markdown("**PRIORITY COUNTER**")
                priority_output = gr.Textbox(
                    label="",
                    show_label=False,
                    value="NORMAL",
                    interactive=False
                )

            with gr.Column(elem_classes="result-card"):
                gr.Markdown("**PRIORITY WARNING**")
                priority_warning_output = gr.Textbox(
                    label="",
                    show_label=False,
                    value="Upload a retinal image to evaluate priority status.",
                    interactive=False,
                    lines=5
                )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------
        gr.HTML(
            """
            <div class="section-title">Classification Probabilities</div>
            <div class="section-description">
                Probability distribution across the five IDRiD severity grades.
            </div>
            """
        )

        with gr.Group(elem_classes="soft-card"):
            probability_output = gr.Dataframe(
                headers=["Grade", "Probability (%)"],
                datatype=["str", "number"],
                interactive=False,
                wrap=True,
                label=""
            )

        # ----------------------------------------------------
        # CLINICAL REFERENCE
        # ----------------------------------------------------
        gr.HTML(
            """
            <div class="section-title">Severity Reference</div>
            <div class="section-description">
                Reference descriptions for the five model output classes.
            </div>

            <div class="info-card">
                <table class="reference-table">
                    <thead>
                        <tr>
                            <th>Grade</th>
                            <th>Severity</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Grade 0</strong></td>
                            <td>No apparent DR</td>
                            <td>No visible retinopathy lesions</td>
                        </tr>
                        <tr>
                            <td><strong>Grade 1</strong></td>
                            <td>Mild DR</td>
                            <td>Microaneurysms</td>
                        </tr>
                        <tr>
                            <td><strong>Grade 2</strong></td>
                            <td>Moderate DR</td>
                            <td>Microaneurysms with hemorrhages and/or hard exudates</td>
                        </tr>
                        <tr>
                            <td><strong>Grade 3</strong></td>
                            <td>Severe DR</td>
                            <td>Extensive hemorrhages, venous beading or IRMA</td>
                        </tr>
                        <tr>
                            <td><strong>Grade 4</strong></td>
                            <td>Proliferative DR</td>
                            <td>Neovascularization</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """
        )

        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------
        gr.HTML(
            f"""
            <div class="section-title">Model Information</div>

            <div class="info-card">
                <div style="
                    display:grid;
                    grid-template-columns:repeat(auto-fit,minmax(175px,1fr));
                    gap:12px;
                ">

                    <div class="metric-card">
                        <div class="metric-label">Architecture</div>
                        <div class="metric-value" style="font-size:16px;">
                            EfficientNet-V2-S
                        </div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Input</div>
                        <div class="metric-value" style="font-size:16px;">
                            RGB + Edge
                        </div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Resolution</div>
                        <div class="metric-value" style="font-size:16px;">
                            384 × 384
                        </div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Training Images</div>
                        <div class="metric-value">{len(train_part)}</div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Validation Images</div>
                        <div class="metric-value">{len(val_part)}</div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Official Test Images</div>
                        <div class="metric-value">{len(test_df)}</div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">TTA</div>
                        <div class="metric-value" style="font-size:16px;">
                            Enabled
                        </div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Best Seed</div>
                        <div class="metric-value">{BEST_SEED}</div>
                    </div>

                </div>
            </div>
            """
        )

        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------
        gr.HTML(
            f"""
            <div class="section-title">Model Performance</div>
            <div class="section-description">
                Metrics generated during the completed experiment.
            </div>

            <div class="info-card">
                <div style="
                    display:grid;
                    grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
                    gap:12px;
                ">

                    <div class="metric-card">
                        <div class="metric-label">Validation Accuracy</div>
                        <div class="metric-value">
                            {best_result['val_metrics']['accuracy']:.2%}
                        </div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Validation Balanced Accuracy</div>
                        <div class="metric-value">
                            {best_result['val_metrics']['balanced_accuracy']:.2%}
                        </div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Validation Macro-F1</div>
                        <div class="metric-value">
                            {best_result['val_metrics']['macro_f1']:.4f}
                        </div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Test Accuracy</div>
                        <div class="metric-value">
                            {test_metrics['accuracy']:.2%}
                        </div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Test Balanced Accuracy</div>
                        <div class="metric-value">
                            {test_metrics['balanced_accuracy']:.2%}
                        </div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Test Macro-F1</div>
                        <div class="metric-value">
                            {test_metrics['macro_f1']:.4f}
                        </div>
                    </div>

                </div>
            </div>
            """
        )

        # ----------------------------------------------------
        # TECHNICAL DETAILS
        # ----------------------------------------------------
        gr.HTML(
            """
            <div class="section-title">Technical Details</div>

            <div class="info-card">
                <div style="
                    display:grid;
                    grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
                    gap:20px;
                ">

                    <div>
                        <strong style="color:#40566d;">Preprocessing</strong>
                        <ul style="color:#718195;line-height:1.8;">
                            <li>Otsu retinal ROI detection</li>
                            <li>Square crop and 384 × 384 resize</li>
                            <li>CLAHE enhancement</li>
                            <li>Illumination normalization</li>
                            <li>Canny edge extraction</li>
                            <li>RGB + edge concatenation</li>
                        </ul>
                    </div>

                    <div>
                        <strong style="color:#40566d;">Training</strong>
                        <ul style="color:#718195;line-height:1.8;">
                            <li>Multi-seed training</li>
                            <li>Focal loss + label smoothing</li>
                            <li>Weighted class balancing</li>
                            <li>Differential learning rates</li>
                            <li>Cosine annealing</li>
                            <li>Early stopping</li>
                        </ul>
                    </div>

                    <div>
                        <strong style="color:#40566d;">Test-Time Augmentation</strong>
                        <ul style="color:#718195;line-height:1.8;">
                            <li>Original image — 40%</li>
                            <li>Horizontal flip — 25%</li>
                            <li>Vertical flip — 20%</li>
                            <li>90° rotation — 15%</li>
                        </ul>
                    </div>

                </div>
            </div>
            """
        )

        # ----------------------------------------------------
        # DISCLAIMER
        # ----------------------------------------------------
        gr.HTML(
            """
            <div class="disclaimer">
                <strong>Research Prototype</strong><br>
                This system is intended for research and educational purposes only.
                It is not a clinical diagnostic device and should not replace
                examination or diagnosis by a qualified ophthalmologist.
            </div>
            """
        )

        gr.HTML(
            """
            <div class="footer">
                RetinaCare · IDRiD Diabetic Retinopathy Research System
            </div>
            """
        )

        # ----------------------------------------------------
        # EVENT HANDLER
        # ----------------------------------------------------
        analyze_button.click(
            fn=analyze_image,
            inputs=[input_image],
            outputs=[
                original_output,
                processed_output,
                edge_output,
                prediction_output,
                confidence_output,
                priority_output,
                priority_warning_output,
                probability_output
            ],
            queue=False
        )


    # ========================================================
    # LAUNCH GRADIO
    # ========================================================

    print()
    print("=" * 80)
    print("RETINACARE GRADIO INTERFACE READY")
    print("=" * 80)
    print()
    print(f"Local server: http://127.0.0.1:{GRADIO_PORT}")
    print("Public URL: Gradio will generate it below")
    print()

    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        show_error=True,
        inbrowser=False
    )


except ImportError:
    print()
    print("=" * 80)
    print("GRADIO NOT INSTALLED")
    print("=" * 80)
    print()
    print("Install Gradio with:")
    print("!pip install -U gradio")
    print()


except Exception as e:
    print()
    print("=" * 80)
    print("GRADIO ERROR")
    print("=" * 80)
    print()
    print(f"Error type: {type(e).__name__}")
    print(f"Error: {e}")
    print()

    import traceback
    traceback.print_exc()
