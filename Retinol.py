# ============================================================
# IDRiD DIABETIC RETINOPATHY GRADING
# RESNET18 + KAGGLE + GRADIO FRONTEND
#
# IMPORTANT:
# This version explicitly selects:
# B. Disease Grading
#
# It will NOT accidentally select:
# A. Segmentation
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import glob
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("SYSTEM")
print("=" * 70)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 70)


# ============================================================
# 4. CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

NUM_CLASSES = 5

BATCH_SIZE = 32

NUM_WORKERS = 2

VAL_SIZE = 0.20

RANDOM_STATE = 42

# Stage 1
STAGE1_EPOCHS = 8
STAGE1_LR = 1e-3

# Stage 2
STAGE2_EPOCHS = 12
STAGE2_LR = 1e-5

WEIGHT_DECAY = 1e-4

PATIENCE = 4

MAX_UPLOAD_MB = 10

MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

CLASS_NAMES = [
    "Grade 0",
    "Grade 1",
    "Grade 2",
    "Grade 3",
    "Grade 4"
]

MODEL_PATH = "/kaggle/working/resnet18_idrid_best.pt"

HISTORY_PATH = "/kaggle/working/resnet18_training_history.csv"

TEST_RESULTS_PATH = "/kaggle/working/resnet18_test_predictions.csv"

CLASS_JSON_PATH = "/kaggle/working/classes.json"


# ============================================================
# 5. DATASET DISCOVERY
# ============================================================
#
# IMPORTANT:
# We explicitly search for:
#
# B. Disease Grading
#
# and never simply search for "Training Set".
# ============================================================

INPUT_ROOT = Path("/kaggle/input")


def find_disease_grading_path():

    candidates = []

    # Normal Kaggle path
    candidates.extend(
        INPUT_ROOT.glob(
            "**/B. Disease Grading/1. Original Images/a. Training Set"
        )
    )

    # URL encoded path
    candidates.extend(
        INPUT_ROOT.glob(
            "**/B.%20Disease%20Grading/1. Original Images/a. Training Set"
        )
    )

    # Search manually as final fallback
    for p in INPUT_ROOT.rglob("a. Training Set"):

        p_string = str(p)

        if (
            "Disease Grading" in p_string
            or "Disease%20Grading" in p_string
        ):
            candidates.append(p)

    # Remove duplicates
    unique_candidates = []

    for p in candidates:
        p = Path(p)

        if p not in unique_candidates:
            unique_candidates.append(p)

    if len(unique_candidates) == 0:

        print("\nCould not find Disease Grading directory.")

        print("\nPossible training directories found:")

        for p in INPUT_ROOT.rglob("a. Training Set"):
            print(" ", p)

        raise FileNotFoundError(
            "B. Disease Grading training directory was not found."
        )

    # Explicitly reject segmentation
    valid_candidates = [
        p for p in unique_candidates
        if "Segmentation" not in str(p)
    ]

    if len(valid_candidates) == 0:
        raise RuntimeError(
            "Only segmentation directories were found. "
            "Disease Grading dataset is missing."
        )

    return valid_candidates[0]


TRAIN_DIR = find_disease_grading_path()


# ============================================================
# 6. FIND TEST DIRECTORY
# ============================================================

TEST_DIR = TRAIN_DIR.parent / "b. Testing Set"

if not TEST_DIR.exists():

    # Try encoded / alternate discovery
    test_candidates = []

    test_candidates.extend(
        INPUT_ROOT.glob(
            "**/B. Disease Grading/1. Original Images/b. Testing Set"
        )
    )

    test_candidates.extend(
        INPUT_ROOT.glob(
            "**/B.%20Disease%20Grading/1. Original Images/b. Testing Set"
        )
    )

    if len(test_candidates) > 0:
        TEST_DIR = test_candidates[0]


# ============================================================
# 7. FIND GROUNDTRUTH CSV FILES
# ============================================================

training_csv_candidates = []

testing_csv_candidates = []


# Normal names
training_csv_candidates.extend(
    INPUT_ROOT.glob(
        "**/B. Disease Grading/2. Groundtruths/"
        "a. IDRiD_Disease Grading_Training Labels.csv"
    )
)

testing_csv_candidates.extend(
    INPUT_ROOT.glob(
        "**/B. Disease Grading/2. Groundtruths/"
        "b. IDRiD_Disease Grading_Testing Labels.csv"
    )
)


# URL encoded names
training_csv_candidates.extend(
    INPUT_ROOT.glob(
        "**/B.%20Disease%20Grading/2. Groundtruths/"
        "a. IDRiD_Disease Grading_Training Labels.csv"
    )
)

testing_csv_candidates.extend(
    INPUT_ROOT.glob(
        "**/B.%20Disease%20Grading/2. Groundtruths/"
        "b. IDRiD_Disease Grading_Testing Labels.csv"
    )
)


# Final recursive fallback
for p in INPUT_ROOT.rglob("*Training Labels.csv"):

    if (
        "Disease Grading" in str(p)
        or "Disease%20Grading" in str(p)
    ):
        training_csv_candidates.append(p)


for p in INPUT_ROOT.rglob("*Testing Labels.csv"):

    if (
        "Disease Grading" in str(p)
        or "Disease%20Grading" in str(p)
    ):
        testing_csv_candidates.append(p)


# Remove duplicates
training_csv_candidates = list(
    dict.fromkeys(map(Path, training_csv_candidates))
)

testing_csv_candidates = list(
    dict.fromkeys(map(Path, testing_csv_candidates))
)


if len(training_csv_candidates) == 0:
    raise FileNotFoundError(
        "Disease Grading training labels CSV not found."
    )

if len(testing_csv_candidates) == 0:
    raise FileNotFoundError(
        "Disease Grading testing labels CSV not found."
    )


TRAIN_CSV = training_csv_candidates[0]

TEST_CSV = testing_csv_candidates[0]


# ============================================================
# 8. PRINT DATASET PATHS
# ============================================================

print("\n")
print("=" * 70)
print("DATASET PATHS")
print("=" * 70)

print("TRAIN_DIR:")
print(TRAIN_DIR)

print("\nTEST_DIR:")
print(TEST_DIR)

print("\nTRAIN_CSV:")
print(TRAIN_CSV)

print("\nTEST_CSV:")
print(TEST_CSV)

print("=" * 70)


# ============================================================
# 9. VERIFY DATASET DIRECTORIES
# ============================================================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training directory does not exist:\n{TRAIN_DIR}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Testing directory does not exist:\n{TEST_DIR}"
    )


# ============================================================
# 10. IMAGE EXTENSIONS
# ============================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# ============================================================
# 11. VERIFY TRAINING IMAGES
# ============================================================

train_images = [
    p for p in TRAIN_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in VALID_EXTENSIONS
]


test_images = [
    p for p in TEST_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in VALID_EXTENSIONS
]


print("\nTraining images found:", len(train_images))
print("Testing images found :", len(test_images))


if len(train_images) == 0:
    raise RuntimeError(
        f"No valid images found in:\n{TRAIN_DIR}"
    )


if len(test_images) == 0:
    raise RuntimeError(
        f"No valid images found in:\n{TEST_DIR}"
    )


# ============================================================
# 12. LOAD LABEL CSVs
# ============================================================

train_labels_df = pd.read_csv(TRAIN_CSV)

test_labels_df = pd.read_csv(TEST_CSV)


print("\nTraining labels shape:", train_labels_df.shape)

print("Testing labels shape :", test_labels_df.shape)

print("\nTraining CSV columns:")
print(train_labels_df.columns.tolist())

print("\nTesting CSV columns:")
print(test_labels_df.columns.tolist())


# ============================================================
# 13. NORMALIZE COLUMN NAMES
# ============================================================

train_labels_df.columns = [
    str(c).strip()
    for c in train_labels_df.columns
]

test_labels_df.columns = [
    str(c).strip()
    for c in test_labels_df.columns
]


# ============================================================
# 14. FIND REQUIRED LABEL COLUMNS
# ============================================================

IMAGE_COLUMN = "Image name"

LABEL_COLUMN = "Retinopathy grade"


if IMAGE_COLUMN not in train_labels_df.columns:

    raise RuntimeError(
        f"'{IMAGE_COLUMN}' column not found in training CSV."
    )


if LABEL_COLUMN not in train_labels_df.columns:

    raise RuntimeError(
        f"'{LABEL_COLUMN}' column not found in training CSV."
    )


if IMAGE_COLUMN not in test_labels_df.columns:

    raise RuntimeError(
        f"'{IMAGE_COLUMN}' column not found in testing CSV."
    )


if LABEL_COLUMN not in test_labels_df.columns:

    raise RuntimeError(
        f"'{LABEL_COLUMN}' column not found in testing CSV."
    )


# ============================================================
# 15. CLEAN LABELS
# ============================================================

train_labels_df = train_labels_df[
    [IMAGE_COLUMN, LABEL_COLUMN]
].copy()

test_labels_df = test_labels_df[
    [IMAGE_COLUMN, LABEL_COLUMN]
].copy()


train_labels_df[LABEL_COLUMN] = pd.to_numeric(
    train_labels_df[LABEL_COLUMN],
    errors="coerce"
)

test_labels_df[LABEL_COLUMN] = pd.to_numeric(
    test_labels_df[LABEL_COLUMN],
    errors="coerce"
)


train_labels_df = train_labels_df.dropna()

test_labels_df = test_labels_df.dropna()


train_labels_df[LABEL_COLUMN] = (
    train_labels_df[LABEL_COLUMN]
    .astype(int)
)

test_labels_df[LABEL_COLUMN] = (
    test_labels_df[LABEL_COLUMN]
    .astype(int)
)


# ============================================================
# 16. BUILD IMAGE MAP
# ============================================================

def build_image_map(directory):

    image_map = {}

    for p in directory.iterdir():

        if (
            p.is_file()
            and p.suffix.lower() in VALID_EXTENSIONS
        ):

            image_map[p.stem] = p

            image_map[p.name] = p

    return image_map


train_image_map = build_image_map(TRAIN_DIR)

test_image_map = build_image_map(TEST_DIR)


# ============================================================
# 17. RESOLVE IMAGE PATH
# ============================================================

def resolve_image_path(image_name, image_map):

    image_name = str(image_name).strip()

    # Direct filename
    if image_name in image_map:
        return image_map[image_name]

    # Stem
    stem = Path(image_name).stem

    if stem in image_map:
        return image_map[stem]

    # Search by stem
    for key, path in image_map.items():

        if Path(key).stem == stem:
            return path

    return None


# ============================================================
# 18. CREATE TRAIN RECORDS
# ============================================================

train_records = []

for _, row in train_labels_df.iterrows():

    image_path = resolve_image_path(
        row[IMAGE_COLUMN],
        train_image_map
    )

    if image_path is not None:

        train_records.append(
            {
                "image": str(image_path),
                "label": int(row[LABEL_COLUMN])
            }
        )


# ============================================================
# 19. CREATE TEST RECORDS
# ============================================================

test_records = []

for _, row in test_labels_df.iterrows():

    image_path = resolve_image_path(
        row[IMAGE_COLUMN],
        test_image_map
    )

    if image_path is not None:

        test_records.append(
            {
                "image": str(image_path),
                "label": int(row[LABEL_COLUMN])
            }
        )


# ============================================================
# 20. DATASET MATCHING CHECK
# ============================================================

print("\n")
print("=" * 70)
print("IMAGE / LABEL MATCHING")
print("=" * 70)

print(
    "Training CSV rows:",
    len(train_labels_df)
)

print(
    "Matched training images:",
    len(train_records)
)

print(
    "Testing CSV rows:",
    len(test_labels_df)
)

print(
    "Matched testing images:",
    len(test_records)
)

print("=" * 70)


if len(train_records) == 0:

    raise RuntimeError(
        "No training images matched the training labels."
    )


if len(test_records) == 0:

    raise RuntimeError(
        "No testing images matched the testing labels."
    )


# ============================================================
# 21. VERIFY LABEL RANGE
# ============================================================

train_labels = np.array(
    [r["label"] for r in train_records]
)

test_labels = np.array(
    [r["label"] for r in test_records]
)


print("\nTraining class distribution:")

print(
    pd.Series(train_labels)
    .value_counts()
    .sort_index()
)


print("\nTesting class distribution:")

print(
    pd.Series(test_labels)
    .value_counts()
    .sort_index()
)


invalid_train = [
    x for x in train_labels
    if x < 0 or x >= NUM_CLASSES
]

invalid_test = [
    x for x in test_labels
    if x < 0 or x >= NUM_CLASSES
]


if invalid_train:

    raise RuntimeError(
        f"Invalid training labels found: {invalid_train}"
    )


if invalid_test:

    raise RuntimeError(
        f"Invalid testing labels found: {invalid_test}"
    )


# ============================================================
# 22. STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

train_indices, val_indices = train_test_split(
    np.arange(len(train_records)),
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE,
    stratify=train_labels
)


train_records_split = [
    train_records[i]
    for i in train_indices
]


val_records_split = [
    train_records[i]
    for i in val_indices
]


print("\n")
print("=" * 70)
print("SPLIT")
print("=" * 70)

print("Training:", len(train_records_split))
print("Validation:", len(val_records_split))
print("Official test:", len(test_records))

print("=" * 70)


# ============================================================
# 23. IMAGE TRANSFORMS
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# Training augmentation
train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])


# Validation/test/inference
eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])


# ============================================================
# 24. DATASET CLASS
# ============================================================

class IDRiDRGBDataset(Dataset):

    def __init__(
        self,
        records,
        transform=None
    ):

        self.records = records
        self.transform = transform

        if len(self.records) == 0:

            raise RuntimeError(
                "Dataset contains zero records."
            )

    def __len__(self):

        return len(self.records)

    def __getitem__(self, idx):

        record = self.records[idx]

        image_path = record["image"]

        label = int(record["label"])

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            raise RuntimeError(
                f"Could not load image:\n"
                f"{image_path}\n"
                f"Error: {e}"
            )

        if self.transform is not None:

            image = self.transform(image)

        return image, label


# ============================================================
# 25. CREATE DATASETS
# ============================================================

train_dataset = IDRiDRGBDataset(
    train_records_split,
    transform=train_transform
)


val_dataset = IDRiDRGBDataset(
    val_records_split,
    transform=eval_transform
)


test_dataset = IDRiDRGBDataset(
    test_records,
    transform=eval_transform
)


# ============================================================
# 26. DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0
)


# ============================================================
# 27. TEST DATA LOADING
# ============================================================

sample_images, sample_labels = next(
    iter(train_loader)
)

print("\n")
print("=" * 70)
print("DATALOADER CHECK")
print("=" * 70)

print("Image batch shape:", sample_images.shape)

print("Label batch shape:", sample_labels.shape)

print("Image dtype:", sample_images.dtype)

print("Labels:", sample_labels.tolist())

print("=" * 70)


# ============================================================
# 28. CLASS WEIGHTS
# ============================================================

train_split_labels = np.array(
    [
        r["label"]
        for r in train_records_split
    ]
)


class_counts = np.bincount(
    train_split_labels,
    minlength=NUM_CLASSES
)


print("\nClass counts:")

for i, count in enumerate(class_counts):

    print(
        f"{CLASS_NAMES[i]}: {count}"
    )


class_weights = (
    len(train_split_labels)
    /
    (
        NUM_CLASSES
        *
        np.maximum(class_counts, 1)
    )
)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)


print("\nClass weights:")

print(class_weights)


# ============================================================
# 29. CREATE RESNET18
# ============================================================

print("\n")
print("=" * 70)
print("CREATING RESNET18")
print("=" * 70)


weights = ResNet18_Weights.DEFAULT

model = resnet18(
    weights=weights
)


# Replace classifier
in_features = model.fc.in_features


model.fc = nn.Sequential(

    nn.Dropout(
        p=0.35
    ),

    nn.Linear(
        in_features,
        NUM_CLASSES
    )
)


model = model.to(DEVICE)


print(model.fc)

print("=" * 70)


# ============================================================
# 30. LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# 31. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1
    }


# ============================================================
# 32. TRAINING EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0.0

    all_true = []
    all_pred = []

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_true.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_pred.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    epoch_loss = (
        total_loss
        /
        len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_true,
        all_pred
    )

    return epoch_loss, metrics


# ============================================================
# 33. VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    all_true = []
    all_pred = []
    all_prob = []

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_true.extend(
            labels.cpu()
            .numpy()
            .tolist()
        )

        all_pred.extend(
            predictions.cpu()
            .numpy()
            .tolist()
        )

        all_prob.append(
            probabilities.cpu()
            .numpy()
        )

    epoch_loss = (
        total_loss
        /
        len(loader.dataset)
    )

    all_prob = np.concatenate(
        all_prob,
        axis=0
    )

    metrics = calculate_metrics(
        all_true,
        all_pred
    )

    return (
        epoch_loss,
        metrics,
        np.array(all_true),
        np.array(all_pred),
        all_prob
    )


# ============================================================
# 34. HISTORY
# ============================================================

history = []

best_macro_f1 = -np.inf

best_epoch = 0

patience_counter = 0


# ============================================================
# 35. STAGE 1
# ============================================================
#
# Freeze backbone.
# Train classifier first.
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 1 — FROZEN BACKBONE")
print("=" * 70)


for param in model.parameters():

    param.requires_grad = False


for param in model.fc.parameters():

    param.requires_grad = True


optimizer = optim.AdamW(
    model.fc.parameters(),
    lr=STAGE1_LR,
    weight_decay=WEIGHT_DECAY
)


for epoch in range(1, STAGE1_EPOCHS + 1):

    train_loss, train_metrics = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )


    val_loss, val_metrics, _, _, _ = evaluate_model(
        model,
        val_loader,
        criterion
    )


    row = {
        "stage": 1,
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_metrics["accuracy"],
        "train_balanced_accuracy": train_metrics["balanced_accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "val_loss": val_loss,
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_macro_f1": val_metrics["macro_f1"]
    }

    history.append(row)


    print(
        f"Stage 1 | "
        f"Epoch {epoch:02d}/{STAGE1_EPOCHS} | "
        f"Train Loss {train_loss:.4f} | "
        f"Train F1 {train_metrics['macro_f1']:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val F1 {val_metrics['macro_f1']:.4f}"
    )


    if val_metrics["macro_f1"] > best_macro_f1:

        best_macro_f1 = (
            val_metrics["macro_f1"]
        )

        best_epoch = epoch

        patience_counter = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_macro_f1": best_macro_f1,
                "epoch": epoch,
                "class_names": CLASS_NAMES
            },
            MODEL_PATH
        )

        print(
            "  -> Best model saved."
        )

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                "Early stopping Stage 1."
            )

            break


# ============================================================
# 36. LOAD BEST STAGE 1 MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ============================================================
# 37. STAGE 2
# ============================================================
#
# Unfreeze layer4 + classifier.
# Fine-tune carefully.
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 2 — FINE TUNING")
print("=" * 70)


for param in model.parameters():

    param.requires_grad = False


for param in model.layer4.parameters():

    param.requires_grad = True


for param in model.fc.parameters():

    param.requires_grad = True


trainable_parameters = [
    p for p in model.parameters()
    if p.requires_grad
]


optimizer = optim.AdamW(
    trainable_parameters,
    lr=STAGE2_LR,
    weight_decay=WEIGHT_DECAY
)


patience_counter = 0


for epoch in range(
    1,
    STAGE2_EPOCHS + 1
):

    train_loss, train_metrics = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )


    val_loss, val_metrics, _, _, _ = evaluate_model(
        model,
        val_loader,
        criterion
    )


    row = {
        "stage": 2,
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_metrics["accuracy"],
        "train_balanced_accuracy": train_metrics["balanced_accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "val_loss": val_loss,
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_macro_f1": val_metrics["macro_f1"]
    }

    history.append(row)


    print(
        f"Stage 2 | "
        f"Epoch {epoch:02d}/{STAGE2_EPOCHS} | "
        f"Train Loss {train_loss:.4f} | "
        f"Train F1 {train_metrics['macro_f1']:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val F1 {val_metrics['macro_f1']:.4f}"
    )


    if val_metrics["macro_f1"] > best_macro_f1:

        best_macro_f1 = (
            val_metrics["macro_f1"]
        )

        best_epoch = len(history)

        patience_counter = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_macro_f1": best_macro_f1,
                "epoch": best_epoch,
                "class_names": CLASS_NAMES
            },
            MODEL_PATH
        )

        print(
            "  -> Best model saved."
        )

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                "Early stopping Stage 2."
            )

            break


# ============================================================
# 38. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# ============================================================
# 39. LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "Best validation Macro-F1:",
    checkpoint["best_macro_f1"]
)

print(
    "Checkpoint:",
    MODEL_PATH
)

print("=" * 70)


# ============================================================
# 40. SAVE CLASS NAMES
# ============================================================

with open(
    CLASS_JSON_PATH,
    "w"
) as f:

    json.dump(
        {
            "classes": CLASS_NAMES
        },
        f,
        indent=4
    )


# ============================================================
# 41. OFFICIAL TEST EVALUATION
# ============================================================
#
# IMPORTANT:
# The official test set is evaluated only after
# model selection.
# ============================================================

print("\n")
print("=" * 70)
print("OFFICIAL TEST EVALUATION")
print("=" * 70)


test_loss, test_metrics, y_true, y_pred, y_prob = evaluate_model(
    model,
    test_loader,
    criterion
)


print(
    f"Test Loss:              {test_loss:.4f}"
)

print(
    f"Test Accuracy:          {test_metrics['accuracy']:.4f}"
)

print(
    f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}"
)

print(
    f"Test Macro-F1:          {test_metrics['macro_f1']:.4f}"
)


# ============================================================
# 42. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# 43. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES))
)


print("Confusion Matrix:")

print(cm)


# ============================================================
# 44. SAVE TEST PREDICTIONS
# ============================================================

test_prediction_rows = []


for i, record in enumerate(test_records):

    predicted_class = int(
        y_pred[i]
    )

    confidence = float(
        np.max(y_prob[i])
    )

    row = {
        "image": Path(
            record["image"]
        ).name,

        "true_grade": int(
            y_true[i]
        ),

        "predicted_grade": predicted_class,

        "predicted_class": CLASS_NAMES[
            predicted_class
        ],

        "confidence": confidence
    }

    for class_index in range(NUM_CLASSES):

        row[
            f"prob_grade_{class_index}"
        ] = float(
            y_prob[i][class_index]
        )

    test_prediction_rows.append(
        row
    )


test_predictions_df = pd.DataFrame(
    test_prediction_rows
)


test_predictions_df.to_csv(
    TEST_RESULTS_PATH,
    index=False
)


# ============================================================
# 45. PLOT CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "IDRiD Official Test Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    range(NUM_CLASSES),
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)

plt.xlabel(
    "Predicted Grade"
)

plt.ylabel(
    "True Grade"
)

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

CONFUSION_PATH = (
    "/kaggle/working/"
    "resnet18_confusion_matrix.png"
)

plt.savefig(
    CONFUSION_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 46. TRAINING CURVES
# ============================================================

history_df = pd.DataFrame(
    history
)


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    history_df["val_macro_f1"],
    marker="o",
    label="Validation Macro-F1"
)

plt.plot(
    history_df["train_macro_f1"],
    marker="o",
    label="Training Macro-F1"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Macro-F1"
)

plt.title(
    "Training / Validation Macro-F1"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

F1_CURVE_PATH = (
    "/kaggle/working/"
    "resnet18_f1_curve.png"
)

plt.savefig(
    F1_CURVE_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    history_df["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.plot(
    history_df["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training / Validation Loss"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

LOSS_CURVE_PATH = (
    "/kaggle/working/"
    "resnet18_loss_curve.png"
)

plt.savefig(
    LOSS_CURVE_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 47. FRONTEND
# ============================================================

print("\n")
print("=" * 70)
print("PREPARING FRONTEND")
print("=" * 70)


import gradio as gr


# ============================================================
# 48. LOAD FRONTEND MODEL
# ============================================================

frontend_model = resnet18(
    weights=None
)


frontend_model.fc = nn.Sequential(

    nn.Dropout(
        p=0.35
    ),

    nn.Linear(
        frontend_model.fc.in_features,
        NUM_CLASSES
    )
)


frontend_checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


frontend_model.load_state_dict(
    frontend_checkpoint["model_state_dict"]
)


frontend_model = frontend_model.to(
    DEVICE
)

frontend_model.eval()


# ============================================================
# 49. FRONTEND PREPROCESSING
# ============================================================
#
# This is the SAME deterministic preprocessing used
# for validation and official testing.
# ============================================================

frontend_transform = eval_transform


# ============================================================
# 50. INPUT VALIDATION
# ============================================================

def validate_uploaded_file(
    file_path
):

    if file_path is None:

        raise ValueError(
            "Please upload an image."
        )


    file_path = str(
        file_path
    )


    if not os.path.exists(
        file_path
    ):

        raise ValueError(
            "Uploaded file could not be found."
        )


    # --------------------------------------------------------
    # File size barrier
    # --------------------------------------------------------

    file_size = os.path.getsize(
        file_path
    )


    if file_size > MAX_UPLOAD_BYTES:

        actual_mb = (
            file_size
            /
            (1024 * 1024)
        )

        raise ValueError(
            f"File is too large: "
            f"{actual_mb:.2f} MB. "
            f"Maximum allowed size is "
            f"{MAX_UPLOAD_MB} MB."
        )


    # --------------------------------------------------------
    # Extension barrier
    # --------------------------------------------------------

    extension = Path(
        file_path
    ).suffix.lower()


    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }


    if extension not in allowed_extensions:

        raise ValueError(
            "Unsupported image format. "
            "Use JPG, JPEG, PNG, BMP, TIFF or TIF."
        )


    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    try:

        image = Image.open(
            file_path
        )

        image.verify()

        image = Image.open(
            file_path
        ).convert("RGB")

    except Exception:

        raise ValueError(
            "The uploaded file is not a valid image."
        )


    # --------------------------------------------------------
    # Dimension barrier
    # --------------------------------------------------------

    width, height = image.size


    if width < 100 or height < 100:

        raise ValueError(
            f"Image resolution is too small "
            f"({width} × {height}). "
            f"Minimum allowed size is 100 × 100 pixels."
        )


    # --------------------------------------------------------
    # Aspect ratio barrier
    # --------------------------------------------------------

    aspect_ratio = (
        max(width, height)
        /
        min(width, height)
    )


    if aspect_ratio > 5:

        raise ValueError(
            "Image aspect ratio is not suitable "
            "for this model."
        )


    return image


# ============================================================
# 51. PREDICTION FUNCTION
# ============================================================

@torch.no_grad()
def predict_from_file(
    file_path
):

    try:

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        image = validate_uploaded_file(
            file_path
        )


        # ----------------------------------------------------
        # Apply SAME evaluation preprocessing
        # ----------------------------------------------------

        tensor = frontend_transform(
            image
        )


        tensor = tensor.unsqueeze(
            0
        ).to(
            DEVICE
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        logits = frontend_model(
            tensor
        )


        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]


        predicted_index = int(
            torch.argmax(
                probabilities
            ).item()
        )


        predicted_class = CLASS_NAMES[
            predicted_index
        ]


        confidence = float(
            probabilities[
                predicted_index
            ].item()
        )


        # ----------------------------------------------------
        # Probability table
        # ----------------------------------------------------

        probability_text = "\n".join(
            [
                f"{CLASS_NAMES[i]}: "
                f"{float(probabilities[i]) * 100:.2f}%"
                for i in range(NUM_CLASSES)
            ]
        )


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        result = (
            f"Prediction: {predicted_class}\n\n"
            f"Confidence: {confidence * 100:.2f}%\n\n"
            f"Class probabilities:\n"
            f"{probability_text}\n\n"
            f"Model: ResNet18\n"
            f"Dataset: IDRiD Disease Grading"
        )


        disclaimer = (
            "Research / decision-support prototype only. "
            "This output is not a medical diagnosis and "
            "should not be used as a substitute for "
            "assessment by a qualified healthcare professional."
        )


        return (
            image,
            result,
            disclaimer
        )


    except Exception as e:

        return (
            None,
            f"Input rejected:\n\n{str(e)}",
            "No prediction was generated."
        )


# ============================================================
# 52. GRADIO FRONTEND
# ============================================================

custom_css = """
.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
}

#title {
    text-align: center;
}

#subtitle {
    text-align: center;
    opacity: 0.70;
}

.result-box {
    min-height: 180px;
}

.disclaimer {
    opacity: 0.65;
    font-size: 0.85rem;
}
"""


with gr.Blocks(
    title="IDRiD Retinopathy Grading",
    css=custom_css
) as demo:

    gr.Markdown(
        """
        # IDRiD Retinopathy Grading
        """,
        elem_id="title"
    )


    gr.Markdown(
        """
        Upload a retinal fundus image for research-model
        prediction using the trained ResNet18 classifier.
        """,
        elem_id="subtitle"
    )


    with gr.Row():

        with gr.Column():

            file_input = gr.File(
                label="Upload retinal image",
                type="filepath",
                file_types=[
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".tif",
                    ".tiff"
                ]
            )


            predict_button = gr.Button(
                "Analyze Image",
                variant="primary"
            )


            gr.Markdown(
                f"""
                **Input limits**

                - Maximum file size: {MAX_UPLOAD_MB} MB
                - Minimum resolution: 100 × 100 pixels
                - Supported: JPG, JPEG, PNG, BMP, TIFF
                - Excessively unusual aspect ratios are rejected
                """,
                elem_classes=["disclaimer"]
            )


        with gr.Column():

            output_image = gr.Image(
                label="Validated Input",
                type="pil"
            )


            output_prediction = gr.Textbox(
                label="Prediction",
                lines=12,
                elem_classes=["result-box"]
            )


            output_disclaimer = gr.Markdown(
                elem_classes=["disclaimer"]
            )


    predict_button.click(
        fn=predict_from_file,
        inputs=file_input,
        outputs=[
            output_image,
            output_prediction,
            output_disclaimer
        ]
    )


# ============================================================
# 53. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best Validation Macro-F1: "
    f"{best_macro_f1:.4f}"
)

print(
    f"Official Test Accuracy: "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Official Test Balanced Accuracy: "
    f"{test_metrics['balanced_accuracy']:.4f}"
)

print(
    f"Official Test Macro-F1: "
    f"{test_metrics['macro_f1']:.4f}"
)

print("\nGenerated files:")

print(
    "Model:",
    MODEL_PATH
)

print(
    "History:",
    HISTORY_PATH
)

print(
    "Test predictions:",
    TEST_RESULTS_PATH
)

print(
    "Classes:",
    CLASS_JSON_PATH
)

print(
    "Confusion matrix:",
    CONFUSION_PATH
)

print(
    "F1 curve:",
    F1_CURVE_PATH
)

print(
    "Loss curve:",
    LOSS_CURVE_PATH
)

print("=" * 70)


# ============================================================
# 54. LAUNCH FRONTEND
# ============================================================

print("\nLaunching Gradio frontend...")

demo.launch(
    share=True,
    debug=False
)