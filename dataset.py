"""
Dataset indexing, lazy on-disk tensor caching, PyTorch Dataset, and DataLoader constructors.
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from config import CACHE_DIR, VAL_SIZE, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS
from preprocessing import preprocess_image, normalize_input

try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False

def build_image_index(folder):
    """Creates a dictionary map of lowercase image stems to absolute paths."""
    index = {}
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            index[path.stem.lower()] = str(path)
    return index

def resolve_image(name, index):
    return index.get(Path(str(name)).stem.lower(), None)

def find_idrid_root():
    candidates = [
        Path("/kaggle/input/datasets/aaryapatel98/indian-diabetic-retinopathy-image-dataset/B.%20Disease%20Grading/B. Disease Grading"),
        Path("/kaggle/input/indian-diabetic-retinopathy-image-dataset/B. Disease Grading"),
        Path("/kaggle/input/B. Disease Grading")
    ]
    for path in candidates:
        if path.exists():
            return path
    for path in Path("/kaggle/input").rglob("IDRiD_Disease Grading_Training Labels.csv"):
        return path.parent
    raise FileNotFoundError("IDRiD Disease Grading directory structure not detected.")

def load_datasets():
    """Locates and parses both IDRiD and APTOS datasets, splitting them into train, val, and test."""
    idrid_root = find_idrid_root()
    idrid_train_img = idrid_root / "1. Original Images" / "a. Training Set"
    idrid_test_img = idrid_root / "1. Original Images" / "b. Testing Set"
    idrid_train_csv = idrid_root / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"
    idrid_test_csv = idrid_root / "2. Groundtruths" / "b. IDRiD_Disease Grading_Testing Labels.csv"

    idrid_train_df = pd.read_csv(idrid_train_csv)
    idrid_test_df = pd.read_csv(idrid_test_csv)
    idrid_train_df.columns = [str(c).strip() for c in idrid_train_df.columns]
    idrid_test_df.columns = [str(c).strip() for c in idrid_test_df.columns]

    idrid_train_df = idrid_train_df[["Image name", "Retinopathy grade"]].rename(columns={"Image name": "image_id", "Retinopathy grade": "grade"})
    idrid_test_df = idrid_test_df[["Image name", "Retinopathy grade"]].rename(columns={"Image name": "image_id", "Retinopathy grade": "grade"})

    train_idx = build_image_index(idrid_train_img)
    test_idx = build_image_index(idrid_test_img)

    idrid_train_df["path"] = idrid_train_df["image_id"].apply(lambda x: resolve_image(x, train_idx))
    idrid_test_df["path"] = idrid_test_df["image_id"].apply(lambda x: resolve_image(x, test_idx))
    idrid_train_df = idrid_train_df.dropna(subset=["path", "grade"]).reset_index(drop=True)
    idrid_test_df = idrid_test_df.dropna(subset=["path", "grade"]).reset_index(drop=True)
    idrid_train_df["grade"] = idrid_train_df["grade"].astype(int)
    idrid_test_df["grade"] = idrid_test_df["grade"].astype(int)
    idrid_train_df["source"] = "IDRiD"
    idrid_test_df["source"] = "IDRiD_Official_Test"

    # APTOS Discovery
    aptos_path = None
    if KAGGLEHUB_AVAILABLE:
        try:
            aptos_path = Path(kagglehub.dataset_download("mariaherrerot/aptos2019"))
        except Exception:
            pass

    if aptos_path is None or not aptos_path.exists():
        for c in [Path("/kaggle/input/datasets/mariaherrerot/aptos2019"), Path("/kaggle/input/aptos2019"), Path("/kaggle/input/aptos2019-blindness-detection")]:
            if c.exists():
                aptos_path = c
                break

    all_csvs = list(aptos_path.rglob("*.csv")) if aptos_path else []
    target_csv = all_csvs[0] if all_csvs else None
    aptos_df = pd.read_csv(target_csv)
    aptos_df.columns = [str(c).strip().lower() for c in aptos_df.columns]
    aptos_df = aptos_df.rename(columns={aptos_df.columns[0]: "image_id", aptos_df.columns[1]: "grade"})
    aptos_df["grade"] = pd.to_numeric(aptos_df["grade"], errors="coerce")
    aptos_df = aptos_df.dropna(subset=["grade"])
    aptos_df["grade"] = aptos_df["grade"].astype(int)
    apt_idx = build_image_index(aptos_path)
    aptos_df["path"] = aptos_df["image_id"].apply(lambda x: resolve_image(x, apt_idx))
    aptos_df = aptos_df.dropna(subset=["path"]).reset_index(drop=True)
    aptos_df["source"] = "APTOS"

    # Split combined cohort
    combined = pd.concat([idrid_train_df, aptos_df], ignore_index=True)
    train_part, val_part = train_test_split(combined, test_size=VAL_SIZE, stratify=combined["grade"], random_state=42)
    return train_part.reset_index(drop=True), val_part.reset_index(drop=True), idrid_test_df

def build_individual_cache(df, split_name):
    """Processes images and serializes tensors to disk to avoid out-of-memory errors."""
    target_dir = CACHE_DIR / split_name
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for _, row in df.iterrows():
        stem = Path(row["path"]).stem
        tensor_path = target_dir / f"{stem}.pt"
        if not tensor_path.exists():
            try:
                combined, _, _, _ = preprocess_image(row["path"])
                torch.save(torch.from_numpy(combined), tensor_path)
            except Exception:
                continue
        metadata.append({
            "tensor_path": str(tensor_path),
            "label": int(row["grade"]),
            "image_id": str(row["image_id"]),
            "source": str(row["source"])
        })
    return metadata

def augment_tensor(x):
    """Performs spatial geometric and color jitter augmentations on 4-channel tensor."""
    if random.random() < 0.6:
        x = torch.flip(x, dims=[2])
    if random.random() < 0.3:
        x = torch.flip(x, dims=[1])
    if random.random() < 0.4:
        x = torch.rot90(x, k=random.choice([1, 2, 3]), dims=[1, 2])
    if random.random() < 0.5:
        rgb = (x[:3] * random.uniform(0.85, 1.15) + random.uniform(-0.05, 0.05)).clamp(0, 1)
        x = torch.cat([rgb, x[3:4]], dim=0)
    if random.random() < 0.3:
        edge = (x[3:4] * random.uniform(0.8, 1.2)).clamp(0, 1)
        x = torch.cat([x[:3], edge], dim=0)
    return x

class LazyCachedRetinaDataset(Dataset):
    """Loads preprocessed tensors dynamically from disk during training and evaluation."""
    def __init__(self, metadata, training=False):
        self.metadata = metadata
        self.training = training

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        item = self.metadata[index]
        x = torch.load(item["tensor_path"], weights_only=True)
        if self.training:
            x = augment_tensor(x)
        return normalize_input(x), torch.tensor(item["label"], dtype=torch.long)

def create_loaders(train_meta, val_meta, test_meta):
    """Initializes PyTorch DataLoaders with a class-balanced WeightedRandomSampler."""
    train_labels = np.array([m["label"] for m in train_meta])
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1))
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    sample_weights = np.array([class_weights[m["label"]] for m in train_meta])
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(train_meta), replacement=True)

    train_loader = DataLoader(LazyCachedRetinaDataset(train_meta, training=True), batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(LazyCachedRetinaDataset(val_meta, training=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(LazyCachedRetinaDataset(test_meta, training=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader, class_weights
