"""
Model training loops, metric computations, evaluation loops, and Test-Time Augmentation (TTA).
"""

import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch
import torch.optim as optim

from config import (
    DEVICE, MAX_EPOCHS, PATIENCE, LEARNING_RATE_BACKBONE,
    LEARNING_RATE_HEAD, WEIGHT_DECAY, LABEL_SMOOTHING, CHECKPOINT_DIR
)
from model import build_model, FocalLabelSmoothingCE

def seed_everything(seed):
    """Seeds random number generators for reproducible operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }

@torch.no_grad()
def evaluate(model, loader, criterion=None):
    """Evaluates performance metrics on a chosen DataLoader."""
    model.eval()
    losses, all_targets, all_preds, all_probs = [], [], [], []
    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        logits = model(images)
        if criterion is not None:
            losses.append(criterion(logits, targets).item())
        probs = torch.softmax(logits, dim=1)
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(probs.argmax(dim=1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true, y_pred = np.array(all_targets), np.array(all_preds)
    probs = np.concatenate(all_probs, axis=0)
    metrics = calculate_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics, y_true, y_pred, probs

@torch.no_grad()
def predict_tta(model, loader):
    """Applies multi-angle flip and rotation Test-Time Augmentation."""
    model.eval()
    all_probs, all_targets = [], []
    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        p1 = torch.softmax(model(images), dim=1)
        p2 = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
        p3 = torch.softmax(model(torch.flip(images, dims=[2])), dim=1)
        p4 = torch.softmax(model(torch.rot90(images, k=1, dims=[2, 3])), dim=1)
        probs = 0.40 * p1 + 0.25 * p2 + 0.20 * p3 + 0.15 * p4
        all_probs.append(probs.cpu().numpy())
        all_targets.extend(targets.numpy())
    probs = np.concatenate(all_probs, axis=0)
    return np.array(all_targets), probs.argmax(axis=1), probs

def train_one_seed(seed, train_loader, val_loader, class_weights):
    """Trains an individual model instance with Cosine Annealing and early stopping."""
    seed_everything(seed)
    model = build_model().to(DEVICE)

    head_params, early_params, late_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier"):
            head_params.append(param)
        elif name.startswith("features.0") or name.startswith("features.1"):
            early_params.append(param)
        else:
            late_params.append(param)

    optimizer = optim.AdamW([
        {"params": early_params, "lr": 1e-6},
        {"params": late_params, "lr": LEARNING_RATE_BACKBONE},
        {"params": head_params, "lr": LEARNING_RATE_HEAD}
    ], weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-7)
    criterion = FocalLabelSmoothingCE(class_weights, smoothing=LABEL_SMOOTHING).to(DEVICE)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_macro_f1, best_epoch, patience_counter = -1.0, 0, 0
    best_path = CHECKPOINT_DIR / f"retinol_seed_{seed}_best.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for images, targets in train_loader:
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

        scheduler.step()
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion)

        if val_metrics["macro_f1"] > best_macro_f1 + 1e-4:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "seed": seed,
                "epoch": epoch,
                "val_metrics": val_metrics
            }, best_path)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics, _, _, _ = evaluate(model, val_loader, criterion)
    return {"seed": seed, "model": model, "val_metrics": val_metrics, "best_epoch": best_epoch, "checkpoint": str(best_path)}
