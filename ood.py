"""
Out-of-Distribution (OOD) shift calculations and micro-perforation lesion detectors.
"""

import cv2
import numpy as np
import torch

def extract_image_shift_features(sample_tensor):
    """Computes distribution statistics (RGB channel moments, edge density, blob morphology)."""
    if isinstance(sample_tensor, torch.Tensor):
        x = sample_tensor.detach().cpu().numpy().astype(np.float32)
    else:
        x = np.asarray(sample_tensor, dtype=np.float32)

    if x.shape[0] == 4:
        rgb, edge = x[:3], x[3]
    elif x.shape[-1] == 4:
        x = np.transpose(x, (2, 0, 1))
        rgb, edge = x[:3], x[3]
    else:
        raise ValueError(f"Expected 4-channel tensor, got {x.shape}")

    gray = np.mean(rgb, axis=0)
    bright = (gray > 0.76).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    
    blob_count, blob_area = 0, 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if 15 <= area <= 2000 and w > 0 and h > 0:
            blob_count += 1
            blob_area += area

    return np.array([
        float(rgb.mean()), float(rgb.std()), float(rgb[0].mean()),
        float(rgb[1].mean()), float(rgb[2].mean()), float(edge.mean()),
        float((edge > 0.25).mean()), float(blob_count),
        float(blob_area) / float(rgb.shape[1] * rgb.shape[2])
    ], dtype=np.float32)

def compute_ood_baseline(train_meta):
    """Calculates feature-space mean and standard deviation across training cohorts."""
    subset = np.random.choice(len(train_meta), size=min(400, len(train_meta)), replace=False)
    samples = [torch.load(train_meta[i]["tensor_path"], map_location="cpu", weights_only=True) for i in subset]
    matrix = np.stack([extract_image_shift_features(s) for s in samples], axis=0)
    return matrix.mean(axis=0), np.maximum(matrix.std(axis=0), 1e-5)
