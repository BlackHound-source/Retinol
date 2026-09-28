"""
Computer vision pipelines for retinal segmentation, illumination normalization, and edge detection.
"""

import cv2
import numpy as np
import torch
from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

def detect_retinal_roi(img):
    """Detects and crops the retinal circular fundus boundary from dark background."""
    if img is None:
        raise ValueError("Null image tensor encountered.")
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < (0.10 * h * w):
        return img
    x, y, cw, ch = cv2.boundingRect(contour)
    margin = int(0.06 * max(cw, ch))
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(w, x + cw + margin), min(h, y + ch + margin)
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else img

def square_crop(img):
    """Crops an image into a symmetric square along its shortest dimension."""
    h, w = img.shape[:2]
    size = min(h, w)
    cx, cy = w // 2, h // 2
    half = size // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    crop = img[y1:y1 + size, x1:x1 + size]
    side = min(crop.shape[0], crop.shape[1])
    return crop[:side, :side]

def apply_clahe(img):
    """Applies Contrast Limited Adaptive Histogram Equalization to the L-channel."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

def normalize_illumination(img):
    """Estimates local background intensity via Gaussian blur and normalizes luminance."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    background = cv2.GaussianBlur(l, (0, 0), sigmaX=40)
    corrected = np.clip(l.astype(np.float32) - background.astype(np.float32) + 128.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([corrected, a, b]), cv2.COLOR_LAB2BGR)

def extract_edges(img):
    """Extracts structural vasculature via Canny edge detection and morphological filtering."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.0)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    return cv2.erode(edges, kernel, iterations=1)

def preprocess_image(path):
    """Full preprocessing pipeline yielding a 4-channel tensor (RGB + Canny Edge)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to read image at: {path}")
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
    return combined.astype(np.float32), original_rgb, processed_rgb, edges

def normalize_input(x):
    """Applies standard ImageNet normalization to RGB channels and center-scales edge map."""
    rgb = (x[:3] - IMAGENET_MEAN) / IMAGENET_STD
    edge = (x[3:4] - 0.5) / 0.5
    return torch.cat([rgb, edge], dim=0).float()
