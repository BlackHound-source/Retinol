"""
Clinical inference engine, out-of-distribution verification, and triage safety logic.
"""

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch

from config import DEVICE, IMAGE_SIZE, CLASS_NAMES
from preprocessing import detect_retinal_roi, square_crop, apply_clahe, normalize_illumination, extract_edges, normalize_input
from ood import extract_image_shift_features

def preprocess_uploaded_image(image):
    """Adapts arbitrary web input images to 4-channel normalized input tensors."""
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    else:
        image = np.asarray(image)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    original_rgb = np.ascontiguousarray(image)
    roi_bgr = detect_retinal_roi(cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR))
    roi_bgr = square_crop(roi_bgr)
    roi_bgr = cv2.resize(roi_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    roi_bgr = apply_clahe(roi_bgr)
    roi_bgr = normalize_illumination(roi_bgr)

    processed_rgb = np.ascontiguousarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)).astype(np.uint8)
    edges = np.ascontiguousarray(extract_edges(roi_bgr)).astype(np.uint8)

    rgb_float = processed_rgb.astype(np.float32) / 255.0
    edge_float = edges.astype(np.float32) / 255.0
    combined = np.transpose(np.concatenate([rgb_float, edge_float[..., None]], axis=2), (2, 0, 1))

    tensor = normalize_input(torch.from_numpy(combined).float()).unsqueeze(0)
    return tensor, original_rgb, processed_rgb, edges

def evaluate_clinical_safety(processed_rgb, edges, tta_probs_stack, final_probs, ood_mean, ood_std):
    """Computes distribution shift flags, perforation alerts, and model prediction instability."""
    current_fp = extract_image_shift_features(
        np.concatenate([
            np.transpose(processed_rgb.astype(np.float32) / 255.0, (2, 0, 1)),
            (edges.astype(np.float32) / 255.0)[None, ...]
        ], axis=0)
    )
    z_scores = np.abs((current_fp - ood_mean) / ood_std)
    ood_deviations = int(np.sum(z_scores > 3.2))
    ood_flag = ood_deviations >= 2
    perforation_flag = bool(z_scores[7] > 3.8 or z_scores[8] > 3.8)

    tta_max_disagreement = float(np.max(np.std(tta_probs_stack, axis=0)))
    tta_instability_flag = tta_max_disagreement >= 0.12

    high_grade_sum = float(final_probs[3] + final_probs[4])
    referable_sum = float(final_probs[2] + final_probs[3] + final_probs[4])

    return {
        "ood_flag": ood_flag,
        "ood_deviations": ood_deviations,
        "perforation_flag": perforation_flag,
        "tta_instability": tta_instability_flag,
        "tta_max_disagreement": tta_max_disagreement,
        "high_grade_sum": high_grade_sum,
        "referable_sum": referable_sum
    }

@torch.no_grad()
def frontend_predict(image, model, ood_mean, ood_std):
    """End-to-end inference orchestrator for real-time frontend screening."""
    if image is None:
        return (None, None, None, "Awaiting Input", 0.0, "PENDING", "Upload a fundus image to initiate screening.", pd.DataFrame(), None, None)

    try:
        tensor, original_rgb, processed_rgb, edges = preprocess_uploaded_image(image)
    except Exception as e:
        return (None, None, None, "Preprocessing Error", 0.0, "ERROR", f"Error: {str(e)[:120]}", pd.DataFrame(), None, None)

    model.eval()
    t = tensor.to(DEVICE)
    p1 = torch.softmax(model(t), dim=1)
    p2 = torch.softmax(model(torch.flip(t, dims=[3])), dim=1)
    p3 = torch.softmax(model(torch.flip(t, dims=[2])), dim=1)
    p4 = torch.softmax(model(torch.rot90(t, k=1, dims=[2, 3])), dim=1)

    probs_stack_np = torch.cat([p1, p2, p3, p4], dim=0).cpu().numpy()
    probs = (0.40 * p1 + 0.25 * p2 + 0.20 * p3 + 0.15 * p4)[0].cpu().numpy()

    model_pred = int(np.argmax(probs))
    confidence = float(probs[model_pred]) * 100.0
    safety = evaluate_clinical_safety(processed_rgb, edges, probs_stack_np, probs, ood_mean, ood_std)

    sorted_idx = np.argsort(probs)[::-1]
    top_idx, second_idx = int(sorted_idx[0]), int(sorted_idx[1])
    prob_margin = float(probs[top_idx] - probs[second_idx]) * 100.0

    triage_status = "NORMAL"
    warning_notes = []
    final_grade = model_pred

    # False-Negative Guardrail: elevate grade if borderline severe
    if (second_idx >= 3 and prob_margin < 12.0) or (safety["high_grade_sum"] >= 0.45 and model_pred < 3):
        final_grade = max(top_idx, second_idx)
        confidence = float(probs[final_grade]) * 100.0
        triage_status = "CRITICAL ELEVATION"
        warning_notes.append(f"High-severity risk identified: Grade {final_grade} signals elevated ({probs[final_grade]*100:.1f}%).")

    if safety["perforation_flag"]:
        triage_status = "PERFORATION / ANOMALY DETECTED"
        warning_notes.append("Lesion/perforation density deviates from baseline. Specialist review advised.")

    if safety["ood_flag"]:
        if triage_status == "NORMAL":
            triage_status = "DATASET SHIFT REVIEW"
        warning_notes.append(f"Distribution Warning: Shift detected ({safety['ood_deviations']} deviations).")

    if safety["tta_instability"]:
        warning_notes.append(f"Instability Warning: {safety['tta_max_disagreement']*100:.1f}% rotation shift.")

    if not warning_notes:
        warning_notes.append("Screening complete. Distribution aligns with expected retinal morphology.")

    prob_table = pd.DataFrame({
        "Grade": CLASS_NAMES,
        "Severity": ["No DR", "Mild", "Moderate", "Severe Non-Proliferative", "Proliferative DR"],
        "Probability (%)": np.round(probs * 100.0, 2)
    })

    screening_state = {
        "grade": int(final_grade),
        "confidence": float(confidence),
        "triage_status": triage_status,
        "safety_flags": safety
    }

    return (
        original_rgb, processed_rgb, edges,
        f"Grade {final_grade}", round(confidence, 2),
        triage_status, " • ".join(warning_notes),
        prob_table, original_rgb, screening_state
    )
