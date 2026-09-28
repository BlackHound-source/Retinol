"""
Main entrypoint: Loads datasets, builds caching, executes multi-seed training, and launches UI.
"""

import gc
import torch

from config import SEEDS, BASE_OUTPUT, DEVICE, USE_TTA
from dataset import load_datasets, build_individual_cache, create_loaders
from ood import compute_ood_baseline
from engine import train_one_seed, predict_tta, evaluate, calculate_metrics
from app import build_app

def main():
    print(f"Engine Device: {DEVICE}")
    
    # 1. Dataset Resolution & Ingestion
    print("\n[1/5] Loading datasets...")
    train_part, val_part, test_part = load_datasets()

    # 2. Caching Tensors
    print("\n[2/5] Caching preprocessed tensors...")
    train_meta = build_individual_cache(train_part, "train")
    val_meta = build_individual_cache(val_part, "val")
    test_meta = build_individual_cache(test_part, "test")

    # 3. OOD Baseline Distribution Modeling
    print("\n[3/5] Computing OOD reference baseline...")
    ood_mean, ood_std = compute_ood_baseline(train_meta)

    # 4. Multi-seed Model Training
    print("\n[4/5] Initiating multi-seed training engine...")
    train_loader, val_loader, test_loader, class_weights = create_loaders(train_meta, val_meta, test_meta)

    results = []
    for s in SEEDS:
        res = train_one_seed(s, train_loader, val_loader, class_weights)
        results.append(res)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best_result = sorted(results, key=lambda r: r["val_metrics"]["macro_f1"], reverse=True)[0]
    best_model = best_result["model"]
    print(f"\nSelected Best Model: Seed {best_result['seed']} (Val Macro-F1: {best_result['val_metrics']['macro_f1']:.4f})")

    # Evaluate on Official Test Benchmark
    test_true, test_pred, _ = predict_tta(best_model, test_loader) if USE_TTA else evaluate(best_model, test_loader)[1:]
    test_metrics = calculate_metrics(test_true, test_pred)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f} | Balanced Acc: {test_metrics['balanced_accuracy']:.4f} | Macro F1: {test_metrics['macro_f1']:.4f}")

    # Serialize Artifact
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "seed": best_result["seed"],
        "ood_mean": ood_mean,
        "ood_std": ood_std
    }, BASE_OUTPUT / "Retinol_BEST.pt")

    # 5. Launch UI
    print("\n[5/5] Launching Gradio diagnostic interface...")
    demo, port = build_app(best_model, ood_mean, ood_std)
    demo.launch(share=True, server_name="0.0.0.0", server_port=port, inbrowser=False)

if __name__ == "__main__":
    main()
