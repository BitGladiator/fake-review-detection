"""
test_inference.py — Fake Review Detection: CLI Inference Test
Run from /model directory:
    python test_inference.py [--artifacts-dir artifacts] [--rating 3.0]
"""

import sys
import os
import argparse

import numpy as np
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import build_features, load_preprocessor, predict_single_review

GENUINE_REVIEWS = [
    "Great food, friendly staff, and the ambiance was perfect for a date night. Will definitely come back!",
    "The pasta was cooked to perfection. Service was a bit slow but worth the wait.",
    "Decent place, nothing extraordinary. The burger was good but a little overpriced for the portion size.",
]

FAKE_REVIEWS = [
    "BEST RESTAURANT EVER!!!! AMAZING FOOD!!!! MUST VISIT!!!! 5 STARS!!!! GO NOW!!!!",
    "Perfect perfect perfect! Everything was absolutely perfect! Best experience of my life! Perfect!",
    "I was paid to write this review. The food was okay I guess. Please give 5 stars they asked me to.",
]


def truncate(text, max_len=80):
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


def label_name(val):
    return "Spam" if val == 1 else "Genuine"


def run_inference(review_text, true_label_int, rating, preprocessor, word_vec, char_vec, scaler, models, thresholds):
    """Run all loaded models on a single review using predict_single_review helper."""
    results = {}
    for model_name, model in models.items():
        thresh = thresholds.get(model_name, None)
        try:
            res = predict_single_review(
                text=review_text,
                rating=rating,
                user_id=None,
                product_id=None,
                model_name=model_name,
                preprocessor=preprocessor,
                word_vectorizer=word_vec,
                char_vectorizer=char_vec,
                scaler=scaler,
                model=model,
                threshold=thresh,
            )
            pred = res["prediction"]
            prob = res["probability"]
            th = res["threshold"]
            conf = res["confidence"]
            confidence_str = f"{conf * 100:.1f}% (P(Spam)={prob:.2f}, thresh={th:.2f})"
            results[model_name] = (pred, confidence_str)
        except Exception as e:
            print(f"    ⚠  Prediction failed for {model_name}: {e}")
            results[model_name] = (None, "ERR")

    return results


def print_review_block(review_text, true_label_int, model_results):
    true_str = label_name(true_label_int)
    print(f"\n  Review : \"{truncate(review_text)}\"")
    print(f"  True   : {true_str}")
    if not model_results:
        print("  (no model results)")
        return
    for model_name, (pred, confidence_str) in model_results.items():
        if pred is None:
            print(f"    [{model_name}]: ERROR")
        else:
            pred_str = label_name(pred)
            match = "✓" if pred == true_label_int else "✗"
            print(f"    [{model_name:<9}]: Pred={pred_str:<7} {match}  Confidence: {confidence_str}")


def main():
    parser = argparse.ArgumentParser(description="Run fake-review inference on sample texts.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Path to artifacts directory")
    parser.add_argument("--rating", type=float, default=4.0, help="Star rating for reviews (default: 4.0)")
    parser.add_argument("--threshold", type=float, default=0.50, help="Decision threshold for calibrated probability (default: 0.50)")
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir
    if not os.path.isdir(artifacts_dir):
        print(f"⚠ Artifacts directory not found: {artifacts_dir}. Run train.py first.")
        sys.exit(1)

    print(f"\n[1/3] Loading preprocessor & vectorizers from '{artifacts_dir}' ...")
    try:
        preprocessor, word_vec, char_vec, scaler = load_preprocessor(artifacts_dir)
        print("  ✓ Preprocessor, dual vectorizers, and scaler loaded.")
    except Exception as e:
        print(f"  ✗ Failed to load preprocessor: {e}")
        sys.exit(1)


    thresholds = {}
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    if os.path.exists(metrics_path):
        import json
        try:
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
            thresholds = metrics_data.get("best_thresholds", {})
        except Exception:
            pass

    print(f"\n[2/3] Loading models from '{artifacts_dir}' ...")
    model_files = {
        "XGBoost":   "XGBoost_best.joblib",
        "LinearSVC": "LinearSVC_best.joblib",
        "SGD":       "SGD_best.joblib",
    }
    models = {}
    for model_name, filename in model_files.items():
        path = os.path.join(artifacts_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            models[model_name] = joblib.load(path)
            thresh_val = thresholds.get(model_name, 0.50)
            print(f"  ✓ {model_name} loaded from {filename} (thresh={thresh_val:.2f})")
        except Exception as e:
            print(f"  ✗ Failed to load {model_name}: {e}")

    if not models:
        print("\n✗ No models found in artifacts. Run train.py first.")
        sys.exit(1)

    print("\n[3/3] Running Inference ...")
    correct_counts = {m: 0 for m in models}
    total_reviews = len(GENUINE_REVIEWS) + len(FAKE_REVIEWS)

    print("\n--- Genuine Reviews (Rating=4.0) ---")
    for rev in GENUINE_REVIEWS:
        res = run_inference(rev, 0, 4.0, preprocessor, word_vec, char_vec, scaler, models, thresholds)
        print_review_block(rev, 0, res)
        for m, (pred, _) in res.items():
            if pred == 0:
                correct_counts[m] += 1

    print("\n--- Spam / Fake Reviews (Rating=5.0) ---")
    for rev in FAKE_REVIEWS:
        res = run_inference(rev, 1, 5.0, preprocessor, word_vec, char_vec, scaler, models, thresholds)
        print_review_block(rev, 1, res)
        for m, (pred, _) in res.items():
            if pred == 1:
                correct_counts[m] += 1

    print("\n" + "=" * 50)
    print("Inference Test Summary:")
    for m in models:
        corr = correct_counts[m]
        acc = (corr / total_reviews) * 100
        print(f"  {m:<10}: {corr}/{total_reviews} correct ({acc:.1f}%)")
    print("=" * 50)


if __name__ == "__main__":
    main()
