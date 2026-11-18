"""
test_inference.py — Fake Review Detection: CLI Inference Test
Run from /model directory: python test_inference.py [--artifacts-dir artifacts] [--rating 3.0]
"""

import sys
import os
import time
import gc
import argparse

import numpy as np
import joblib
import scipy.sparse
import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import TextPreprocessor, build_features, load_preprocessor


GENUINE_REVIEWS = [
    "Great food, friendly staff, and the ambiance was perfect for a date night. Will definitely come back!",
    "The pasta was cooked to perfection. Service was a bit slow but worth the wait.",
    "Decent place, nothing extraordinary. The burger was good but a little overpriced for the portion size."
]

FAKE_REVIEWS = [
    "BEST RESTAURANT EVER!!!! AMAZING FOOD!!!! MUST VISIT!!!! 5 STARS!!!! GO NOW!!!!",
    "Perfect perfect perfect! Everything was absolutely perfect! Best experience of my life! Perfect!",
    "I was paid to write this review. The food was okay I guess. Please give 5 stars they asked me to."
]



def truncate(text, max_len=80):
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


def label_name(val):
    return "Spam" if val == 1 else "Genuine"


def run_inference(review_text, true_label_int, args, preprocessor, vectorizer, scaler, models):
    """Run all loaded models on a single review; return dict of {model_name: pred_int}."""
    series = pd.Series([review_text])
    ratings = pd.Series([args.rating])

    try:
        cleaned, aux = preprocessor.transform(series, ratings)
    except Exception as e:
        print(f"    ⚠  Preprocessing failed: {e}")
        return {}

    try:
        X = build_features(cleaned, aux, vectorizer, scaler, is_train=False)
    except Exception as e:
        print(f"    ⚠  Feature building failed: {e}")
        return {}

    results = {}
    for model_name, model in models.items():
        try:
            pred = int(model.predict(X)[0])
            confidence_str = "N/A"
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X)[0]
                    confidence = float(proba[pred]) * 100
                    confidence_str = f"{confidence:.1f}%"
                except Exception:
                    pass
            elif hasattr(model, 'decision_function'):
               
                try:
                    score = model.decision_function(X)[0]
                    confidence = 100.0 / (1.0 + np.exp(-abs(score)))
                    confidence_str = f"{confidence:.1f}%"
                except Exception:
                    pass
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
            print(f"  [{model_name}]: ERROR")
        else:
            pred_str = label_name(pred)
            correct = "✓" if pred == true_label_int else "✗"
            print(f"  [{model_name}]: Pred={pred_str}, Confidence={confidence_str}  {correct}")


def main():
    print("=== Fake Review Detection: Inference Test ===")
    t0 = time.time()

   
    parser = argparse.ArgumentParser(description="Run fake-review inference on sample texts.")
    parser.add_argument("--artifacts-dir", default="artifacts",
                        help="Path to artifacts directory (default: artifacts)")
    parser.add_argument("--rating", type=float, default=3.0,
                        help="Star rating to use for all sample reviews (default: 3.0)")
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir
    if not os.path.isdir(artifacts_dir):
        print(f"⚠  Artifacts directory not found: {artifacts_dir}")
        print("   Run train.py first to generate artifacts.")
        sys.exit(1)

    
    print(f"\n[1/3] Loading preprocessor from '{artifacts_dir}' ...")
    try:
        preprocessor, vectorizer, scaler = load_preprocessor(artifacts_dir)
        print("  ✓ preprocessor, vectorizer, scaler loaded.")
    except Exception as e:
        print(f"  ✗ Failed to load preprocessor: {e}")
        sys.exit(1)

   
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
            print(f"  ⚠  {filename} not found — skipping {model_name}.")
            continue
        try:
            m = joblib.load(path)
            models[model_name] = m
            print(f"  ✓ {model_name} loaded from {filename}")
        except Exception as e:
            print(f"  ✗ Failed to load {model_name} ({filename}): {e}")

    if not models:
        print("\n✗ No models loaded. Cannot run inference.")
        sys.exit(1)

    
    print(f"\n[3/3] Running inference (rating={args.rating}) ...")

   
    correct_counts = {m: 0 for m in models}
    total_counts = {m: 0 for m in models}

    review_sets = [
        ("GENUINE REVIEWS", GENUINE_REVIEWS, 0),
        ("FAKE REVIEWS",    FAKE_REVIEWS,    1),
    ]

    for set_name, reviews, true_label in review_sets:
        print(f"\n{'─'*60}")
        print(f"  {set_name}")
        print(f"{'─'*60}")
        for review in reviews:
            results = run_inference(review, true_label, args, preprocessor, vectorizer, scaler, models)
            print_review_block(review, true_label, results)
            for model_name, (pred, _) in results.items():
                if pred is not None:
                    total_counts[model_name] += 1
                    if pred == true_label:
                        correct_counts[model_name] += 1

   
    total_reviews = len(GENUINE_REVIEWS) + len(FAKE_REVIEWS)
    print(f"\n{'═'*60}")
    print("  SUMMARY")
    print(f"{'═'*60}")
    print(f"  Total sample reviews: {total_reviews}")
    for model_name in models:
        tot = total_counts[model_name]
        cor = correct_counts[model_name]
        pct = (cor / tot * 100) if tot > 0 else 0.0
        print(f"  [{model_name}]: {cor}/{tot} correct ({pct:.1f}%)")

    gc.collect()
    elapsed = time.time() - t0
    print(f"\nInference test complete!  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
