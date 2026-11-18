"""
train.py — Fake Review Detection: Model Training Pipeline
Trains 9 combinations (3 samplers × 3 classifiers) on TF-IDF + auxiliary features.

"""

import os
import gc
import json
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import psutil
import scipy.sparse

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from xgboost import XGBClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import joblib

from preprocessing import TextPreprocessor, build_features, save_preprocessor

warnings.filterwarnings("ignore")


DATA_PATH = "data/processed/yelp_nyc_clean.parquet"
ARTIFACTS_DIR = "artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TFIDF_MAX_FEATURES = 15000
N_JOBS = 4


def memory_usage() -> None:
    """Print current process RSS memory in MB."""
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    print(f"  [Memory] RSS: {rss_mb:.1f} MB")


def elapsed(start: float) -> str:
    """Return elapsed time as 'Xm Ys' string."""
    secs = int(time.time() - start)
    return f"{secs // 60}m {secs % 60}s"



def main() -> None:
    total_start = time.time()

    
    print("\n" + "=" * 70)
    print("STEP 1 — Load Data")
    print("=" * 70)

    df = pd.read_parquet(DATA_PATH)
    print(f"  Dataset shape: {df.shape}")
    print("  Class distribution:")
    class_counts = df['label'].value_counts().sort_index()
    for lbl, cnt in class_counts.items():
        name = "Genuine" if lbl == 0 else "Spam"
        print(f"    label={lbl} ({name}): {cnt:,} ({cnt / len(df) * 100:.1f}%)")
    memory_usage()
    gc.collect()


    print("\n" + "=" * 70)
    print("STEP 2 — Train/Test Split")
    print("=" * 70)

    X_text = df['text']
    y = df['label']
    ratings = df['rating'] if 'rating' in df.columns else pd.Series(
        np.full(len(df), 3.0), index=df.index, name='rating'
    )

    (
        train_texts, test_texts,
        train_ratings, test_ratings,
        y_train, y_test,
    ) = train_test_split(
        X_text, ratings, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"  Train size: {len(train_texts):,}  |  Test size: {len(test_texts):,}")
    print(f"  y_train distribution: {dict(pd.Series(y_train).value_counts().sort_index())}")
    print(f"  y_test  distribution: {dict(pd.Series(y_test).value_counts().sort_index())}")

    
    print("\n" + "=" * 70)
    print("STEP 3 — Initialise Preprocessor and TF-IDF Vectorizer")
    print("=" * 70)

    preprocessor = TextPreprocessor(stemming=True, remove_stopwords=True)
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
        dtype=np.float32,
    )
    scaler = StandardScaler(with_mean=False)   

    print("  TextPreprocessor: stemming=True, remove_stopwords=True")
    print(f"  TfidfVectorizer: max_features={TFIDF_MAX_FEATURES}, ngram_range=(1,2), sublinear_tf=True, dtype=float32")
    print("  StandardScaler: with_mean=False (sparse-safe)")

    
    print("\n" + "=" * 70)
    print("STEP 4 — Fit Preprocessor on Train, Transform Both Splits")
    print("=" * 70)

    t4 = time.time()
    cleaned_train, aux_train = preprocessor.fit_transform(train_texts, train_ratings)
    print(f"  Train texts cleaned  ({elapsed(t4)})")

    t4b = time.time()
    cleaned_test, aux_test = preprocessor.transform(test_texts, test_ratings)
    print(f"  Test texts cleaned   ({elapsed(t4b)})")
    gc.collect()

  
    print("\n" + "=" * 70)
    print("STEP 5 — Build Feature Matrices (sparse)")
    print("=" * 70)

    t5 = time.time()
    X_train = build_features(cleaned_train, aux_train, vectorizer, scaler, is_train=True)
    X_test  = build_features(cleaned_test,  aux_test,  vectorizer, scaler, is_train=False)

    print(f"  X_train shape: {X_train.shape},  X_test shape: {X_test.shape}")
    print(f"  Build time: {elapsed(t5)}")
    gc.collect()
    memory_usage()

   
    
    print("\n" + "=" * 70)
    print("STEP 6 — Define Samplers")
    print("=" * 70)

    samplers = [
        ("Original",     None),
        ("OverSampled",  RandomOverSampler(random_state=RANDOM_STATE)),
        ("UnderSampled", RandomUnderSampler(random_state=RANDOM_STATE)),
    ]
    for name, _ in samplers:
        print(f"  · {name}")

   
    print("\n" + "=" * 70)
    print("STEP 7 — Define Classifiers")
    print("=" * 70)

    classifiers = [
        (
            "XGBoost",
            XGBClassifier(
                tree_method="hist",
                max_bin=256,
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
            ),
        ),
        (
            "LinearSVC",
            CalibratedClassifierCV(
                LinearSVC(max_iter=2000, random_state=RANDOM_STATE), cv=3
            ),
        ),
        (
            "SGD",
            SGDClassifier(
                loss="log_loss",
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
                max_iter=1000,
            ),
        ),
    ]
    for name, _ in classifiers:
        print(f"  · {name}")

   
    print("\n" + "=" * 70)
    print("STEP 8 — Training Loop (3 samplers × 3 classifiers = 9 runs)")
    print("=" * 70)

    results: dict = {}

    for sampler_name, sampler in samplers:
        for clf_name, clf in classifiers:
            key = f"{clf_name}_{sampler_name}"
            print(f"\n--- Training {key} ---")
            t0 = time.time()

           
            if sampler is not None:
                X_res, y_res = sampler.fit_resample(X_train, y_train)
                resampled_counts = pd.Series(y_res).value_counts().sort_index()
                print(f"  Resampled class distribution: {dict(resampled_counts)}")
            else:
                X_res, y_res = X_train, y_train

        
            clf.fit(X_res, y_res)
            training_time = time.time() - t0

          
            y_pred = clf.predict(X_test)
            y_prob = (
                clf.predict_proba(X_test)[:, 1]
                if hasattr(clf, 'predict_proba')
                else None
            )

            # ── Metrics ─────────────────────────────────────────────────────
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(
                y_test, y_pred,
                target_names=['Genuine', 'Spam'],
                output_dict=True,
            )
            spam_precision = report['Spam']['precision']
            spam_recall    = report['Spam']['recall']
            spam_f1        = report['Spam']['f1-score']
            roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
            cm = confusion_matrix(y_test, y_pred)

     
            print(f"  {'Metric':<22} {'Value':>10}")
            print(f"  {'-'*34}")
            print(f"  {'Accuracy':<22} {accuracy:>10.4f}")
            print(f"  {'Spam Precision':<22} {spam_precision:>10.4f}")
            print(f"  {'Spam Recall':<22} {spam_recall:>10.4f}")
            print(f"  {'Spam F1':<22} {spam_f1:>10.4f}")
            if roc_auc is not None:
                print(f"  {'ROC AUC':<22} {roc_auc:>10.4f}")
            else:
                print(f"  {'ROC AUC':<22} {'N/A':>10}")
            print(f"  {'Training time':<22} {training_time:>9.1f}s")
            print(f"  Confusion matrix:\n{cm}")

         
            results[key] = {
                "accuracy":        float(accuracy),
                "spam_precision":  float(spam_precision),
                "spam_recall":     float(spam_recall),
                "spam_f1":         float(spam_f1),
                "roc_auc":         float(roc_auc) if roc_auc is not None else None,
                "cm":              cm.tolist(),
                "y_pred":          y_pred.tolist(),
                "y_prob":          y_prob.tolist() if y_prob is not None else None,
                "clf_name":        clf_name,
                "sampler_name":    sampler_name,
                "training_time":   float(training_time),
                "_clf":            clf,
            }

            gc.collect()
            memory_usage()

  
    print("\n" + "=" * 70)
    print("STEP 9 — Select Best Model per Algorithm (by Spam F1)")
    print("=" * 70)

    best_models: dict = {}   
    for alg in ["XGBoost", "LinearSVC", "SGD"]:
        candidates = {
            k: v for k, v in results.items() if v["clf_name"] == alg
        }
        best_key = max(candidates, key=lambda k: candidates[k]["spam_f1"])
        best_f1 = candidates[best_key]["spam_f1"]
        best_models[alg] = best_key
        print(f"  Best {alg}: {best_key} (Spam F1={best_f1:.4f})")

  
    for key in results:
        results[key]["is_best"] = (key in best_models.values())

  
    print("\n" + "=" * 70)
    print("STEP 10 — Save Artifacts")
    print("=" * 70)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

  
    save_preprocessor(preprocessor, vectorizer, scaler, ARTIFACTS_DIR)
    print(f"  Preprocessor/vectorizer/scaler saved to {ARTIFACTS_DIR}/")

 
    for alg, best_key in best_models.items():
        clf_obj = results[best_key]["_clf"]
        out_path = os.path.join(ARTIFACTS_DIR, f"{alg}_best.joblib")
        joblib.dump(clf_obj, out_path)
        print(f"  Saved {alg} model → {out_path}")

   
    y_test_arr = np.asarray(y_test)
    np.save(os.path.join(ARTIFACTS_DIR, "y_test.npy"), y_test_arr)
    print(f"  Saved y_test ({len(y_test_arr):,} samples)")

  
    for alg, best_key in best_models.items():
        r = results[best_key]
        pred_arr = np.asarray(r["y_pred"])
        np.save(os.path.join(ARTIFACTS_DIR, f"{alg}_y_pred.npy"), pred_arr)
        if r["y_prob"] is not None:
            prob_arr = np.asarray(r["y_prob"])
            np.save(os.path.join(ARTIFACTS_DIR, f"{alg}_y_prob.npy"), prob_arr)
        print(f"  Saved {alg} predictions/probabilities")

 
    xtest_path = os.path.join(ARTIFACTS_DIR, "X_test.npz")
    scipy.sparse.save_npz(xtest_path, X_test.astype(np.float32))
    print(f"  Saved X_test sparse matrix → {xtest_path}")

  
    metrics_to_save: dict = {}
    for key, r in results.items():
        metrics_to_save[key] = {
            "accuracy":       r["accuracy"],
            "spam_precision": r["spam_precision"],
            "spam_recall":    r["spam_recall"],
            "spam_f1":        r["spam_f1"],
            "roc_auc":        r["roc_auc"],
            "cm":             r["cm"],
            "clf_name":       r["clf_name"],
            "sampler_name":   r["sampler_name"],
            "training_time":  r["training_time"],
            "is_best":        r["is_best"],
        }

    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics_to_save, fh, indent=2)
    print(f"  Saved metrics → {metrics_path}")

    gc.collect()
    memory_usage()


    print("\n" + "=" * 70)
    print("STEP 11 — Final Summary (all 9 models, sorted by Spam F1 ↓)")
    print("=" * 70)

    sorted_keys = sorted(results.keys(), key=lambda k: results[k]["spam_f1"], reverse=True)

    header = f"{'Model':<30} {'Accuracy':>9} {'Spam P':>8} {'Spam R':>8} {'Spam F1':>9} {'ROC AUC':>9} {'Best':>5}"
    print(f"\n  {header}")
    print(f"  {'-' * len(header)}")
    for key in sorted_keys:
        r = results[key]
        roc_str = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "  N/A "
        best_str = "★" if r["is_best"] else ""
        print(
            f"  {key:<30} {r['accuracy']:>9.4f} {r['spam_precision']:>8.4f}"
            f" {r['spam_recall']:>8.4f} {r['spam_f1']:>9.4f}"
            f" {roc_str:>9} {best_str:>5}"
        )

    print(f"\n  Total elapsed: {elapsed(total_start)}")
    print("\n Training complete!")



if __name__ == "__main__":
    main()
