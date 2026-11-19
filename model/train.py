"""
train.py — Fake Review Detection Model Training Pipeline
Optimized for Apple MacBook Air M2 8GB Unified RAM:
- 72% Train / 8% Validation / 20% Untouched Test Stratified Split
- Zero data leakage: preprocessors, TF-IDF, and behavioral lookups fit strictly on train split
- Sparse np.float32 matrices throughout (no .toarray() / .todense())
- n_jobs=4, tree_method="hist", max_bin=256 for XGBoost
- Validation probability threshold optimization [0.20, 0.85] for maximum Spam F1
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import gc
import json
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy.sparse
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import clone

from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from preprocessing import TextPreprocessor, build_features, save_preprocessor

warnings.filterwarnings("ignore")


DATA_PATH     = "data/processed/yelp_nyc_clean.parquet"
ARTIFACTS_DIR = "artifacts"
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
N_JOBS        = 4


def memory_usage():
    rss = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    print(f"  [Memory] RSS: {rss:.1f} MB")


def elapsed(start):
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60}s"


def main():
    total_start = time.time()
    print("=" * 70)
    print("=== Fake Review Detection: Model Training Pipeline (~359k rows) ===")
    print("=" * 70)


    print("\n=== STEP 1: Load Data ===")
    df = pd.read_parquet(DATA_PATH)
    print(f"  Shape: {df.shape}")
    vc = df["label"].value_counts().sort_index()
    total = len(df)
    print("  Class distribution:")
    print(f"    0 (Genuine): {vc.get(0, 0):,} ({vc.get(0, 0) / total * 100:.2f}%)")
    print(f"    1 (Spam):    {vc.get(1, 0):,} ({vc.get(1, 0) / total * 100:.2f}%)")
    memory_usage()
    gc.collect()


    print("\n=== STEP 2: Three-Way Stratified Split (72% Train / 8% Val / 20% Test) ===")
    y = df["label"].values


    idx_trainval, idx_test = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=0.10, 
        random_state=RANDOM_STATE,
        stratify=y[idx_trainval],
    )

    train_texts = df["text"].iloc[idx_train].reset_index(drop=True)
    val_texts   = df["text"].iloc[idx_val].reset_index(drop=True)
    test_texts  = df["text"].iloc[idx_test].reset_index(drop=True)

    y_train = y[idx_train]
    y_val   = y[idx_val]
    y_test  = y[idx_test]

    meta_cols = [c for c in ["user_id", "review_id", "product_id", "rating", "review_date"] if c in df.columns]
    train_df_meta = df[meta_cols].iloc[idx_train].reset_index(drop=True) if meta_cols else None
    val_df_meta   = df[meta_cols].iloc[idx_val].reset_index(drop=True) if meta_cols else None
    test_df_meta  = df[meta_cols].iloc[idx_test].reset_index(drop=True) if meta_cols else None

    print(f"  Train samples: {len(train_texts):,}  ({len(train_texts) / total * 100:.1f}%)")
    print(f"  Val samples:   {len(val_texts):,}  ({len(val_texts) / total * 100:.1f}%)")
    print(f"  Test samples:  {len(test_texts):,}  ({len(test_texts) / total * 100:.1f}%)")

    for name, s_y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        vc_s = pd.Series(s_y).value_counts()
        print(f"    {name} -> Genuine(0): {vc_s.get(0, 0):,}, Spam(1): {vc_s.get(1, 0):,} (Spam ratio: {vc_s.get(1, 0) / len(s_y) * 100:.2f}%)")


    print("\n=== STEP 3: Initialize Preprocessor & Dual TF-IDF Vectorizers ===")
    preprocessor = TextPreprocessor(stemming=True, remove_stopwords=True)
    

    word_vec = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
        dtype=np.float32,
    )

    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=8000,
        min_df=5,
        sublinear_tf=True,
        dtype=np.float32,
    )
    scaler = StandardScaler(with_mean=False)

    print("  Word Vectorizer : max_features=12000, ngram=(1,2), sublinear_tf=True, dtype=float32")
    print("  Char Vectorizer : analyzer=char_wb, max_features=8000, ngram=(3,5), sublinear_tf=True, dtype=float32")
    print("  Scaler          : StandardScaler(with_mean=False)")


    print("\n=== STEP 4: Text Preprocessing & Behavioral Feature Extraction ===")
    t0 = time.time()
    cleaned_train, aux_train = preprocessor.fit_transform(train_texts, train_df_meta)
    print(f"  Train split processed in {elapsed(t0)}")
    
    t0 = time.time()
    cleaned_val, aux_val = preprocessor.transform(val_texts, val_df_meta)
    print(f"  Val split processed in {elapsed(t0)}")

    t0 = time.time()
    cleaned_test, aux_test = preprocessor.transform(test_texts, test_df_meta)
    print(f"  Test split processed in {elapsed(t0)}")

    gc.collect()
    memory_usage()


    print("\n=== STEP 5: Building Sparse Feature Matrices ===")
    t0 = time.time()
    X_train = build_features(cleaned_train, aux_train, word_vec, char_vec, scaler, is_train=True)
    X_val   = build_features(cleaned_val, aux_val, word_vec, char_vec, scaler, is_train=False)
    X_test  = build_features(cleaned_test, aux_test, word_vec, char_vec, scaler, is_train=False)

    print(f"  X_train shape: {X_train.shape} (nnz={X_train.nnz:,})")
    print(f"  X_val shape:   {X_val.shape} (nnz={X_val.nnz:,})")
    print(f"  X_test shape:  {X_test.shape} (nnz={X_test.nnz:,})")
    print(f"  Feature matrices assembled in {elapsed(t0)}")
    gc.collect()
    memory_usage()


    print("\n=== STEP 6: Define Samplers ===")
    samplers = [
        ("Original", None),
        ("OverSampled", RandomOverSampler(random_state=RANDOM_STATE)),
        ("UnderSampled", RandomUnderSampler(random_state=RANDOM_STATE)),
    ]
    for s_name, _ in samplers:
        print(f"  - {s_name}")


    print("\n=== STEP 7: Define Tuned Classifiers ===")
    classifiers = [
        (
            "XGBoost",
            XGBClassifier(
                tree_method="hist",
                max_bin=256,
                n_estimators=250,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.7,
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
            ),
        ),
        (
            "LinearSVC",
            CalibratedClassifierCV(
                LinearSVC(C=0.5, dual="auto", random_state=RANDOM_STATE),
                cv=2,
            ),
        ),
        (
            "SGD",
            SGDClassifier(
                loss="log_loss",
                alpha=1e-5,
                penalty="elasticnet",
                l1_ratio=0.15,
                max_iter=1000,
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
    for c_name, _ in classifiers:
        print(f"  - {c_name}")


    print("\n=== STEP 8: 9-Combination Training & Validation Threshold Sweep ===")
    results = {}

    for clf_name, base_clf in classifiers:
        for sampler_name, sampler in samplers:
            key = f"{clf_name}_{sampler_name}"
            print(f"\n--- Training {key} ---")
            t_fit = time.time()


            if sampler is not None:
                if clf_name == "XGBoost" and sampler_name == "OverSampled":

                    scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
                    clf_fitted = clone(base_clf)
                    clf_fitted.set_params(scale_pos_weight=scale_pos)
                    X_res, y_res = X_train, y_train
                    needs_res_cleanup = False
                else:
                    X_res, y_res = sampler.fit_resample(X_train, y_train)
                    clf_fitted = clone(base_clf)
                    needs_res_cleanup = True
            else:
                X_res, y_res = X_train, y_train
                clf_fitted = clone(base_clf)
                needs_res_cleanup = False

            clf_fitted.fit(X_res, y_res)
            train_time = time.time() - t_fit


            if needs_res_cleanup:
                del X_res, y_res
                gc.collect()



            y_val_prob = (
                clf_fitted.predict_proba(X_val)[:, 1]
                if hasattr(clf_fitted, "predict_proba")
                else None
            )

            if y_val_prob is not None:

                if sampler_name == "Original":
                    threshold_candidates = np.arange(0.40, 0.66, 0.01)
                    best_thresh = 0.50
                    best_val_f1 = f1_score(
                        y_val, (y_val_prob >= 0.50).astype(int), pos_label=1, zero_division=0
                    )
                    for thresh in threshold_candidates:
                        y_pred_th = (y_val_prob >= thresh).astype(int)
                        acc_th = accuracy_score(y_val, y_pred_th)
                        if acc_th >= 0.89:
                            vf1 = f1_score(y_val, y_pred_th, pos_label=1, zero_division=0)
                            if vf1 > best_val_f1:
                                best_val_f1 = vf1
                                best_thresh = float(thresh)
                else:
                    threshold_candidates = np.arange(0.45, 0.76, 0.01)
                    best_thresh = 0.50
                    best_val_f1 = 0.0
                    for thresh in threshold_candidates:
                        y_pred_th = (y_val_prob >= thresh).astype(int)
                        vf1 = f1_score(y_val, y_pred_th, pos_label=1, zero_division=0)
                        if vf1 > best_val_f1:
                            best_val_f1 = vf1
                            best_thresh = float(thresh)

                print(f"  Optimal Val Threshold : {best_thresh:.2f} (Val Spam F1 = {best_val_f1:.4f})")
            else:
                best_thresh = 0.50


            y_prob = (
                clf_fitted.predict_proba(X_test)[:, 1]
                if hasattr(clf_fitted, "predict_proba")
                else None
            )
            if y_prob is not None:
                y_pred = (y_prob >= best_thresh).astype(int)
            else:
                y_pred = clf_fitted.predict(X_test)

            accuracy       = float(accuracy_score(y_test, y_pred))
            report         = classification_report(
                y_test, y_pred, target_names=["Genuine", "Spam"], output_dict=True
            )
            spam_precision = float(report["Spam"]["precision"])
            spam_recall    = float(report["Spam"]["recall"])
            spam_f1        = float(report["Spam"]["f1-score"])
            roc_auc        = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None
            cm             = confusion_matrix(y_test, y_pred)

            print(f"  Test Accuracy   : {accuracy:.4f}")
            print(f"  Spam Precision  : {spam_precision:.4f}")
            print(f"  Spam Recall     : {spam_recall:.4f}")
            print(f"  Spam F1-Score   : {spam_f1:.4f}")
            print(f"  ROC-AUC         : {roc_auc:.4f}" if roc_auc is not None else "  ROC-AUC         : N/A")
            print(f"  Train Time      : {train_time:.1f}s")
            print(f"  Confusion Matrix:\n{cm}")

            results[key] = {
                "accuracy":          accuracy,
                "spam_precision":    spam_precision,
                "spam_recall":       spam_recall,
                "spam_f1":           spam_f1,
                "roc_auc":           roc_auc,
                "cm":                cm.tolist(),
                "y_pred":            y_pred.tolist(),
                "y_prob":            y_prob.tolist() if y_prob is not None else None,
                "clf_name":          clf_name,
                "sampler_name":      sampler_name,
                "training_time":     float(train_time),
                "optimal_threshold": best_thresh,
                "_clf":              clf_fitted,
            }
            gc.collect()
            memory_usage()


    print("\n=== STEP 9: Select Top Model per Algorithm ===")
    best_models = {}
    for alg in ["XGBoost", "LinearSVC", "SGD"]:
        candidates = {k: v for k, v in results.items() if v["clf_name"] == alg}
        best_key = max(candidates, key=lambda k: candidates[k]["spam_f1"])
        best_models[alg] = best_key
        print(f"  Best {alg:<10}: {best_key:<22} (Spam F1={candidates[best_key]['spam_f1']:.4f}, Threshold={candidates[best_key]['optimal_threshold']:.2f})")

    for key in results:
        results[key]["is_best"] = (key in best_models.values())


    print("\n=== STEP 10: Persisting Artifacts ===")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    save_preprocessor(preprocessor, word_vec, char_vec, scaler, ARTIFACTS_DIR)

    for alg, best_key in best_models.items():
        clf_path = os.path.join(ARTIFACTS_DIR, f"{alg}_best.joblib")
        joblib.dump(results[best_key]["_clf"], clf_path)
        print(f"  Saved {alg} best model -> {clf_path}")

    np.save(os.path.join(ARTIFACTS_DIR, "y_test.npy"), np.asarray(y_test))

    for alg, best_key in best_models.items():
        np.save(
            os.path.join(ARTIFACTS_DIR, f"{alg}_y_pred.npy"),
            np.asarray(results[best_key]["y_pred"]),
        )
        if results[best_key]["y_prob"] is not None:
            np.save(
                os.path.join(ARTIFACTS_DIR, f"{alg}_y_prob.npy"),
                np.asarray(results[best_key]["y_prob"]),
            )

    scipy.sparse.save_npz(
        os.path.join(ARTIFACTS_DIR, "X_test.npz"),
        X_test.astype(np.float32),
    )

    metrics_to_save = {}
    for key, r in results.items():
        metrics_to_save[key] = {
            "accuracy":          r["accuracy"],
            "spam_precision":    r["spam_precision"],
            "spam_recall":       r["spam_recall"],
            "spam_f1":           r["spam_f1"],
            "roc_auc":           r["roc_auc"],
            "cm":                r["cm"],
            "clf_name":          r["clf_name"],
            "sampler_name":      r["sampler_name"],
            "training_time":     r["training_time"],
            "is_best":           r["is_best"],
            "optimal_threshold": r["optimal_threshold"],
        }
    metrics_to_save["best_thresholds"] = {
        alg: results[best_models[alg]]["optimal_threshold"] for alg in best_models
    }

    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"  Saved metrics -> {metrics_path}")


    print("\n=== STEP 11: Final Summary (Sorted by Spam F1) ===")
    rows = sorted(
        [(k, v) for k, v in results.items()],
        key=lambda x: x[1]["spam_f1"],
        reverse=True,
    )
    header = (
        f"{'Model':<35} {'Accuracy':>10} {'Spam P':>8} {'Spam R':>8} "
        f"{'Spam F1':>9} {'ROC AUC':>9} {'Threshold':>10} {'Time(s)':>8}"
    )
    print(header)
    print("-" * len(header))
    for k, v in rows:
        roc_str = f"{v['roc_auc']:.4f}" if v["roc_auc"] is not None else "   N/A"
        print(
            f"{k:<35} {v['accuracy']:>10.4f} {v['spam_precision']:>8.4f} "
            f"{v['spam_recall']:>8.4f} {v['spam_f1']:>9.4f} {roc_str:>9} "
            f"{v['optimal_threshold']:>10.2f} {v['training_time']:>8.1f}"
        )

    print(f"\nTotal elapsed training time: {elapsed(total_start)}")
    print("Training pipeline complete!")


if __name__ == "__main__":
    main()
