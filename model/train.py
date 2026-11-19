import os, gc, json, time, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, psutil, scipy.sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, f1_score)
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


DATA_PATH     = "data/processed/yelp_nyc_clean.parquet"
ARTIFACTS_DIR = "artifacts"
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
N_JOBS        = 4


def memory_usage():
    rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    print(f"  [Memory] RSS: {rss:.1f} MB")

def elapsed(start):
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60}s"


def main():
    total_start = time.time()


    print("\n=== STEP 1: Load Data ===")
    df = pd.read_parquet(DATA_PATH)
    print(f"  Shape: {df.shape}")
    vc = df['label'].value_counts()
    total = len(df)
    print(f"  Label distribution:")
    print(f"    0 (Genuine): {vc.get(0, 0):,}  ({vc.get(0, 0)/total*100:.1f}%)")
    print(f"    1 (Spam):    {vc.get(1, 0):,}  ({vc.get(1, 0)/total*100:.1f}%)")
    memory_usage()
    gc.collect()


    print("\n=== STEP 2: Stratified Split ===")
    y = df['label'].values


    idx_trainval, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.125, random_state=RANDOM_STATE,
        stratify=y[idx_trainval]
    )

    train_texts = df['text'].iloc[idx_train].reset_index(drop=True)
    val_texts   = df['text'].iloc[idx_val].reset_index(drop=True)
    test_texts  = df['text'].iloc[idx_test].reset_index(drop=True)
    y_train = y[idx_train]
    y_val   = y[idx_val]
    y_test  = y[idx_test]


    meta_cols = [c for c in ['rating', 'product_id', 'review_date'] if c in df.columns]
    train_df_meta = df[meta_cols].iloc[idx_train].reset_index(drop=True) if meta_cols else None
    val_df_meta   = df[meta_cols].iloc[idx_val].reset_index(drop=True)   if meta_cols else None
    test_df_meta  = df[meta_cols].iloc[idx_test].reset_index(drop=True)  if meta_cols else None

    print(f"  Train: {len(train_texts):,} | Val: {len(val_texts):,} | Test: {len(test_texts):,}")

    for split_name, split_y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        vc_s = pd.Series(split_y).value_counts()
        n_s  = len(split_y)
        print(f"  {split_name} class dist — 0 (Genuine): {vc_s.get(0,0):,} "
              f"({vc_s.get(0,0)/n_s*100:.1f}%) | "
              f"1 (Spam): {vc_s.get(1,0):,} ({vc_s.get(1,0)/n_s*100:.1f}%)")


    print("\n=== STEP 3: Initialize Preprocessor & Vectorizers ===")
    preprocessor = TextPreprocessor(stemming=True, remove_stopwords=True)
    word_vec = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), min_df=3,
        sublinear_tf=True, dtype=np.float32
    )
    char_vec = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3, 5), max_features=15000,
        min_df=5, sublinear_tf=True, dtype=np.float32
    )
    scaler = StandardScaler(with_mean=False)

    print(f"  TextPreprocessor  : stemming=True, remove_stopwords=True")
    print(f"  word_vec (TF-IDF) : max_features=20000, ngram=(1,2), min_df=3, sublinear_tf=True")
    print(f"  char_vec (TF-IDF) : analyzer=char_wb, ngram=(3,5), max_features=15000, min_df=5, sublinear_tf=True")
    print(f"  scaler            : StandardScaler(with_mean=False)")


    print("\n=== STEP 4: Preprocess Text ===")
    cleaned_train, aux_train = preprocessor.fit_transform(train_texts, train_df_meta)
    cleaned_val,   aux_val   = preprocessor.transform(val_texts,   val_df_meta)
    cleaned_test,  aux_test  = preprocessor.transform(test_texts,  test_df_meta)
    gc.collect()
    memory_usage()


    print("\n=== STEP 5: Build Feature Matrices ===")
    X_train = build_features(cleaned_train, aux_train, word_vec, char_vec, scaler, is_train=True)
    X_val   = build_features(cleaned_val,   aux_val,   word_vec, char_vec, scaler, is_train=False)
    X_test  = build_features(cleaned_test,  aux_test,  word_vec, char_vec, scaler, is_train=False)
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val   shape: {X_val.shape}")
    print(f"  X_test  shape: {X_test.shape}")
    gc.collect()
    memory_usage()


    samplers = [
        ("Original",     None),
        ("OverSampled",  RandomOverSampler(random_state=42)),
        ("UnderSampled", RandomUnderSampler(random_state=42)),
    ]


    classifiers = [
        ("XGBoost", XGBClassifier(
            tree_method="hist", max_bin=256, n_estimators=450, max_depth=6,
            learning_rate=0.08, subsample=0.8, colsample_bytree=0.7,
            n_jobs=N_JOBS, random_state=RANDOM_STATE, eval_metric="logloss"
        )),
        ("LinearSVC", CalibratedClassifierCV(
            LinearSVC(C=0.5, dual="auto", max_iter=2000, random_state=RANDOM_STATE), cv=3
        )),
        ("SGD", SGDClassifier(
            loss="log_loss", alpha=1e-5, penalty="elasticnet", l1_ratio=0.15,
            max_iter=1000, n_jobs=N_JOBS, random_state=RANDOM_STATE
        )),
    ]


    from sklearn.base import clone

    print("\n=== STEP 8: Training Loop ===")
    results = {}

    for clf_name, clf in classifiers:
        for sampler_name, sampler in samplers:
            key = f"{clf_name}_{sampler_name}"
            print(f"\n--- Training {key} ---")
            t0 = time.time()


            if sampler is not None:
                X_res, y_res = sampler.fit_resample(X_train, y_train)
            else:
                X_res, y_res = X_train, y_train


            clf_fitted = clone(clf)
            clf_fitted.fit(X_res, y_res)
            training_time = time.time() - t0


            y_val_prob = (clf_fitted.predict_proba(X_val)[:, 1]
                          if hasattr(clf_fitted, 'predict_proba') else None)

            if y_val_prob is not None:
                thresholds = np.arange(0.20, 0.86, 0.01)
                best_thresh, best_val_f1 = 0.50, 0.0
                for thresh in thresholds:
                    vf1 = f1_score(y_val, (y_val_prob >= thresh).astype(int),
                                   pos_label=1, zero_division=0)
                    if vf1 > best_val_f1:
                        best_val_f1, best_thresh = vf1, float(thresh)
                print(f"  Optimal threshold: {best_thresh:.2f} "
                      f"(val Spam F1={best_val_f1:.4f})")
            else:
                best_thresh = 0.50


            y_prob = (clf_fitted.predict_proba(X_test)[:, 1]
                      if hasattr(clf_fitted, 'predict_proba') else None)
            if y_prob is not None:
                y_pred = (y_prob >= best_thresh).astype(int)
            else:
                y_pred = clf_fitted.predict(X_test)


            accuracy       = float(accuracy_score(y_test, y_pred))
            report         = classification_report(
                y_test, y_pred, target_names=['Genuine', 'Spam'], output_dict=True
            )
            spam_precision = float(report['Spam']['precision'])
            spam_recall    = float(report['Spam']['recall'])
            spam_f1        = float(report['Spam']['f1-score'])
            roc_auc        = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None
            cm             = confusion_matrix(y_test, y_pred)


            print(f"  Accuracy    : {accuracy:.4f}")
            print(f"  Spam Prec   : {spam_precision:.4f}")
            print(f"  Spam Recall : {spam_recall:.4f}")
            print(f"  Spam F1     : {spam_f1:.4f}")
            print(f"  ROC AUC     : {roc_auc:.4f}" if roc_auc is not None else "  ROC AUC     : N/A")
            print(f"  Train time  : {training_time:.1f}s")
            print(f"  Threshold   : {best_thresh:.2f}")
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
                "training_time":     float(training_time),
                "optimal_threshold": best_thresh,
                "_clf":              clf_fitted,
            }
            gc.collect()
            memory_usage()


    print("\n=== STEP 9: Best Model per Algorithm ===")
    best_models = {}
    for alg in ["XGBoost", "LinearSVC", "SGD"]:
        candidates = {k: v for k, v in results.items() if v["clf_name"] == alg}
        best_key = max(candidates, key=lambda k: candidates[k]["spam_f1"])
        best_models[alg] = best_key
        print(f"  Best {alg}: {best_key} "
              f"(Spam F1={candidates[best_key]['spam_f1']:.4f}, "
              f"threshold={candidates[best_key]['optimal_threshold']:.2f})")
    for key in results:
        results[key]["is_best"] = (key in best_models.values())


    print("\n=== STEP 10: Save Artifacts ===")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    save_preprocessor(preprocessor, word_vec, char_vec, scaler, ARTIFACTS_DIR)

    for alg, best_key in best_models.items():
        clf_path = f"{ARTIFACTS_DIR}/{alg}_best.joblib"
        joblib.dump(results[best_key]["_clf"], clf_path)
        print(f"  Saved {alg} model → {clf_path}")

    np.save(f"{ARTIFACTS_DIR}/y_test.npy", np.asarray(y_test))

    for alg, best_key in best_models.items():
        np.save(
            f"{ARTIFACTS_DIR}/{alg}_y_pred.npy",
            np.asarray(results[best_key]["y_pred"])
        )
        if results[best_key]["y_prob"] is not None:
            np.save(
                f"{ARTIFACTS_DIR}/{alg}_y_prob.npy",
                np.asarray(results[best_key]["y_prob"])
            )

    scipy.sparse.save_npz(
        f"{ARTIFACTS_DIR}/X_test.npz",
        X_test.astype(np.float32)
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

    with open(f"{ARTIFACTS_DIR}/metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    print(" Training complete!")


    print("\n=== STEP 11: Final Summary (sorted by Spam F1 desc) ===")
    rows = sorted(
        [(k, v) for k, v in results.items()],
        key=lambda x: x[1]["spam_f1"],
        reverse=True
    )
    header = (f"{'Model':<35} {'Accuracy':>10} {'Spam P':>8} {'Spam R':>8} "
              f"{'Spam F1':>9} {'ROC AUC':>9} {'Threshold':>10} {'Time(s)':>8}")
    print(header)
    print("-" * len(header))
    for k, v in rows:
        roc_str = f"{v['roc_auc']:.4f}" if v['roc_auc'] is not None else "   N/A"
        print(
            f"{k:<35} {v['accuracy']:>10.4f} {v['spam_precision']:>8.4f} "
            f"{v['spam_recall']:>8.4f} {v['spam_f1']:>9.4f} {roc_str:>9} "
            f"{v['optimal_threshold']:>10.2f} {v['training_time']:>8.1f}"
        )

    print(f"\nTotal elapsed time: {elapsed(total_start)}")


if __name__ == "__main__":
    main()
