"""
visualize.py — Fake Review Detection: Visualization & Publication Plots
Generates 300 DPI publication-ready PNG plots inside model/plots/ using saved
model/artifacts/metrics.json, preprocessor artifacts, and trained .joblib models.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import json
import warnings
import gc
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
import scipy.sparse
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split

from preprocessing import load_preprocessor, build_features

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "yelp_nyc_clean.parquet")

sns.set_style("whitegrid")
sns.set_palette("husl")


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)




def plot_class_distribution(df):
    print("  → Plotting 01_class_distribution.png ...")
    label_map = {0: "Genuine (0)", 1: "Spam/Fake (1)"}
    counts = df["label"].value_counts().sort_index()
    total = counts.sum()
    labels = [label_map.get(i, str(i)) for i in counts.index]
    colors = ["#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)

    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_title("Class Distribution: Genuine vs Fake/Spam Reviews", fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylim(0, counts.max() * 1.18)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_class_distribution.png"), dpi=300)
    plt.close()



def plot_rating_and_length_eda(df):
    print("  → Plotting 02_rating_and_length_eda.png ...")

    text_col = next((c for c in df.columns if c.lower() in ("review_text", "text", "review")), None)
    rating_col = next((c for c in df.columns if c.lower() in ("rating", "stars", "star")), None)
    label_col = "label"

    if text_col is None:
        warnings.warn("No review text column found; skipping plot 2.")
        return
    if rating_col is None:
        warnings.warn("No rating column found; skipping plot 2.")
        return

    genuine = df[df[label_col] == 0]
    spam = df[df[label_col] == 1]

    char_len = df[text_col].fillna("").astype(str).str.len()
    genuine_len = genuine[text_col].fillna("").astype(str).str.len()
    spam_len = spam[text_col].fillna("").astype(str).str.len()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("EDA: Rating and Review Length by Class", fontsize=16, fontweight="bold", y=1.01)


    ax = axes[0, 0]
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    ax.hist(genuine[rating_col].dropna(), bins=bins, alpha=0.6, color="#2ecc71", label="Genuine", edgecolor="white")
    ax.hist(spam[rating_col].dropna(), bins=bins, alpha=0.6, color="#e74c3c", label="Spam/Fake", edgecolor="white")
    ax.set_title("Rating Distribution by Class", fontsize=12, fontweight="bold")
    ax.set_xlabel("Stars", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(fontsize=10)


    ax = axes[0, 1]
    clip_len = 2000
    g_clip = genuine_len.clip(upper=clip_len)
    s_clip = spam_len.clip(upper=clip_len)
    if len(g_clip) > 1:
        sns.kdeplot(g_clip, ax=ax, label="Genuine", color="#2ecc71", fill=True, alpha=0.4)
    if len(s_clip) > 1:
        sns.kdeplot(s_clip, ax=ax, label="Spam/Fake", color="#e74c3c", fill=True, alpha=0.4)
    ax.set_title("Review Length Distribution by Class (KDE)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Character Length (clipped at 2000)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(0, clip_len)
    ax.legend(fontsize=10)


    ax = axes[1, 0]
    plot_df = pd.DataFrame({
        "Length": char_len,
        "Class": df[label_col].map({0: "Genuine", 1: "Spam/Fake"}),
    })
    sns.boxplot(data=plot_df, x="Class", y="Length", palette=["#2ecc71", "#e74c3c"], ax=ax, order=["Genuine", "Spam/Fake"])
    ax.set_title("Review Length by Class (Box Plot)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Character Length", fontsize=11)


    ax = axes[1, 1]
    means = [genuine[rating_col].mean(), spam[rating_col].mean()]
    stds = [genuine[rating_col].std(), spam[rating_col].std()]
    class_labels = ["Genuine", "Spam/Fake"]
    bars = ax.bar(class_labels, means, yerr=stds, capsize=6, color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.2, error_kw={"linewidth": 2})
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{m:.2f}★", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Mean Star Rating per Class (±1 std)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Mean Rating", fontsize=11)
    ax.set_ylim(0, 6)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_rating_and_length_eda.png"), dpi=300)
    plt.close()


def plot_model_comparison():
    print("  → Plotting 03_model_comparison_metrics.png ...")
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        warnings.warn(f"metrics.json not found at {metrics_path}; skipping plot 3.")
        return

    with open(metrics_path, "r") as f:
        metrics = json.load(f)


    expected_order = [
        "XGBoost_Original", "XGBoost_OverSampled", "XGBoost_UnderSampled",
        "LinearSVC_Original", "LinearSVC_OverSampled", "LinearSVC_UnderSampled",
        "SGD_Original", "SGD_OverSampled", "SGD_UnderSampled",
    ]
    model_names = [m for m in expected_order if m in metrics and isinstance(metrics[m], dict)]
    if not model_names:
        model_names = [k for k in metrics.keys() if k != "best_thresholds" and isinstance(metrics[k], dict)]

    metric_keys = ["accuracy", "spam_precision", "spam_recall", "spam_f1"]
    metric_labels = ["Accuracy", "Spam Precision", "Spam Recall", "Spam F1"]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]


    algo_best = {}
    for name in model_names:
        algo = name.split("_")[0]
        f1 = metrics[name].get("spam_f1", 0)
        if algo not in algo_best or f1 > algo_best[algo][1]:
            algo_best[algo] = (name, f1)
    best_names = {v[0] for v in algo_best.values()}

    n_models = len(model_names)
    n_metrics = len(metric_keys)
    x = np.arange(n_models)
    width = 0.18

    fig, ax = plt.subplots(figsize=(18, 8))
    for i, (mk, ml, color) in enumerate(zip(metric_keys, metric_labels, colors)):
        vals = [float(metrics[name].get(mk, 0.0) or 0.0) for name in model_names]
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=ml, color=color, alpha=0.85, edgecolor="white")


    for idx, name in enumerate(model_names):
        if name in best_names:
            ax.text(x[idx], 1.03, "★", ha="center", va="bottom", fontsize=14, color="gold", fontweight="bold", transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Comparison: All 9 Training Configurations", fontsize=15, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_model_comparison_metrics.png"), dpi=300)
    plt.close()

def plot_confusion_matrices_grid():
    print("  → Plotting 04_confusion_matrices_grid.png ...")
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        warnings.warn(f"metrics.json not found at {metrics_path}; skipping plot 4.")
        return

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    classifiers = ["XGBoost", "LinearSVC", "SGD"]
    samplers = ["Original", "OverSampled", "UnderSampled"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle("Confusion Matrices Grid: All 9 Configurations", fontsize=18, fontweight="bold", y=1.01)

    for r, clf in enumerate(classifiers):
        for c, sampler in enumerate(samplers):
            ax = axes[r, c]
            key = f"{clf}_{sampler}"
            if key not in metrics or "cm" not in metrics[key]:
                ax.text(0.5, 0.5, f"No data\n({key})", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
                ax.set_title(f"{clf} / {sampler}", fontsize=10)
                ax.axis("off")
                continue

            cm = np.array(metrics[key]["cm"])
            total = cm.sum()
            annot = np.array([
                [f"{cm[i, j]:,}\n({cm[i, j] / total * 100:.1f}%)" for j in range(cm.shape[1])]
                for i in range(cm.shape[0])
            ])
            sns.heatmap(
                cm, ax=ax, annot=annot, fmt="", cmap="Blues",
                xticklabels=["Pred Genuine", "Pred Spam"],
                yticklabels=["True Genuine", "True Spam"],
                linewidths=0.5, linecolor="white", cbar=False,
                annot_kws={"size": 10},
            )
            acc = metrics[key].get("accuracy", 0.0)
            f1 = metrics[key].get("spam_f1", 0.0)
            thresh = metrics[key].get("optimal_threshold", 0.50)
            ax.set_title(
                f"{clf} / {sampler}\nAcc: {acc * 100:.1f}% | Spam F1: {f1:.3f} (th={thresh:.2f})",
                fontsize=11, fontweight="bold",
            )
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("Actual", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_confusion_matrices_grid.png"), dpi=300)
    plt.close()


def plot_roc_and_pr_curves():
    print("  → Plotting 05_roc_and_pr_curves.png ...")
    models_config = [
        ("XGBoost", "XGBoost_best.joblib", "#3498db"),
        ("LinearSVC", "LinearSVC_best.joblib", "#e74c3c"),
        ("SGD", "SGD_best.joblib", "#2ecc71"),
    ]

  
    if not os.path.exists(DATA_PATH):
        warnings.warn(f"Dataset not found at {DATA_PATH}; skipping plot 5.")
        return

    print(f"    Loading {DATA_PATH} to recreate exact 80/20 stratified test split...")
    df = pd.read_parquet(DATA_PATH)
    y = df["label"].values
    total_len = len(df)


    _, idx_test = train_test_split(
        np.arange(total_len),
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    test_texts = df["text"].iloc[idx_test].reset_index(drop=True)
    y_test = y[idx_test]

    meta_cols = [c for c in ["user_id", "review_id", "product_id", "rating", "review_date"] if c in df.columns]
    test_df_meta = df[meta_cols].iloc[idx_test].reset_index(drop=True) if meta_cols else None
    del df
    gc.collect()

    print(f"    Test split: {len(test_texts):,} samples. Transforming with saved preprocessor artifacts...")
    preprocessor, word_vec, char_vec, scaler = load_preprocessor(ARTIFACTS_DIR)
    cleaned_test, aux_test = preprocessor.transform(test_texts, test_df_meta)
    X_test = build_features(cleaned_test, aux_test, word_vec, char_vec, scaler, is_train=False)
    print(f"    Transformed X_test shape: {X_test.shape}")

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "ROC Curves (left) and Precision-Recall Curves (right) — Best Models",
        fontsize=14, fontweight="bold", y=1.02,
    )

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")

    any_plotted = False
    for label, filename, color in models_config:
        model_path = os.path.join(ARTIFACTS_DIR, filename)
        if not os.path.exists(model_path):
            warnings.warn(f"{filename} not found at {model_path}; skipping.")
            continue

        try:
            model = joblib.load(model_path)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X_test)
                y_prob = 1.0 / (1.0 + np.exp(-scores))
            else:
                warnings.warn(f"Model {label} has neither predict_proba nor decision_function.")
                continue

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {roc_auc:.4f})")

            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            ap = average_precision_score(y_test, y_prob)
            ax_pr.plot(recall, precision, color=color, lw=2, label=f"{label} (AP = {ap:.4f})")
            any_plotted = True
            print(f"    ✓ {label}: ROC-AUC = {roc_auc:.4f}, Average Precision = {ap:.4f}")
        except Exception as e:
            warnings.warn(f"Could not compute curves for {label}: {e}")

    if not any_plotted:
        for ax in (ax_roc, ax_pr):
            ax.text(0.5, 0.5, "No model probability data generated", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")

    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate", fontsize=12)
    ax_roc.set_title("ROC Curves", fontsize=13, fontweight="bold")
    ax_roc.legend(fontsize=10, loc="lower right")
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1.05])

    ax_pr.set_xlabel("Recall", fontsize=12)
    ax_pr.set_ylabel("Precision", fontsize=12)
    ax_pr.set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
    ax_pr.legend(fontsize=10, loc="upper right")
    ax_pr.set_xlim([0, 1])
    ax_pr.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_roc_and_pr_curves.png"), dpi=300)
    plt.close()

def plot_top_spam_features():
    print("  → Plotting 06_top_spam_features.png ...")
    word_vec_path = os.path.join(ARTIFACTS_DIR, "word_vectorizer.joblib")
    char_vec_path = os.path.join(ARTIFACTS_DIR, "char_vectorizer.joblib")
    legacy_vec_path = os.path.join(ARTIFACTS_DIR, "vectorizer.joblib")
    model_path = os.path.join(ARTIFACTS_DIR, "XGBoost_best.joblib")

    if not os.path.exists(model_path):
        warnings.warn(f"XGBoost_best.joblib not found at {model_path}; skipping plot 6.")
        return

    try:
        model = joblib.load(model_path)
    except Exception as e:
        warnings.warn(f"Could not load model: {e}; skipping plot 6.")
        return

    auxiliary_names = [
        "char_length", "word_count", "exclamation_count", "uppercase_ratio",
        "first_person_ratio", "rating", "user_review_count", "user_daily_burst", "rating_deviation",
        "extreme_rating", "product_review_count", "daily_burst_count", "sentiment_mismatch",
    ]

    tfidf_names = []
    if os.path.exists(word_vec_path):
        try:
            w_vec = joblib.load(word_vec_path)
            tfidf_names += [f"word:{f}" for f in w_vec.get_feature_names_out()]
        except Exception:
            pass
    if os.path.exists(char_vec_path):
        try:
            c_vec = joblib.load(char_vec_path)
            tfidf_names += [f"char:{f}" for f in c_vec.get_feature_names_out()]
        except Exception:
            pass
    if not tfidf_names and os.path.exists(legacy_vec_path):
        try:
            l_vec = joblib.load(legacy_vec_path)
            tfidf_names += list(l_vec.get_feature_names_out())
        except Exception:
            pass

    all_names = tfidf_names + auxiliary_names

    try:
        importances = model.feature_importances_
    except Exception as e:
        warnings.warn(f"Could not get feature importances: {e}; skipping plot 6.")
        return

    n_expected = len(all_names)
    if len(importances) < n_expected:
        all_names = all_names[:len(importances)]
    elif len(importances) > n_expected:
        importances = importances[:n_expected]

    top_idx = np.argsort(importances)[-20:][::-1]
    top_names = [all_names[i] for i in top_idx]
    top_vals = importances[top_idx]
    top_colors = ["#f39c12" if all_names[i] in auxiliary_names else "#3498db" for i in top_idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_vals[::-1], color=top_colors[::-1], edgecolor="white", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Top 20 Features for Spam Detection (XGBoost)", fontsize=14, fontweight="bold", pad=12)

    blue_patch = mpatches.Patch(color="#3498db", label="TF-IDF n-gram")
    orange_patch = mpatches.Patch(color="#f39c12", label="Auxiliary feature")
    ax.legend(handles=[blue_patch, orange_patch], fontsize=11, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "06_top_spam_features.png"), dpi=300)
    plt.close()



def main():
    t0 = time.time()
    print("=== Fake Review Detection: Visualization ===")
    ensure_plots_dir()

    df = None
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_parquet(DATA_PATH, engine="pyarrow")
            print(f"  Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} cols")
            if "label" not in df.columns:
                label_col = next((c for c in df.columns if c.lower() == "label"), None)
                if label_col:
                    df = df.rename(columns={label_col: "label"})
        except Exception as e:
            warnings.warn(f"Could not load data from {DATA_PATH}: {e}")
    else:
        warnings.warn(f"Data not found at {DATA_PATH}; skipping data-dependent plots.")


    if df is not None and "label" in df.columns:
        try:
            plot_class_distribution(df)
        except Exception as e:
            warnings.warn(f"Plot 1 failed: {e}")
    else:
        print("  ⚠ Skipping plot 1 (no data).")


    if df is not None:
        try:
            plot_rating_and_length_eda(df)
        except Exception as e:
            warnings.warn(f"Plot 2 failed: {e}")
    else:
        print("  ⚠ Skipping plot 2 (no data).")

    del df
    gc.collect()


    try:
        plot_model_comparison()
    except Exception as e:
        warnings.warn(f"Plot 3 failed: {e}")


    try:
        plot_confusion_matrices_grid()
    except Exception as e:
        warnings.warn(f"Plot 4 failed: {e}")


    try:
        plot_roc_and_pr_curves()
    except Exception as e:
        warnings.warn(f"Plot 5 failed: {e}")


    try:
        plot_top_spam_features()
    except Exception as e:
        warnings.warn(f"Plot 6 failed: {e}")

    print(f"\n All visualizations generated in {time.time() - t0:.1f}s. Saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
