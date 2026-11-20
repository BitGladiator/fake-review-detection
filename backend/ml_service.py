"""
ml_service.py — Machine Learning Service for Fake Review Detection
Wraps preprocessor, vectorizers, scalers, and trained models (XGBoost, LinearSVC, SGD).
Loaded once at startup; serves single and batch inference with explainability signals.
"""

import os
import sys
import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse
import joblib
import json


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
ARTIFACTS_DIR = os.path.join(MODEL_DIR, "artifacts")
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")

if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from preprocessing import (
    TextPreprocessor,
    build_features,
    load_preprocessor,
    DEFAULT_MODEL_THRESHOLDS,
)
from schemas import (
    PredictRequest,
    PredictResponse,
    ModelPrediction,
    ExplainabilitySignals,
    BatchRowPrediction,
    BatchSummary,
    BatchPredictResponse,
)

logger = logging.getLogger("backend.ml_service")


class MLService:
    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.preprocessor: Optional[TextPreprocessor] = None
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.scaler = None
        self.models: Dict[str, Any] = {}
        self.thresholds: Dict[str, float] = {
            "XGBoost": 0.64,
            "LinearSVC": 0.63,
            "SGD": 0.73,
        }
        self.metrics_data: Dict[str, Any] = {}
        self.is_loaded: bool = False

        # In-memory store for generated batch export CSVs
        self.export_cache: Dict[str, str] = {}

    def load_artifacts(self) -> None:
        """Load preprocessor, vectorizers, scalers, and all 3 models into memory once."""
        t0 = time.perf_counter()
        logger.info("Loading ML artifacts from: %s", self.artifacts_dir)

        if not os.path.isdir(self.artifacts_dir):
            raise FileNotFoundError(f"Artifacts directory not found: {self.artifacts_dir}")

        # 1. Load Preprocessor, Vectorizers, and Scaler
        self.preprocessor, self.word_vectorizer, self.char_vectorizer, self.scaler = load_preprocessor(
            self.artifacts_dir
        )
        logger.info("Loaded preprocessor, dual TF-IDF vectorizers, and scaler.")

        # 2. Load metrics.json for optimal thresholds & model metrics
        metrics_file = os.path.join(self.artifacts_dir, "metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r") as f:
                    self.metrics_data = json.load(f)
                best_th = self.metrics_data.get("best_thresholds", {})
                for k, v in best_th.items():
                    if k in self.thresholds and isinstance(v, (int, float)):
                        self.thresholds[k] = float(v)
                logger.info("Loaded model thresholds from metrics.json: %s", self.thresholds)
            except Exception as e:
                logger.warning("Could not parse metrics.json: %s. Using default thresholds.", e)

        # 3. Load Best Classifiers
        model_map = {
            "XGBoost": "XGBoost_best.joblib",
            "LinearSVC": "LinearSVC_best.joblib",
            "SGD": "SGD_best.joblib",
        }

        for model_name, filename in model_map.items():
            path = os.path.join(self.artifacts_dir, filename)
            if not os.path.exists(path):
                # Fallback to non-best filename if exists
                fallback = os.path.join(self.artifacts_dir, f"{model_name}.joblib")
                if os.path.exists(fallback):
                    path = fallback
                else:
                    logger.error("Model file missing: %s", path)
                    continue
            self.models[model_name] = joblib.load(path)
            logger.info("Loaded %s model from %s (thresh=%.2f)", model_name, os.path.basename(path), self.thresholds.get(model_name, 0.50))

        if len(self.models) < 3:
            logger.warning("Only %d/3 models successfully loaded: %s", len(self.models), list(self.models.keys()))

        self.is_loaded = True
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("MLService initialized in %.1f ms. Ready for inference.", elapsed)

    def extract_signals(self, text: str, rating: float) -> ExplainabilitySignals:
        """Compute linguistic and behavioral risk signals with human-readable explanations."""
        char_length = len(text)
        words = text.split()
        word_count = len(words)
        exclamation_count = text.count("!")
        alpha_chars = [c for c in text if c.isalpha()]
        uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)

        # Sentiment scoring via VADER if available
        sentiment_compound = 0.0
        sentiment_rescaled = 3.0
        if self.preprocessor is not None and getattr(self.preprocessor, "_sia", None) is not None:
            try:
                scores = self.preprocessor._sia.polarity_scores(text)
                sentiment_compound = float(scores.get("compound", 0.0))
                # Map [-1.0, 1.0] to [1.0, 5.0] star scale
                sentiment_rescaled = float((sentiment_compound + 1.0) / 2.0 * 4.0 + 1.0)
            except Exception:
                pass
        else:
            # Simple keyword sentiment heuristic
            t_low = text.lower()
            pos_words = ("great", "amazing", "excellent", "best", "good", "love", "perfect", "delicious", "recommend")
            neg_words = ("terrible", "worst", "horrible", "bad", "awful", "rude", "dirty", "poor", "slow", "waste")
            pos_cnt = sum(1 for w in pos_words if w in t_low)
            neg_cnt = sum(1 for w in neg_words if w in t_low)
            if pos_cnt > neg_cnt:
                sentiment_compound = 0.6
                sentiment_rescaled = 4.2
            elif neg_cnt > pos_cnt:
                sentiment_compound = -0.6
                sentiment_rescaled = 1.8

        sentiment_mismatch = abs(sentiment_rescaled - rating)

        # Generate intelligent risk & authenticity signals
        risk_signals: List[str] = []
        t_lower = text.lower()

        # High risk flags
        if exclamation_count >= 3:
            risk_signals.append(f"High exclamation count ({exclamation_count}) indicates promotional hyperbole.")

        if uppercase_ratio >= 0.15 and len(alpha_chars) >= 15:
            risk_signals.append(f"High uppercase letter ratio ({uppercase_ratio * 100:.1f}%) suggests forced urgency or spam phrasing.")

        if word_count < 10:
            risk_signals.append("Short review length (<10 words) lacking substantive consumer details.")

        superlative_buzzwords = ["best restaurant ever", "must visit", "5 stars", "perfection", "greatest", "perfect perfect"]
        if any(sw in t_lower for sw in superlative_buzzwords) and rating >= 4.5:
            risk_signals.append("Repeated generic superlative praise combined with extreme 5-star rating.")

        incentive_cues = ["paid to write", "free food", "gift card", "asked me to give", "discount in exchange", "promotional review"]
        if any(ic in t_lower for ic in incentive_cues):
            risk_signals.append("Review explicitly references incentivized or compensated posting.")

        if sentiment_mismatch >= 1.8:
            risk_signals.append(f"Emotional mismatch: text tone ({sentiment_rescaled:.1f}/5) sharply deviates from star rating ({rating:.1f}/5).")

        # Positive / authenticity flags
        if word_count >= 20 and exclamation_count <= 1 and uppercase_ratio < 0.10:
            risk_signals.append("Balanced paragraph structure with natural vocabulary and descriptive depth.")

        if sentiment_mismatch <= 0.6:
            risk_signals.append("Strong emotional consistency between star rating and written review text.")

        if not risk_signals:
            risk_signals.append("Standard review characteristics matching typical consumer feedback patterns.")

        return ExplainabilitySignals(
            word_count=word_count,
            char_length=char_length,
            exclamation_count=exclamation_count,
            uppercase_ratio=round(uppercase_ratio, 4),
            sentiment_score=round(sentiment_compound, 3),
            sentiment_rating_mismatch=round(sentiment_mismatch, 3),
            top_risk_signals=risk_signals,
        )

    def predict_single(
        self,
        text: str,
        rating: float = 4.0,
        model_choice: str = "ensemble",
        user_id: Optional[int] = None,
        product_id: Optional[int] = None,
    ) -> PredictResponse:
        """
        Evaluate a single review across XGBoost, LinearSVC, and SGD classifiers in <15ms.
        Extracts features once and evaluates all 3 models side-by-side.
        """
        if not self.is_loaded:
            self.load_artifacts()

        t_start = time.perf_counter()

        # 1. Prepare 1-row DataFrame for behavioral feature extractor
        meta_df = pd.DataFrame({
            "rating": [rating if rating is not None else 4.0],
            "user_id": [user_id],
            "product_id": [product_id],
            "review_date": [None],
        })

        # 2. Extract features once
        series = pd.Series([text])
        cleaned, aux = self.preprocessor.transform(series, meta_df)
        X = build_features(
            cleaned,
            aux,
            self.word_vectorizer,
            self.char_vectorizer,
            self.scaler,
            is_train=False,
        )

        # 3. Predict across each individual model
        all_model_preds: List[ModelPrediction] = []
        model_order = ["XGBoost", "LinearSVC", "SGD"]
        model_map_by_name: Dict[str, ModelPrediction] = {}

        for m_name in model_order:
            if m_name not in self.models:
                continue
            model = self.models[m_name]
            thresh = self.thresholds.get(m_name, 0.50)

            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X)[0, 1])
            else:
                prob = float(model.predict(X)[0])

            is_spam = bool(prob >= thresh)
            pred_label = "Fake" if is_spam else "Genuine"
            conf = prob if is_spam else (1.0 - prob)

            m_pred = ModelPrediction(
                model_name=m_name,
                prediction=pred_label,
                is_spam=is_spam,
                spam_probability=round(prob, 4),
                confidence=round(conf, 4),
                threshold=round(thresh, 4),
            )
            all_model_preds.append(m_pred)
            model_map_by_name[m_name.lower()] = m_pred

        # 4. Compute Ensemble Verdict (Majority Vote)
        fake_votes = sum(1 for m in all_model_preds if m.is_spam)
        total_models = len(all_model_preds) or 1
        ensemble_is_spam = fake_votes >= (total_models / 2.0)
        ensemble_verdict = "Fake" if ensemble_is_spam else "Genuine"

        # Consensus confidence
        avg_spam_prob = sum(m.spam_probability for m in all_model_preds) / total_models
        ensemble_conf = avg_spam_prob if ensemble_is_spam else (1.0 - avg_spam_prob)

        ensemble_model_pred = ModelPrediction(
            model_name="Ensemble (Majority Vote)",
            prediction=ensemble_verdict,
            is_spam=ensemble_is_spam,
            spam_probability=round(avg_spam_prob, 4),
            confidence=round(ensemble_conf, 4),
            threshold=0.50,
        )

        # 5. Select the requested model
        choice_key = model_choice.lower().strip()
        if choice_key == "ensemble" or choice_key not in model_map_by_name:
            selected = ensemble_model_pred
        else:
            selected = model_map_by_name[choice_key]

        # 6. Extract Explainability Signals
        signals = self.extract_signals(text, rating)

        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return PredictResponse(
            final_verdict=ensemble_verdict,
            consensus_confidence=round(ensemble_conf, 4),
            selected_model=selected,
            all_models=all_model_preds,
            signals=signals,
            latency_ms=latency_ms,
        )

    def predict_batch_df(
        self,
        df: pd.DataFrame,
        model_choice: str = "ensemble",
        chunk_size: int = 5000,
    ) -> BatchPredictResponse:
        """
        Process tabular data in memory-safe chunks (max 5,000 rows/batch on 8GB RAM).
        Supports flexible column names: review_text, text, review, content, etc.
        """
        if not self.is_loaded:
            self.load_artifacts()

        t_start = time.perf_counter()

        # Locate review text column
        text_col = None
        for col in df.columns:
            if str(col).lower() in ["review_text", "text", "review", "content", "body", "comment"]:
                text_col = col
                break

        if text_col is None:
            # Fallback to first string-like object column
            for col in df.columns:
                if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                    text_col = col
                    break

        if text_col is None:
            raise ValueError(
                "Could not find a review text column in the uploaded file. "
                "Expected column name: 'review_text', 'text', or 'review'."
            )

        # Locate optional rating column
        rating_col = None
        for col in df.columns:
            if str(col).lower() in ["rating", "stars", "star_rating", "score"]:
                rating_col = col
                break

        # Locate optional user_id and product_id columns
        user_col = None
        for col in df.columns:
            if str(col).lower() in ["user_id", "reviewer_id", "review_id", "author_id"]:
                user_col = col
                break

        prod_col = None
        for col in df.columns:
            if str(col).lower() in ["product_id", "item_id", "business_id", "restaurant_id"]:
                prod_col = col
                break

        total_rows = len(df)
        rows_result: List[BatchRowPrediction] = []
        fake_count = 0
        conf_sum = 0.0

        target_model_name = "XGBoost" if model_choice.lower() == "xgboost" else (
            "LinearSVC" if model_choice.lower() == "linearsvc" else (
                "SGD" if model_choice.lower() == "sgd" else "XGBoost"
            )
        )
        use_ensemble = model_choice.lower() == "ensemble"

        # Process in chunks to prevent memory spikes
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_df = df.iloc[start_idx:end_idx]

            texts = chunk_df[text_col].fillna("").astype(str)
            ratings = (
                pd.to_numeric(chunk_df[rating_col], errors="coerce").fillna(4.0).values
                if rating_col
                else np.full(len(chunk_df), 4.0, dtype=np.float32)
            )

            meta_chunk = pd.DataFrame({
                "rating": ratings,
                "user_id": chunk_df[user_col].values if user_col else [None] * len(chunk_df),
                "product_id": chunk_df[prod_col].values if prod_col else [None] * len(chunk_df),
                "review_date": [None] * len(chunk_df),
            })

            cleaned, aux = self.preprocessor.transform(texts, meta_chunk)
            X_chunk = build_features(
                cleaned,
                aux,
                self.word_vectorizer,
                self.char_vectorizer,
                self.scaler,
                is_train=False,
            )

            # Predict probabilities
            if use_ensemble:
                p_xgb = self.models["XGBoost"].predict_proba(X_chunk)[:, 1]
                p_svc = self.models["LinearSVC"].predict_proba(X_chunk)[:, 1]
                p_sgd = self.models["SGD"].predict_proba(X_chunk)[:, 1]

                v_xgb = (p_xgb >= self.thresholds.get("XGBoost", 0.64)).astype(int)
                v_svc = (p_svc >= self.thresholds.get("LinearSVC", 0.63)).astype(int)
                v_sgd = (p_sgd >= self.thresholds.get("SGD", 0.73)).astype(int)

                votes = v_xgb + v_svc + v_sgd
                is_spams = votes >= 2
                probs = (p_xgb + p_svc + p_sgd) / 3.0
            else:
                model = self.models.get(target_model_name, self.models["XGBoost"])
                thresh = self.thresholds.get(target_model_name, 0.50)
                probs = model.predict_proba(X_chunk)[:, 1]
                is_spams = probs >= thresh

            for i, (is_sp, pr, raw_text, r_val) in enumerate(
                zip(is_spams, probs, texts, ratings)
            ):
                global_idx = start_idx + i
                conf = float(pr if is_sp else (1.0 - pr))
                conf_sum += conf
                if is_sp:
                    fake_count += 1

                # Quick signal summary
                sigs = []
                if raw_text.count("!") >= 3:
                    sigs.append("Multiple '!'")
                if float(r_val) in (1.0, 5.0):
                    sigs.append("Extreme rating")

                rows_result.append(
                    BatchRowPrediction(
                        row_index=global_idx + 1,
                        review_text=raw_text if len(raw_text) <= 120 else raw_text[:117] + "...",
                        rating=round(float(r_val), 1) if rating_col else None,
                        prediction="Fake" if is_sp else "Genuine",
                        is_spam=bool(is_sp),
                        spam_probability=round(float(pr), 4),
                        confidence=round(conf, 4),
                        key_signals=sigs,
                    )
                )

        genuine_count = total_rows - fake_count
        fake_pct = round((fake_count / total_rows * 100.0), 2) if total_rows > 0 else 0.0
        avg_conf = round((conf_sum / total_rows), 4) if total_rows > 0 else 0.0

        summary = BatchSummary(
            total=total_rows,
            fake_count=fake_count,
            genuine_count=genuine_count,
            fake_percentage=fake_pct,
            average_confidence=avg_conf,
        )

        # Store CSV export representation in memory for one-click download
        export_id = str(uuid.uuid4())
        export_df = pd.DataFrame([
            {
                "Row": r.row_index,
                "Review Text": r.review_text,
                "Rating": r.rating,
                "Verdict": r.prediction,
                "Is Spam": r.is_spam,
                "Spam Probability": r.spam_probability,
                "Confidence": r.confidence,
                "Risk Signals": "; ".join(r.key_signals),
            }
            for r in rows_result
        ])
        self.export_cache[export_id] = export_df.to_csv(index=False)

        latency = round((time.perf_counter() - t_start) * 1000, 2)

        return BatchPredictResponse(
            summary=summary,
            rows=rows_result,
            model_used=model_choice.capitalize(),
            latency_ms=latency,
            export_id=export_id,
        )

    def get_export_csv(self, export_id: str) -> Optional[str]:
        """Retrieve cached CSV export string by ID."""
        return self.export_cache.get(export_id)


# Global MLService singleton instance
ml_service = MLService()
