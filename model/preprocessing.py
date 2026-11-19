"""
preprocessing.py — Fake Review Detection
Text cleaning, reviewer & behavioral feature extraction, dual TF-IDF build, artifact I/O.
Optimized for Apple M2 8GB RAM:
- All large feature matrices stay strictly sparse float32 (scipy.sparse.csr_matrix)
- Auxiliary behavioral feature matrix is small (12 columns) and converted to sparse after StandardScaler
- Zero calls to .toarray() / .todense() on TF-IDF or combined matrices
"""

import re
import string
import gc
import time
import logging
import os

import numpy as np
import pandas as pd
import scipy.sparse
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


for _res in ("stopwords", "punkt", "punkt_tab", "vader_lexicon"):
    try:
        nltk.download(_res, quiet=True)
    except Exception as _e:
        logger.warning("NLTK download failed for %s: %s", _res, _e)

_NLTK_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing",
    "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't",
    "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its",
    "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't",
    "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't",
    "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than",
    "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't",
    "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves"
}

try:
    _STOPWORDS = set(stopwords.words("english"))
    if not _STOPWORDS:
        _STOPWORDS = _NLTK_STOPWORDS
except Exception:
    _STOPWORDS = _NLTK_STOPWORDS



class TextPreprocessor:
    """
    Cleans raw review text and extracts a 13-column dense auxiliary behavioral
    feature array.
    
    Leakage-Free Behavioral Aggregates (fitted on train split only):
    1. user_review_count       : log1p(count) for reviewer user_id
    2. user_daily_burst        : count of reviews by (user_id, review_date)
    3. product_mean_rating     : mean star rating for product_id
    4. product_review_count    : log1p(count) for product_id
    5. daily_burst_count       : log1p(count) for (product_id, review_date)
    6. rating_deviation        : |rating - product_mean_rating|
    7. extreme_rating          : 1 if rating is 1 or 5, else 0
    8. sentiment_rating_mismatch: |vader_rescaled_[1,5] - rating|
    9. char_len                : review length in characters
    10. word_cnt               : token count
    11. exclamation_count      : count of '!'
    12. uppercase_ratio        : uppercase letters / total alphabetic chars
    13. first_person_ratio     : (i, me, my, mine, myself) / total words
    """

    def __init__(self, stemming: bool = True, remove_stopwords: bool = True):
        self.stemming = stemming
        self.remove_stopwords = remove_stopwords
        self._stemmer = PorterStemmer() if stemming else None


        try:
            self._sia = SentimentIntensityAnalyzer()
        except Exception:
            logger.warning("VADER lexicon not found in local environment. Fallback sentiment will be used.")
            self._sia = None

        self.user_review_count_: dict = {}
        self.user_daily_burst_: dict = {}
        self.product_mean_rating_: dict = {}
        self.product_review_count_: dict = {}
        self.daily_burst_count_: dict = {}

        self.global_user_review_count_: float = float(np.log1p(1))
        self.global_user_daily_burst_: float = 1.0
        self.global_mean_rating_: float = 4.0
        self.global_product_review_count_: float = float(np.log1p(1))
        self.global_burst_count_: float = 0.0

    @property
    def median_user_reviews(self) -> float:
        """Safe review-level median user count: np.log1p(3.0) ~ 1.3863."""
        return getattr(self, "_median_user_reviews", float(np.log1p(3.0)))

    @property
    def median_product_reviews(self) -> float:
        """Safe product review count median from training data."""
        return getattr(self, "_median_product_reviews", getattr(self, "global_product_review_count_", float(np.log1p(175.0))))

    @property
    def global_mean_rating(self) -> float:
        """Safe neutral average star rating from training data."""
        return getattr(self, "global_mean_rating_", 4.0)


    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text) if text else ""
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\S+@\S+\.\S+", " ", text)
        

        punct_no_excl = string.punctuation.replace("!", "")
        text = text.translate(str.maketrans(punct_no_excl, " " * len(punct_no_excl)))
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in _STOPWORDS]
        if self.stemming and self._stemmer:
            tokens = [self._stemmer.stem(t) for t in tokens]
        return re.sub(r"\s+", " ", " ".join(tokens)).strip()


    def _fit_behavioral_aggregates(self, df: pd.DataFrame) -> None:
        """Fit lookup dictionaries strictly on training data."""
        if df is None:
            return

        has_user = "user_id" in df.columns or "review_id" in df.columns
        has_prod = "product_id" in df.columns
        has_rating = "rating" in df.columns
        has_date = "review_date" in df.columns


        user_col = "user_id" if "user_id" in df.columns else "review_id"
        if has_user:
            u_counts = df[user_col].value_counts()
            self.user_review_count_ = np.log1p(u_counts).to_dict()
            self.global_user_review_count_ = float(np.log1p(1))

            if has_date:
                u_date_str = (
                    pd.to_datetime(df["review_date"], errors="coerce")
                    .dt.strftime("%Y-%m-%d")
                    .fillna("unknown")
                )
                u_burst = df.groupby([df[user_col], u_date_str]).size()
                self.user_daily_burst_ = {
                    str(k): float(v) for k, v in u_burst.items()
                }
                self.global_user_daily_burst_ = 1.0


        if has_rating:
            self.global_mean_rating_ = float(
                pd.to_numeric(df["rating"], errors="coerce").fillna(3.0).mean()
            )

        if has_prod and has_rating:
            grp_p = df.groupby("product_id")
            self.product_mean_rating_ = grp_p["rating"].mean().to_dict()
            p_counts = grp_p.size()
            self.product_review_count_ = np.log1p(p_counts).to_dict()
            self.global_product_review_count_ = float(np.log1p(max(p_counts.median(), 1)))


        if has_prod and has_date:
            date_str = (
                pd.to_datetime(df["review_date"], errors="coerce")
                .dt.strftime("%Y-%m-%d")
                .fillna("unknown")
            )
            burst = df.groupby(["product_id", date_str]).size()
            self.daily_burst_count_ = {
                str(k): float(np.log1p(v)) for k, v in burst.items()
            }
            self.global_burst_count_ = (
                float(np.log1p(burst.mean())) if len(burst) > 0 else 0.0
            )


    def _extract_behavioral_features(self, texts: pd.Series, df=None) -> np.ndarray:
        n = len(texts)


        char_len = texts.str.len().fillna(0).values.astype(np.float32)
        word_cnt = texts.str.split().str.len().fillna(0).values.astype(np.float32)
        excl_cnt = texts.str.count("!").fillna(0).values.astype(np.float32)

        _FIRST_PERSON = {"i", "me", "my", "mine", "myself"}

        def _upr(s: str) -> float:
            if not isinstance(s, str) or not s:
                return 0.0
            alpha = [c for c in s if c.isalpha()]
            return sum(1 for c in alpha if c.isupper()) / max(len(alpha), 1)

        def _fpr(s: str) -> float:
            if not isinstance(s, str) or not s:
                return 0.0
            w = s.lower().split()
            return sum(1 for x in w if x in _FIRST_PERSON) / max(len(w), 1)

        upr = np.array([_upr(t) for t in texts], dtype=np.float32)
        fpr = np.array([_fpr(t) for t in texts], dtype=np.float32)



        if df is not None and "rating" in df.columns:
            rating = (
                pd.to_numeric(df["rating"], errors="coerce")
                .fillna(self.global_mean_rating)
                .values.astype(np.float32)
            )
        else:
            rating = np.full(n, self.global_mean_rating, dtype=np.float32)

        extreme = ((rating == 1.0) | (rating == 5.0)).astype(np.float32)


        user_col = None
        if df is not None:
            if "user_id" in df.columns:
                user_col = "user_id"
            elif "review_id" in df.columns:
                user_col = "review_id"

        if user_col is not None:
            uids = df[user_col].values
            user_cnt = np.array(
                [
                    self.user_review_count_.get(u, self.global_user_review_count_)
                    if (u is not None and not (isinstance(u, float) and np.isnan(u)))
                    else self.median_user_reviews
                    for u in uids
                ],
                dtype=np.float32,
            )

            if "review_date" in df.columns:
                u_date_strs = (
                    pd.to_datetime(df["review_date"], errors="coerce")
                    .dt.strftime("%Y-%m-%d")
                    .fillna("unknown")
                    .values
                )
                user_burst = np.array(
                    [
                        self.user_daily_burst_.get(str((u, d)), self.global_user_daily_burst_)
                        if (u is not None and not (isinstance(u, float) and np.isnan(u)))
                        else self.global_user_daily_burst_
                        for u, d in zip(uids, u_date_strs)
                    ],
                    dtype=np.float32,
                )
            else:
                user_burst = np.full(n, self.global_user_daily_burst_, dtype=np.float32)
        else:
            user_cnt = np.full(n, self.median_user_reviews, dtype=np.float32)
            user_burst = np.full(n, self.global_user_daily_burst_, dtype=np.float32)


        if df is not None and "product_id" in df.columns:
            pids = df["product_id"].values
            prod_means = np.array(
                [
                    self.product_mean_rating_.get(p, self.global_mean_rating)
                    if (p is not None and not (isinstance(p, float) and np.isnan(p)))
                    else self.global_mean_rating
                    for p in pids
                ],
                dtype=np.float32,
            )
            rat_dev = np.array(
                [
                    abs(r - pm)
                    if (p is not None and not (isinstance(p, float) and np.isnan(p)))
                    else 0.0
                    for r, pm, p in zip(rating, prod_means, pids)
                ],
                dtype=np.float32,
            )
            prod_cnt = np.array(
                [
                    self.product_review_count_.get(p, self.median_product_reviews)
                    if (p is not None and not (isinstance(p, float) and np.isnan(p)))
                    else self.median_product_reviews
                    for p in pids
                ],
                dtype=np.float32,
            )
        else:
            rat_dev = np.zeros(n, dtype=np.float32)
            prod_cnt = np.full(n, self.median_product_reviews, dtype=np.float32)


        if df is not None and "product_id" in df.columns and "review_date" in df.columns:
            date_strs = (
                pd.to_datetime(df["review_date"], errors="coerce")
                .dt.strftime("%Y-%m-%d")
                .fillna("unknown")
                .values
            )
            pids_b = df["product_id"].values
            burst = np.array(
                [
                    self.daily_burst_count_.get(str((p, d)), self.global_burst_count_)
                    if (p is not None and not (isinstance(p, float) and np.isnan(p)))
                    else self.global_burst_count_
                    for p, d in zip(pids_b, date_strs)
                ],
                dtype=np.float32,
            )
        else:
            burst = np.full(n, self.global_burst_count_, dtype=np.float32)


        def _sent_mismatch(text, r: float) -> float:
            if self._sia is not None:
                try:
                    compound = self._sia.polarity_scores(str(text))["compound"]
                    scaled = (compound + 1.0) / 2.0 * 4.0 + 1.0  
                    return float(abs(scaled - r))
                except Exception:
                    pass

            pos_words = ("great", "amazing", "excellent", "best", "good", "love", "perfect", "delicious", "recommend")
            neg_words = ("terrible", "worst", "horrible", "bad", "awful", "rude", "dirty", "poor", "slow", "waste")
            t_lower = str(text).lower()
            pos = sum(1 for w in pos_words if w in t_lower)
            neg = sum(1 for w in neg_words if w in t_lower)
            comp = 1.0 if pos > neg else (-1.0 if neg > pos else 0.0)
            scaled = (comp + 1.0) / 2.0 * 4.0 + 1.0
            return float(abs(scaled - r))

        sent = np.array(
            [_sent_mismatch(t, r) for t, r in zip(texts, rating)],
            dtype=np.float32
        )


        return np.column_stack([
            char_len, word_cnt, excl_cnt, upr, fpr, rating,
            user_cnt, user_burst, rat_dev, extreme, prod_cnt, burst, sent
        ]).astype(np.float32)




    def fit_transform(self, texts: pd.Series, df=None):
        """Fit behavioral aggregates on training split, then extract features & clean texts."""
        if df is not None:
            self._fit_behavioral_aggregates(df)
        aux = self._extract_behavioral_features(texts, df)
        cleaned = texts.apply(self._clean_text)
        gc.collect()
        return cleaned, aux

    def transform(self, texts: pd.Series, df=None):
        """Extract features & clean texts using pre-fitted training aggregates."""
        aux = self._extract_behavioral_features(texts, df)
        cleaned = texts.apply(self._clean_text)
        gc.collect()
        return cleaned, aux





def build_features(
    cleaned_texts,
    aux_features: np.ndarray,
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
    scaler: StandardScaler,
    is_train: bool = True,
) -> scipy.sparse.csr_matrix:
    """
    Combine word TF-IDF + char TF-IDF + scaled auxiliary features into one
    sparse CSR matrix (np.float32).
    
    aux_features is a small (n, 12) dense NumPy array. StandardScaler fits directly
    on it without needing to dense-convert any large TF-IDF matrices.
    """
    if is_train:
        logger.info("Fitting word TF-IDF on %d docs...", len(cleaned_texts))
        word_vectorizer.fit(cleaned_texts)
        logger.info("Fitting char TF-IDF on %d docs...", len(cleaned_texts))
        char_vectorizer.fit(cleaned_texts)

    word_mat = word_vectorizer.transform(cleaned_texts)
    char_mat = char_vectorizer.transform(cleaned_texts)

    if word_mat.dtype != np.float32:
        word_mat = word_mat.astype(np.float32)
    if char_mat.dtype != np.float32:
        char_mat = char_mat.astype(np.float32)
    gc.collect()

    aux_arr = np.asarray(aux_features, dtype=np.float32)
    if is_train:
        scaled_aux = scaler.fit_transform(aux_arr)
    else:
        scaled_aux = scaler.transform(aux_arr)

    sparse_aux = scipy.sparse.csr_matrix(scaled_aux.astype(np.float32))

    combined = scipy.sparse.hstack([word_mat, char_mat, sparse_aux], format="csr")
    logger.info("Combined sparse matrix shape=%s, nnz=%d", combined.shape, combined.nnz)
    gc.collect()
    return combined



def save_preprocessor(
    preprocessor: TextPreprocessor,
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
    scaler: StandardScaler,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    artifacts = {
        "text_preprocessor.joblib": preprocessor,
        "word_vectorizer.joblib": word_vectorizer,
        "char_vectorizer.joblib": char_vectorizer,
        "scaler.joblib": scaler,
    }
    for fname, obj in artifacts.items():
        dest = os.path.join(output_dir, fname)
        joblib.dump(obj, dest, compress=3)
        print(f"  Saved {fname} ({os.path.getsize(dest) / 1024:.1f} KB)")


def load_preprocessor(artifact_dir: str):
    preprocessor = joblib.load(os.path.join(artifact_dir, "text_preprocessor.joblib"))
    word_vectorizer = joblib.load(os.path.join(artifact_dir, "word_vectorizer.joblib"))
    char_vectorizer = joblib.load(os.path.join(artifact_dir, "char_vectorizer.joblib"))
    scaler = joblib.load(os.path.join(artifact_dir, "scaler.joblib"))
    return preprocessor, word_vectorizer, char_vectorizer, scaler


_MODEL_CACHE = {}


DEFAULT_MODEL_THRESHOLDS = {
    "XGBoost": 0.64,
    "LinearSVC": 0.63,
    "SGD": 0.73,
}


def predict_single_review(
    text: str,
    rating: float = 4.0,
    user_id=None,
    product_id=None,
    model_name: str = "XGBoost",
    artifact_dir: str = None,
    preprocessor=None,
    word_vectorizer=None,
    char_vectorizer=None,
    scaler=None,
    model=None,
    threshold: float = None,
) -> dict:
    """
    Reusable prediction helper function for single reviews (for FastAPI backend & CLI).
    
    Uses safe median/neutral fallback values when user_id or product_id is None.
    Uses the model's direct predict_proba probability and compares against each
    model's saved validation threshold from metrics.json (0.64 for XGBoost,
    0.63 for LinearSVC, 0.73 for SGD).
    """
    global _MODEL_CACHE

    if artifact_dir is None:
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
        artifact_dir = default_dir if os.path.isdir(default_dir) else "artifacts"

    if threshold is None:
        threshold = DEFAULT_MODEL_THRESHOLDS.get(model_name, 0.50)

    if preprocessor is None or word_vectorizer is None or char_vectorizer is None or scaler is None:
        cache_key_prep = f"prep_{artifact_dir}"
        if cache_key_prep not in _MODEL_CACHE:
            _MODEL_CACHE[cache_key_prep] = load_preprocessor(artifact_dir)
        preprocessor, word_vectorizer, char_vectorizer, scaler = _MODEL_CACHE[cache_key_prep]

    if model is None:
        cache_key_model = f"model_{artifact_dir}_{model_name}"
        if cache_key_model not in _MODEL_CACHE:
            model_file = os.path.join(artifact_dir, f"{model_name}_best.joblib")
            if not os.path.exists(model_file):
                model_file = os.path.join(artifact_dir, f"{model_name}.joblib")
            _MODEL_CACHE[cache_key_model] = joblib.load(model_file)
        model = _MODEL_CACHE[cache_key_model]

    meta_df = pd.DataFrame({
        "rating": [rating if rating is not None else 4.0],
        "user_id": [user_id],
        "product_id": [product_id],
        "review_date": [None],
    })

    series = pd.Series([text])
    cleaned, aux = preprocessor.transform(series, meta_df)
    X = build_features(cleaned, aux, word_vectorizer, char_vectorizer, scaler, is_train=False)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0, 1])
    else:
        prob = float(model.predict(X)[0])

    prediction = 1 if prob >= threshold else 0
    label = "Spam" if prediction == 1 else "Genuine"
    confidence = prob if prediction == 1 else (1.0 - prob)

    return {
        "text": text,
        "prediction": prediction,
        "label": label,
        "is_spam": bool(prediction == 1),
        "probability": prob,
        "probability_raw": prob,
        "confidence": confidence,
        "threshold": threshold,
        "model_name": model_name,
    }
