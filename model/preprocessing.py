"""
preprocessing.py — Fake Review Detection
Text cleaning, behavioral feature extraction, TF-IDF build, artifact I/O.
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


for _resource in ("stopwords", "punkt", "punkt_tab", "vader_lexicon"):
    try:
        nltk.download(_resource, quiet=True)
    except Exception as _e:
        logger.warning("NLTK download failed for %s: %s", _resource, _e)

try:
    _STOPWORDS = set(stopwords.words("english"))
except Exception:
    _STOPWORDS = set()



class TextPreprocessor:
    """
    Cleans raw review text and extracts an 11-column dense auxiliary feature
    array.  Behavioral aggregates (product means, burst counts) are fitted on
    training data only via fit_transform(); transform() re-uses them.
    """

    def __init__(self, stemming: bool = True, remove_stopwords: bool = True):
        self.stemming = stemming
        self.remove_stopwords = remove_stopwords
        self._stemmer = PorterStemmer() if stemming else None
        self._sia = SentimentIntensityAnalyzer()


        self.product_mean_rating_: dict = {}
        self.product_review_count_: dict = {}
        self.daily_burst_count_: dict = {}
        self.global_mean_rating_: float = 3.0
        self.global_review_count_: float = 0.0
        self.global_burst_count_: float = 0.0


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
        """Fit lookup dicts from TRAINING data only."""
        if df is None:
            return

        has_pid = "product_id" in df.columns
        has_rating = "rating" in df.columns
        has_date = "review_date" in df.columns

        if has_rating:
            self.global_mean_rating_ = float(
                pd.to_numeric(df["rating"], errors="coerce").fillna(3.0).mean()
            )

        if has_pid and has_rating:
            grp = df.groupby("product_id")
            self.product_mean_rating_ = grp.apply(
                lambda g: pd.to_numeric(g["rating"], errors="coerce").fillna(3.0).mean()
            ).to_dict()
            counts = grp.size()
            self.product_review_count_ = np.log1p(counts).to_dict()
            self.global_review_count_ = float(np.log1p(counts.mean()))

        if has_pid and has_date:
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


    def _extract_behavioral_features(
        self, texts: pd.Series, df=None
    ) -> np.ndarray:
        """
        Returns np.ndarray shape (n, 11) float32.
        ALWAYS dense — only 11 columns, StandardScaler works directly on it.
        """
        n = len(texts)


        char_len = texts.str.len().fillna(0).values.astype(np.float32)
        word_cnt = texts.str.split().str.len().fillna(0).values.astype(np.float32)
        excl_cnt = texts.str.count("!").fillna(0).values.astype(np.float32)

        _FP = {"i", "me", "my", "mine", "myself"}

        def _upr(s: str) -> float:
            if not isinstance(s, str) or not s:
                return 0.0
            alpha = [c for c in s if c.isalpha()]
            return sum(1 for c in alpha if c.isupper()) / max(len(alpha), 1)

        def _fpr(s: str) -> float:
            if not isinstance(s, str) or not s:
                return 0.0
            w = s.lower().split()
            return sum(1 for x in w if x in _FP) / max(len(w), 1)

        upr = np.array([_upr(t) for t in texts], dtype=np.float32)
        fpr = np.array([_fpr(t) for t in texts], dtype=np.float32)


        if df is not None and "rating" in df.columns:
            rating = (
                pd.to_numeric(df["rating"], errors="coerce")
                .fillna(3.0)
                .values.astype(np.float32)
            )
        else:
            rating = np.full(n, 3.0, dtype=np.float32)


        if df is not None and "product_id" in df.columns:
            pids = df["product_id"].values
            prod_means = np.array(
                [
                    self.product_mean_rating_.get(p, self.global_mean_rating_)
                    for p in pids
                ],
                dtype=np.float32,
            )
            rat_dev = np.abs(rating - prod_means)
            prod_cnt = np.array(
                [
                    self.product_review_count_.get(p, self.global_review_count_)
                    for p in pids
                ],
                dtype=np.float32,
            )
        else:
            rat_dev = np.zeros(n, dtype=np.float32)
            prod_cnt = np.full(n, self.global_review_count_, dtype=np.float32)

        extreme = ((rating == 1.0) | (rating == 5.0)).astype(np.float32)


        if (
            df is not None
            and "product_id" in df.columns
            and "review_date" in df.columns
        ):
            date_strs = (
                pd.to_datetime(df["review_date"], errors="coerce")
                .dt.strftime("%Y-%m-%d")
                .fillna("unknown")
                .values
            )
            pids_b = df["product_id"].values
            burst = np.array(
                [
                    self.daily_burst_count_.get(
                        str((p, d)), self.global_burst_count_
                    )
                    for p, d in zip(pids_b, date_strs)
                ],
                dtype=np.float32,
            )
        else:
            burst = np.full(n, self.global_burst_count_, dtype=np.float32)


        def _sent_mismatch(text, r: float) -> float:
            try:
                compound = self._sia.polarity_scores(str(text))["compound"]
                scaled = (compound + 1) / 2 * 4 + 1  # [1, 5]
                return float(abs(scaled - r))
            except Exception:
                return 0.0

        sent = np.array(
            [_sent_mismatch(t, r) for t, r in zip(texts, rating)],
            dtype=np.float32,
        )


        return np.column_stack(
            [char_len, word_cnt, excl_cnt, upr, fpr, rating,
             rat_dev, extreme, prod_cnt, burst, sent]
        ).astype(np.float32)


    def fit_transform(self, texts: pd.Series, df=None):
        """
        Fit behavioral aggregates on training data, then clean + extract.

        Returns
        -------
        cleaned : pd.Series
        aux     : np.ndarray (n, 11)
        """
        if df is not None:
            self._fit_behavioral_aggregates(df)
        aux = self._extract_behavioral_features(texts, df)
        cleaned = texts.apply(self._clean_text)
        gc.collect()
        return cleaned, aux

    def transform(self, texts: pd.Series, df=None):
        """
        Apply pre-fitted aggregates to new data.

        Returns
        -------
        cleaned : pd.Series
        aux     : np.ndarray (n, 11)
        """
        aux = self._extract_behavioral_features(texts, df)
        cleaned = texts.apply(self._clean_text)
        gc.collect()
        return cleaned, aux



def build_features(
    cleaned_texts,
    aux_features,
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
    scaler: StandardScaler,
    is_train: bool = True,
) -> scipy.sparse.csr_matrix:
    """
    Combine word TF-IDF + char TF-IDF + scaled auxiliary features into one
    sparse CSR matrix.

    Parameters
    ----------
    cleaned_texts   : pd.Series of cleaned strings
    aux_features    : np.ndarray (n, 11) — dense, StandardScaler works directly
    word_vectorizer : TfidfVectorizer (word n-grams)
    char_vectorizer : TfidfVectorizer (char n-grams)
    scaler          : StandardScaler for the aux array
    is_train        : if True, fit vectorizers + scaler; else only transform

    Returns
    -------
    scipy.sparse.csr_matrix, float32
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
    logger.info(
        "Combined feature matrix: shape=%s, nnz=%d", combined.shape, combined.nnz
    )
    gc.collect()
    return combined



def save_preprocessor(
    preprocessor: TextPreprocessor,
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
    scaler: StandardScaler,
    output_dir: str,
) -> None:
    """Persist all preprocessing artifacts to *output_dir*."""
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
    """
    Load all preprocessing artifacts from *artifact_dir*.

    Returns
    -------
    (preprocessor, word_vectorizer, char_vectorizer, scaler)
    """
    preprocessor = joblib.load(
        os.path.join(artifact_dir, "text_preprocessor.joblib")
    )
    word_vectorizer = joblib.load(
        os.path.join(artifact_dir, "word_vectorizer.joblib")
    )
    char_vectorizer = joblib.load(
        os.path.join(artifact_dir, "char_vectorizer.joblib")
    )
    scaler = joblib.load(os.path.join(artifact_dir, "scaler.joblib"))
    return preprocessor, word_vectorizer, char_vectorizer, scaler



if __name__ == "__main__":
    print("=== Preprocessing Demo ===")

    sample_texts = pd.Series(
        [
            "AMAZING food! Best place EVER!!! Must visit!!!",
            "Decent place, nothing special. Food was ok.",
            "Visit http://spam.com for deals! <b>Click now</b>",
        ]
    )
    sample_df = pd.DataFrame(
        {
            "rating": [5.0, 3.0, 5.0],
            "product_id": [1, 2, 1],
            "review_date": ["2020-01-01", "2020-01-01", "2020-01-01"],
        }
    )

    preprocessor = TextPreprocessor()
    cleaned, aux = preprocessor.fit_transform(sample_texts, sample_df)
    print("Cleaned:", cleaned.tolist())
    print("Aux shape:", aux.shape)


    word_vec = TfidfVectorizer(max_features=100, ngram_range=(1, 2), dtype=np.float32)
    char_vec = TfidfVectorizer(
        max_features=100, analyzer="char_wb", ngram_range=(3, 4), dtype=np.float32
    )
    scaler_obj = StandardScaler()
    X = build_features(cleaned, aux, word_vec, char_vec, scaler_obj, is_train=True)
    print("Feature matrix shape:", X.shape, "| type:", type(X).__name__)
    print("Demo complete.")
