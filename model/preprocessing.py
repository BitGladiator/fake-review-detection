"""
preprocessing.py — Text preprocessing pipeline for Fake Review Detection.

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
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)




def _nltk_download(resource: str, kind: str = "corpora") -> None:
    """Download an NLTK resource quietly, ignoring errors if already present."""
    try:
        nltk.download(resource, quiet=True)
    except Exception as exc: 
        logger.warning("NLTK download failed for '%s': %s", resource, exc)


_nltk_download("stopwords", kind="corpora")
_nltk_download("punkt", kind="tokenizers")
_nltk_download("punkt_tab", kind="tokenizers")

try:
    _STOPWORDS = set(stopwords.words("english"))
except Exception:  
    logger.warning("Could not load NLTK stopwords; using empty set.")
    _STOPWORDS = set()





class TextPreprocessor:
    """
    Cleans raw review text and extracts lightweight auxiliary features.

    Parameters
    ----------
    stemming : bool
        Apply Porter stemming after tokenisation.
    remove_stopwords : bool
        Remove English stop-words during cleaning.
    """

    def __init__(self, stemming: bool = True, remove_stopwords: bool = True) -> None:
        self.stemming = stemming
        self.remove_stopwords = remove_stopwords
        self._stemmer = PorterStemmer() if stemming else None





    def _clean_text(self, text: str) -> str:
        """
        Return a cleaned version of *text*.

        Steps (in order):
          1. Lower-case
          2. Remove URLs  (http / https / www)
          3. Remove HTML tags
          4. Remove e-mail addresses
          5. Keep exclamation marks  (they are fraud signals)
          6. Remove all other punctuation
          7. Optionally remove stop-words and/or apply stemming
          8. Collapse extra whitespace
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""


        text = text.lower()


        text = re.sub(r"https?://\S+|www\.\S+", " ", text)


        text = re.sub(r"<[^>]+>", " ", text)


        text = re.sub(r"\S+@\S+\.\S+", " ", text)



        punct_no_excl = string.punctuation.replace("!", "")
        translator = str.maketrans(punct_no_excl, " " * len(punct_no_excl))
        text = text.translate(translator)


        tokens = text.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in _STOPWORDS]
        if self.stemming and self._stemmer is not None:
            tokens = [self._stemmer.stem(t) for t in tokens]


        text = " ".join(tokens)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_auxiliary_features(
        self,
        texts: pd.Series,
        ratings: pd.Series = None,
    ) -> np.ndarray:
        """
        Build a (n_samples, 5) float32 numpy array from *raw* texts.

        Returned as a plain ndarray (only 5 columns — negligible memory).
        ``build_features`` handles scaling and sparse conversion so that
        .toarray() / .todense() are never called on any large matrix.

        Features
        --------
        0  char_length       — total character count
        1  word_count        — whitespace-delimited token count
        2  exclamation_count — number of '!' characters
        3  uppercase_ratio   — fraction of alphabetic chars that are uppercase
        4  rating            — star rating (0.0 if not provided)
        """
        char_lengths = texts.str.len().fillna(0).values.astype(np.float32)
        word_counts = texts.str.split().str.len().fillna(0).values.astype(np.float32)
        excl_counts = texts.str.count(r"!").fillna(0).values.astype(np.float32)

        def _uppercase_ratio(s: str) -> float:
            if not isinstance(s, str) or len(s) == 0:
                return 0.0
            alpha = [c for c in s if c.isalpha()]
            if not alpha:
                return 0.0
            return sum(1 for c in alpha if c.isupper()) / len(alpha)

        upper_ratios = np.array(
            [_uppercase_ratio(t) for t in texts], dtype=np.float32
        )

        if ratings is not None:
            rating_vals = ratings.fillna(0.0).values.astype(np.float32)
        else:
            rating_vals = np.zeros(len(texts), dtype=np.float32)

      
        return np.column_stack(
            [char_lengths, word_counts, excl_counts, upper_ratios, rating_vals]
        ).astype(np.float32)  

   

    def fit_transform(
        self,
        texts: pd.Series,
        ratings: pd.Series = None,
    ) -> tuple:
        """
        Fit (stateless here) and transform raw texts.

        Returns
        -------
        cleaned_texts : pd.Series
        aux_features  : np.ndarray, shape (n, 5), dtype float32
        """
        # Auxiliary features MUST be computed on the ORIGINAL texts
        aux_features = self._extract_auxiliary_features(texts, ratings)

        logger.info("Cleaning %d texts…", len(texts))
        t0 = time.time()
        cleaned_texts = texts.apply(self._clean_text)
        logger.info("Text cleaning done in %.1fs", time.time() - t0)

        gc.collect()
        return cleaned_texts, aux_features

    def transform(
        self,
        texts: pd.Series,
        ratings: pd.Series = None,
    ) -> tuple:
        """
        Transform raw texts (stateless — identical to fit_transform).

        Returns
        -------
        cleaned_texts : pd.Series
        aux_features  : np.ndarray, shape (n, 5), dtype float32
        """
        return self.fit_transform(texts, ratings)




def build_features(
    cleaned_texts,
    aux_features: np.ndarray,
    vectorizer: TfidfVectorizer,
    scaler: StandardScaler = None,
    is_train: bool = True,
) -> scipy.sparse.csr_matrix:
    """
    Combine TF-IDF and auxiliary features into a single sparse CSR matrix.

    Parameters
    ----------
    cleaned_texts : array-like of str
        Already-cleaned review texts.
    aux_features : np.ndarray, shape (n, 5)
        Dense auxiliary features from TextPreprocessor._extract_auxiliary_features.
        Kept as a plain ndarray (only 5 cols) so StandardScaler can be applied
        directly — no .toarray() / .todense() calls ever needed on large matrices.
    vectorizer : TfidfVectorizer
        Must have dtype=np.float32 to avoid dense conversions.
    scaler : StandardScaler or None
        If provided, scales aux_features (fit on train, transform on both).
    is_train : bool
        True → fit vectorizer (and scaler) first; False → transform only.

    Returns
    -------
    combined : csr_matrix, shape (n, tfidf_features + 5)
        Fully sparse; .toarray() / .todense() are never called.
    """
    if is_train:
        logger.info("Fitting TF-IDF vectorizer on %d documents…", len(cleaned_texts))
        vectorizer.fit(cleaned_texts)

    tfidf_matrix = vectorizer.transform(cleaned_texts)  

   
    if tfidf_matrix.dtype != np.float32:
        tfidf_matrix = tfidf_matrix.astype(np.float32)

   
    if scaler is not None:
        aux_arr = np.asarray(aux_features, dtype=np.float32)  
        if is_train:
            logger.info("Fitting StandardScaler on auxiliary features…")
            scaled_aux = scaler.fit_transform(aux_arr)
        else:
            scaled_aux = scaler.transform(aux_arr)
        sparse_aux = scipy.sparse.csr_matrix(scaled_aux.astype(np.float32))
    else:
        sparse_aux = scipy.sparse.csr_matrix(
            np.asarray(aux_features, dtype=np.float32)
        )

    combined = scipy.sparse.hstack([tfidf_matrix, sparse_aux], format="csr")
    logger.info(
        "Combined feature matrix: shape=%s, nnz=%d", combined.shape, combined.nnz
    )

    gc.collect()
    return combined




def save_preprocessor(
    preprocessor: TextPreprocessor,
    vectorizer: TfidfVectorizer,
    scaler: StandardScaler,
    output_dir: str,
) -> None:
    """
    Persist the preprocessing artefacts to *output_dir*.

    Files written
    -------------
    text_preprocessor.joblib
    vectorizer.joblib
    scaler.joblib
    """
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "text_preprocessor.joblib": preprocessor,
        "vectorizer.joblib": vectorizer,
        "scaler.joblib": scaler,
    }

    for filename, obj in paths.items():
        dest = os.path.join(output_dir, filename)
        joblib.dump(obj, dest, compress=3)
        size_kb = os.path.getsize(dest) / 1024
        print(f"  Saved {filename:35s}  ({size_kb:>8.1f} KB)  →  {dest}")

    logger.info("All preprocessing artefacts saved to '%s'.", output_dir)


def load_preprocessor(artifact_dir: str) -> tuple:
    """
    Load and return (preprocessor, vectorizer, scaler) from *artifact_dir*.
    """
    preprocessor = joblib.load(os.path.join(artifact_dir, "text_preprocessor.joblib"))
    vectorizer = joblib.load(os.path.join(artifact_dir, "vectorizer.joblib"))
    scaler = joblib.load(os.path.join(artifact_dir, "scaler.joblib"))
    logger.info("Preprocessing artefacts loaded from '%s'.", artifact_dir)
    return preprocessor, vectorizer, scaler

if __name__ == "__main__":
    print("=" * 60)
    print("  Fake Review Detection — Preprocessing Demo")
    print("=" * 60)

    sample_texts = pd.Series(
        [
            "This restaurant is AMAZING!!! Best food EVER! Highly recommend!!!",
            "Terrible experience. The staff was rude and food was cold.",
            "Visit http://www.spam-link.com for great deals! <b>Click now</b>",
        ]
    )
    sample_ratings = pd.Series([5.0, 1.0, 5.0])


    preprocessor = TextPreprocessor(stemming=True, remove_stopwords=True)

    
    cleaned, aux = preprocessor.fit_transform(sample_texts, sample_ratings)
    print("\n--- Cleaned texts ---")
    for i, t in enumerate(cleaned):
        print(f"  [{i}] {t!r}")
    print(f"\nAuxiliary features shape : {aux.shape}")
    print(f"Auxiliary features dtype : {aux.dtype}")
    print(f"Is sparse                : {scipy.sparse.issparse(aux)}")

  
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        dtype=np.float32,
    )
    scaler = StandardScaler()

    combined = build_features(
        cleaned,
        aux,
        vectorizer=vectorizer,
        scaler=scaler,
        is_train=True,
    )
    print(f"\nCombined feature matrix shape : {combined.shape}")
    print(f"Combined feature matrix dtype : {combined.dtype}")
    print(f"Is sparse                     : {scipy.sparse.issparse(combined)}")

  
    cleaned_test, aux_test = preprocessor.transform(sample_texts, sample_ratings)
    combined_test = build_features(
        cleaned_test,
        aux_test,
        vectorizer=vectorizer,
        scaler=scaler,
        is_train=False,
    )
    print(f"\nTest combined matrix shape    : {combined_test.shape}")

   
    _demo_dir = "/tmp/fake_review_preproc_demo"
    save_preprocessor(preprocessor, vectorizer, scaler, _demo_dir)
    p2, v2, s2 = load_preprocessor(_demo_dir)
    print(f"\nLoaded preprocessor type : {type(p2).__name__}")
    print(f"Loaded vectorizer type   : {type(v2).__name__}")
    print(f"Loaded scaler type       : {type(s2).__name__}")

    print("\n Preprocessing demo complete.")
