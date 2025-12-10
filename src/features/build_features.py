"""Utilitários de engenharia de atributos (TF-IDF) para modelos de sentimento."""

from pathlib import Path
from typing import Iterable, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"


def fit_vectorizer(
    texts: Iterable[str],
    max_features: int = 20000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
) -> Tuple[TfidfVectorizer, any]:
    """Ajusta um `TfidfVectorizer` e retorna o vetor junto com a matriz transformada."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def transform_texts(vectorizer: TfidfVectorizer, texts: Iterable[str]):
    """Transforma textos usando um vectorizer já ajustado."""
    return vectorizer.transform(texts)


def persist_vectorizer(vectorizer: TfidfVectorizer, filename: str = "tfidf_vectorizer.pkl") -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / filename
    joblib.dump(vectorizer, path)
    return path


def load_vectorizer(path: Path | str = MODEL_DIR / "tfidf_vectorizer.pkl") -> TfidfVectorizer:
    return joblib.load(path)


def build_train_test_features(
    train_texts: Iterable[str],
    test_texts: Iterable[str],
    **vectorizer_kwargs,
):
    """Fit vectorizer on train set and transform both splits."""
    vectorizer, X_train = fit_vectorizer(train_texts, **vectorizer_kwargs)
    X_test = transform_texts(vectorizer, test_texts)
    return vectorizer, X_train, X_test
