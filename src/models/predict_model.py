"""Carrega um modelo treinado e executa previsões sobre textos brutos."""

from pathlib import Path
from typing import Iterable, List

import joblib

from src.features.build_features import load_vectorizer, transform_texts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "linear_svc_model.pkl"
DEFAULT_VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


def load_model(path: Path | str = DEFAULT_MODEL_PATH):
    return joblib.load(path)


def predict_texts(
    texts: Iterable[str],
    model_path: Path | str = DEFAULT_MODEL_PATH,
    vectorizer_path: Path | str = DEFAULT_VECTORIZER_PATH,
) -> List[int]:
    model = load_model(model_path)
    vectorizer = load_vectorizer(vectorizer_path)
    features = transform_texts(vectorizer, texts)
    return model.predict(features).tolist()


def main():
    sample_texts = [
        "Esse filme foi fantástico, eu adorei!",
        "Experiência terrível, quero meu dinheiro de volta.",
    ]
    preds = predict_texts(sample_texts)
    for text, pred in zip(sample_texts, preds):
        label = "positivo" if pred == 1 else "negativo"
        print(f"{label}: {text}")


if __name__ == "__main__":
    main()
