"""Limpa texto bruto e normaliza rótulos de sentimento antes da extração de características."""

from pathlib import Path
from typing import Optional

import pandas as pd
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_PATH = PROJECT_ROOT / "data" / "interim" / "combined_data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def clean_text(text: str) -> str:
    """Converte para minúsculas, remove URLs, pontuação e colapsa espaços em branco."""
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def normalize_sentiment(value) -> Optional[int]:
    """Mapeia formatos variados de rótulo para o binário {0,1} de sentimento."""
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    mapping = {
        "0": 0,
        "4": 1,  # Sentiment140 positive
        "neg": 0,
        "negativo": 0,
        "negative": 0,
        "pos": 1,
        "positivo": 1,
        "positive": 1,
        "1": 1,
        "-1": 0,
    }
    # handle explicit mapping first
    if s in mapping:
        return mapping[s]

    # handle Sentiment140 style numeric labels '0' and '4' already covered,
    # handle amazon-style labels like '__label__1' or '__label__2'
    if s.startswith("__label__"):
        try:
            lbl = int(s.split("__label__")[-1])
            # common convention in some amazon datasets: 1=negative, 2=positive
            return 1 if lbl == 2 else 0
        except Exception:
            return None

    # fallback: try numeric conversion
    try:
        return 1 if float(s) > 0 else 0
    except ValueError:
        return None


def preprocess_data(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    if "text" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("O CSV de entrada deve conter as colunas 'text' e 'sentiment'.")

    df = df.dropna(subset=["text", "sentiment"])
    df["text"] = df["text"].astype(str)
    df["sentiment"] = df["sentiment"].apply(normalize_sentiment)
    df = df.dropna(subset=["sentiment"])
    df["clean_text"] = df["text"].apply(clean_text)

    keep_cols = [col for col in ["clean_text", "sentiment", "source"] if col in df.columns]
    return df[keep_cols]


def save_processed(df: pd.DataFrame, filename: str = "sentiment_data.csv") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / filename
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    processed = preprocess_data(INTERIM_PATH)
    output_path = save_processed(processed)
    print(f"Conjunto processado gravado em {output_path}")


if __name__ == "__main__":
    main()
