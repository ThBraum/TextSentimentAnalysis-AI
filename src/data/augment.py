"""Aumenta o dataset processado de sentimento com operações simples ao nível de tokens.

O script lê `data/processed/sentiment_data.csv`, aplica aumentos simples
(`delete` aleatório, `swap`, `insertion`) em `clean_text` e grava um dataset
aumentado em `data/processed/sentiment_data_augmented.csv`.

Uso:
    python -m src.data.augment --factor 0.5
"""

from pathlib import Path
import random
import argparse
import copy
from typing import List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed" / "sentiment_data.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "sentiment_data_augmented.csv"


def random_deletion(words: List[str], p: float = 0.1) -> List[str]:
    if len(words) == 1:
        return words
    remaining = [w for w in words if random.random() > p]
    if not remaining:
        return [random.choice(words)]
    return remaining


def random_swap(words: List[str], n_swaps: int = 1) -> List[str]:
    words = words.copy()
    for _ in range(n_swaps):
        i = random.randrange(len(words))
        j = random.randrange(len(words))
        words[i], words[j] = words[j], words[i]
    return words


def random_insertion(words: List[str], n_ins: int = 1) -> List[str]:
    words = words.copy()
    for _ in range(n_ins):
        idx = random.randrange(len(words))
        # insere uma duplicata de uma palavra aleatória existente (proxy simples para sinônimo)
        words.insert(idx, random.choice(words))
    return words


def augment_text(text: str) -> str:
    words = text.split()
    strategy = random.choice(["del", "swap", "ins"])
    if strategy == "del":
        new = random_deletion(words, p=0.12)
    elif strategy == "swap":
        new = random_swap(words, n_swaps=max(1, int(len(words) * 0.05)))
    else:
        new = random_insertion(words, n_ins=max(1, int(len(words) * 0.05)))
    return " ".join(new)


def augment_dataframe(df: pd.DataFrame, factor: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """Retorna o dataframe aumentado com aproximadamente (1+factor) * len(df) linhas.

    O aumento é proporcional por classe para preservar o balanceamento de rótulos.
    """
    random.seed(seed)
    df = df.copy()
    n_original = len(df)
    n_target = int(n_original * (1.0 + factor))
    n_to_generate = n_target - n_original

    if n_to_generate <= 0:
        return df

    per_class = df.groupby("sentiment").size().to_dict()
    gen_rows = []
    for label, count in per_class.items():
        # allocate generated samples proportional to class size
        k = int(round(n_to_generate * (count / n_original)))
        # sample with replacement from class
        class_rows = df[df["sentiment"] == label]
        for _ in range(k):
            row = class_rows.sample(n=1, replace=True).iloc[0]
            new_row = row.copy()
            new_row["clean_text"] = augment_text(row["clean_text"])
            gen_rows.append(new_row)

    # if rounding left some gap, fill from random classes
    while len(gen_rows) < n_to_generate:
        row = df.sample(n=1, replace=True).iloc[0]
        new_row = row.copy()
        new_row["clean_text"] = augment_text(row["clean_text"])
        gen_rows.append(new_row)

    aug_df = pd.concat([df, pd.DataFrame(gen_rows)], ignore_index=True)
    return aug_df


def main():
    parser = argparse.ArgumentParser(description="Aumenta o dataset processado aplicando pequenas alterações nos textos.")
    parser.add_argument("--factor", type=float, default=0.5, help="Fator de aumento (ex.: 0.5 adiciona 50% de amostras a mais)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not PROCESSED.exists():
        raise SystemExit(f"Dataset processado não encontrado em {PROCESSED}. Rode o pré-processamento primeiro.")

    df = pd.read_csv(PROCESSED)
    aug = augment_dataframe(df, factor=args.factor, seed=args.seed)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    aug.to_csv(OUT_PATH, index=False)
    print(f"Gravou dataset aumentado em {OUT_PATH} (linhas: {len(aug)})")


if __name__ == "__main__":
    main()
