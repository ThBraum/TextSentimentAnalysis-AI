#!/usr/bin/env python3
"""Grid search rápido para LinearSVC (amostragem + Tfidf).

Usar para encontrar combinações rápidas de `max_features`, `C`, `ngram_range` e `class_weight`.
Exemplo:
    poetry run python scripts/tune_svc.py --sample-size 200000

Observação: roda em CPU por padrão; ajuste `--sample-size` para reduzir tempo.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_data(path: str = "data/processed/sentiment_data.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(p)


def sample_df(df: pd.DataFrame, n: int | None, seed: int = 42) -> pd.DataFrame:
    if n is None or n <= 0 or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def run_grid(df: pd.DataFrame, sample_size: int, out_report: str | None = None):
    df = sample_df(df, sample_size)

    # detectar colunas
    text_col = 'text' if 'text' in df.columns else ('clean_text' if 'clean_text' in df.columns else df.select_dtypes(include=['object']).columns[0])
    label_col = 'sentiment' if 'sentiment' in df.columns else ('label' if 'label' in df.columns else df.columns[1])

    X_raw = df[text_col].fillna("")
    y = df[label_col]

    results = []

    max_features_list = [50000, 80000]
    C_list = [1.0, 1.5]
    ngram_ranges = [(1, 1), (1, 2)]
    class_weights = [None, 'balanced']

    for maxf in max_features_list:
        for ngram in ngram_ranges:
            print(f"Vetorizando com max_features={maxf}, ngram_range={ngram}")
            vec = TfidfVectorizer(max_features=maxf, ngram_range=ngram, stop_words='english')
            X = vec.fit_transform(X_raw)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

            for cw in class_weights:
                for C in C_list:
                    print(f"Treinando LinearSVC (C={C}, class_weight={cw})")
                    t0 = time.time()
                    svc = LinearSVC(C=C, max_iter=10000, dual=False, class_weight=cw)
                    svc.fit(X_train, y_train)
                    y_pred = svc.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    elapsed = time.time() - t0
                    print(f"-> acc={acc:.4f} time={elapsed:.1f}s")
                    results.append({
                        'max_features': maxf,
                        'ngram_range': f"{ngram[0]}-{ngram[1]}",
                        'C': C,
                        'class_weight': str(cw),
                        'accuracy': acc,
                        'time_s': elapsed,
                    })

    rst_df = pd.DataFrame(results).sort_values('accuracy', ascending=False).reset_index(drop=True)
    print('\nMelhores resultados:')
    print(rst_df.head(10).to_string(index=False))
    if out_report:
        rst_df.to_csv(out_report, index=False)
        print(f'Relatório salvo em {out_report}')
    return rst_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=200000, help='Número de exemplos para amostragem (smoke/test); 0 para usar tudo')
    parser.add_argument('--data-path', type=str, default='data/processed/sentiment_data.csv')
    parser.add_argument('--out', type=str, default='reports/tune_svc_results.csv')
    args = parser.parse_args()

    df = load_data(args.data_path)
    sample = args.sample_size if args.sample_size and args.sample_size > 0 else None
    out = args.out
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    run_grid(df, sample, out_report=out)


if __name__ == '__main__':
    main()
