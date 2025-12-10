#!/usr/bin/env python3
"""Treino rápido/fine-tune de um Transformer (DistilBERT) para classificação de sentimento.

Uso (smoke-test):
    poetry run python scripts/train_transformer.py --sample-size 2000 --epochs 1

O script tenta carregar `data/interim/combined_data.csv` ou `data/processed/sentiment_data.csv`.
Se as dependências (transformers/datasets/torch) não estiverem instaladas, imprime instruções.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

try:
    import pandas as pd
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    import evaluate
except Exception as e:  # pragma: no cover - fail gracefully if deps missing
    print("Dependências ausentes para treino com Transformers:", e)
    print("Instale com: `poetry add transformers datasets torch` e depois rode novamente.")
    sys.exit(1)


def load_data(preferred_paths=None):
    if preferred_paths is None:
        preferred_paths = [
            Path("data/interim/combined_data.csv"),
            Path("data/processed/sentiment_data.csv"),
        ]
    for p in preferred_paths:
        if p.exists():
            df = pd.read_csv(p)
            return df
    raise FileNotFoundError("Nenhum arquivo de dados encontrado em data/interim ou data/processed.")


def prepare_dataset(df: pd.DataFrame, sample_size: int | None = None, random_state: int = 42):
    # Normalize column names (procurar colunas comuns text/sentiment)
    text_col = None
    for candidate in ["text", "review", "review_text", "content"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None and "tweet" in df.columns:
        text_col = "tweet"
    if text_col is None:
        # fallback: primeira coluna de string
        for c in df.columns:
            if df[c].dtype == object:
                text_col = c
                break
    if text_col is None:
        raise ValueError("Não consegui identificar coluna de texto no dataframe.")

    label_col = None
    for candidate in ["sentiment", "label", "target"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError("Não consegui identificar coluna de label no dataframe.")

    df = df[[text_col, label_col]].dropna()
    df = df.rename(columns={text_col: "text", label_col: "label"})

    # mapear rótulos para 0/1 se necessário
    if df["label"].dtype != int and df["label"].dtype != bool:
        df["label"] = df["label"].map(lambda x: 1 if str(x).strip().lower() in {"1", "pos", "positive", "4", "positivo"} else 0)

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=random_state, stratify=df["label"])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def tokenize_function(examples, tokenizer, max_length=256):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)


def compute_metrics(eval_pred):
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1": metric_f1.compute(predictions=preds, references=labels, average="binary")["f1"]}


def main():
    parser = argparse.ArgumentParser(description="Treino Transformer (DistilBERT) - smoke test configurable")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="Nome do modelo HuggingFace")
    parser.add_argument("--sample-size", type=int, default=2000, help="Número de exemplos para smoke-test (ou use 0 para usar tudo)")
    parser.add_argument("--epochs", type=int, default=1, help="Número de épocas (smoke:1)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size por dispositivo")
    parser.add_argument("--output-dir", default="models/distilbert_sentiment", help="Onde salvar o modelo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_data()
    sample_size = args.sample_size if args.sample_size and args.sample_size > 0 else None
    train_ds, eval_ds = prepare_dataset(df, sample_size=sample_size, random_state=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Criar torch Dataset customizado (tokenização on-the-fly)
    import torch

    class TorchTextDataset(torch.utils.data.Dataset):
        def __init__(self, df, tokenizer, max_length=256):
            self.tokenizer = tokenizer
            self.texts = df["text"].tolist()
            self.labels = [int(x) for x in df["label"].tolist()]
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_dataset = TorchTextDataset(train_ds, tokenizer)
    eval_dataset = TorchTextDataset(eval_ds, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Treino manual com PyTorch (evita dependências do Trainer/accelerate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    from torch.utils.data import DataLoader
    import torch.optim as optim

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    # usaremos sklearn.metrics diretamente para evitar dependências de `datasets`
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Avaliação
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in eval_loader:
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                preds = logits.argmax(-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary")
        print(f"Epoch {epoch+1} - accuracy: {acc:.4f} - f1: {f1:.4f}")

    # salvar artefatos
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Modelo salvo em {args.output_dir}")


if __name__ == "__main__":
    main()
