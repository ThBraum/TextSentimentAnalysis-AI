from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
DEFAULT_OUT = EXTERNAL_DIR / "rebalanced"


def _ensure_text_sentiment(df: pd.DataFrame, text_col: str, label_col: str, source: str) -> pd.DataFrame:
    df = df.rename(columns={text_col: "text", label_col: "sentiment"})
    df = df[["text", "sentiment"]].copy()
    df["source"] = source
    return df


def _downsample(df: pd.DataFrame, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed)


def _upsample(df: pd.DataFrame, multiplier: float, seed: int) -> pd.DataFrame:
    if multiplier <= 1.0:
        return df
    current = len(df)
    target = int(current * multiplier)
    extra = max(target - current, 0)
    if extra == 0:
        return df
    resampled = df.sample(n=extra, replace=True, random_state=seed)
    return pd.concat([df, resampled], ignore_index=True)


def save_df(df: pd.DataFrame, original_path: Path, out_dir: Path, overwrite: bool) -> Path:
    if overwrite:
        out_dir = original_path.parent
        out_path = original_path
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / original_path.name
    df.to_csv(out_path, index=False)
    return out_path


def process_datasets(
    amazon_max: Optional[int],
    imdb_multiplier: float,
    sentiment140_multiplier: float,
    overwrite: bool,
    seed: int,
    out_dir: Path,
) -> None:
    amazon_train = EXTERNAL_DIR / "amazon_train.csv"
    amazon_test = EXTERNAL_DIR / "amazon_test.csv"
    imdb_path = EXTERNAL_DIR / "imdb_reviews.csv"
    sent140_path = EXTERNAL_DIR / "sentiment140.csv"

    if not amazon_train.exists() or not amazon_test.exists():
        raise SystemExit("Amazon CSVs not found in data/external. Run BZ2_to_ CSV.py first.")

    frames = []

    for path in [amazon_train, amazon_test]:
        df = _ensure_text_sentiment(pd.read_csv(path), text_col="text", label_col="sentiment", source="amazon")
        df = _downsample(df, max_rows=amazon_max, seed=seed)
        frames.append((df, path))

    if imdb_path.exists():
        imdb_df = _ensure_text_sentiment(pd.read_csv(imdb_path), text_col="review", label_col="sentiment", source="imdb")
        imdb_df = _upsample(imdb_df, multiplier=imdb_multiplier, seed=seed)
        frames.append((imdb_df, imdb_path))
    else:
        print(f"IMDb file missing, skipping: {imdb_path}")

    if sent140_path.exists():
        s140_df = _ensure_text_sentiment(pd.read_csv(sent140_path), text_col="text", label_col="sentiment", source="sentiment140")
        s140_df = _upsample(s140_df, multiplier=sentiment140_multiplier, seed=seed)
        frames.append((s140_df, sent140_path))
    else:
        print(f"Sentiment140 file missing, skipping: {sent140_path}")

    for df, original_path in frames:
        out_path = save_df(df, original_path=original_path, out_dir=out_dir, overwrite=overwrite)
        print(f"Wrote {len(df):,} rows -> {out_path}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebalance external sentiment CSVs by down/upsampling real rows.")
    parser.add_argument("--amazon-max", type=int, default=150_000, help="Cap rows per Amazon split (None to keep all).")
    parser.add_argument("--imdb-multiplier", type=float, default=2.0, help="Upsample IMDb by this factor (1.0 keeps as is).")
    parser.add_argument("--sentiment140-multiplier", type=float, default=1.5, help="Upsample Sentiment140 by this factor.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite original CSVs instead of writing to data/external/rebalanced/.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory when not overwriting.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_datasets(
        amazon_max=args.amazon_max,
        imdb_multiplier=args.imdb_multiplier,
        sentiment140_multiplier=args.sentiment140_multiplier,
        overwrite=args.overwrite,
        seed=args.seed,
        out_dir=args.out_dir,
    )
