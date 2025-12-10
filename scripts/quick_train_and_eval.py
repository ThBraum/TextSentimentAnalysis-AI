from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sentiment_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main(sample_n: int = 20000):
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["clean_text", "sentiment"]).copy()
    df["clean_text"] = df["clean_text"].astype(str)
    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)

    X = df["clean_text"]
    y = df["sentiment"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"Quick model accuracy: {acc:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "quick_model.pkl"
    vec_path = MODEL_DIR / "quick_tfidf_vectorizer.pkl"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"Saved quick model -> {model_path}")

    # save reports
    pd.Series(y_test).to_csv(REPORTS_DIR / "y_true.csv", index=False)
    pd.Series(y_pred).to_csv(REPORTS_DIR / "y_pred.csv", index=False)

    # try to save scores
    score_vals = None
    if hasattr(model, "decision_function"):
        try:
            score_vals = model.decision_function(X_test_vec)
        except Exception:
            score_vals = None
    elif hasattr(model, "predict_proba"):
        try:
            score_vals = model.predict_proba(X_test_vec)[:, 1]
        except Exception:
            score_vals = None

    if score_vals is not None:
        pd.Series(score_vals).to_csv(REPORTS_DIR / "y_score.csv", index=False)
        print(f"Saved score -> {REPORTS_DIR / 'y_score.csv'}")


if __name__ == "__main__":
    main()
