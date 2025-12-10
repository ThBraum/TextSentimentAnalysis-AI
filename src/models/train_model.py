import argparse
from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib


# Função para carregar e preparar os dados
def load_and_preprocess_data():
    data = pd.read_csv('data/processed/sentiment_data.csv')
    # detectar coluna de texto
    if 'text' in data.columns:
        X = data['text']
    elif 'clean_text' in data.columns:
        X = data['clean_text']
    else:
        # fallback: primeira coluna de tipo object
        text_cols = [c for c in data.columns if data[c].dtype == object]
        if not text_cols:
            raise ValueError('Nenhuma coluna de texto encontrada no dataset processado')
        X = data[text_cols[0]]

    if 'sentiment' in data.columns:
        y = data['sentiment']
    elif 'label' in data.columns:
        y = data['label']
    else:
        raise ValueError('Nenhuma coluna de rótulo encontrada no dataset processado')

    return X, y


# Função para treinar o modelo
def train_naive_bayes_model(X, y):
    vectorizer = TfidfVectorizer(stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Predição
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia: {accuracy * 100:.2f}%')

    # Salvar o modelo treinado
    joblib.dump(model, 'models/naive_bayes_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')


if __name__ == "__main__":
    X, y = load_and_preprocess_data()
    train_naive_bayes_model(X, y)
