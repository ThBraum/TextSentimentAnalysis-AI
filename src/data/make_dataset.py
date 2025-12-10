import pandas as pd
from pathlib import Path

# Função para carregar o dataset do IMDb
def load_imdb_data(path):
    return pd.read_csv(path)

# Função para carregar o dataset do Sentiment140
def load_sentiment140_data(path):
    return pd.read_csv(path, encoding='latin1')

# Função para carregar o dataset do Amazon
def load_amazon_data(path):
    return pd.read_csv(path)

# Função para combinar os datasets
def combine_datasets(imdb, sentiment140, amazon):
    imdb['source'] = 'IMDb'
    sentiment140['source'] = 'Sentiment140'
    amazon['source'] = 'Amazon'
    
    # Concatena
    combined = pd.concat([imdb, sentiment140, amazon], ignore_index=True)
    return combined

def main():
    imdb_data = load_imdb_data('data/external/imdb_reviews.csv')
    sentiment140_data = load_sentiment140_data('data/external/sentiment140.csv')
    amazon_candidates = [
        Path('data/external/amazon_reviews.csv'),
        Path('data/external/amazon.csv'),
        Path('data/external/amazon_train.csv'),
    ]
    amazon_data = None
    for cand in amazon_candidates:
        if cand.exists():
            # se for amazon_train.csv e existir also amazon_test.csv, concatene
            if cand.name == 'amazon_train.csv':
                test_path = cand.parent / 'amazon_test.csv'
                if test_path.exists():
                    a_train = load_amazon_data(str(cand))
                    a_test = load_amazon_data(str(test_path))
                    amazon_data = pd.concat([a_train, a_test], ignore_index=True)
                    break
            amazon_data = load_amazon_data(str(cand))
            break
    if amazon_data is None:
        raise FileNotFoundError('Nenhum arquivo Amazon encontrado em data/external (procure por amazon_reviews.csv, amazon.csv ou amazon_train.csv + amazon_test.csv)')

    combined_data = combine_datasets(imdb_data, sentiment140_data, amazon_data)
    out_path = 'data/interim/combined_data.csv'
    combined_data.to_csv(out_path, index=False)
    print(f'Conjunto combinado salvo em {out_path} (linhas: {len(combined_data)})')


if __name__ == "__main__":
    main()
