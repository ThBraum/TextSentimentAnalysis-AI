from pathlib import Path

import bz2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
BZ2_DIR = PROJECT_ROOT / "bz2"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"


def decompress_and_process_bz2(file_path: Path) -> list[list[str]]:
    with bz2.open(file_path, "rt") as file:
        lines = file.readlines()

    data = []
    for line in lines:
        sentiment, text = line.split(" ", 1)
        data.append([sentiment, text.strip()])

    return data


def save_to_csv(data: list[list[str]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data, columns=["sentiment", "text"])
    df.to_csv(output_file, index=False)


def main() -> None:
    train_bz2 = BZ2_DIR / "train.ft.txt.bz2"
    test_bz2 = BZ2_DIR / "test.ft.txt.bz2"

    if not train_bz2.exists() or not test_bz2.exists():
        raise SystemExit("Coloque train.ft.txt.bz2 e test.ft.txt.bz2 dentro de ./bz2 antes de rodar este script.")

    train_data = decompress_and_process_bz2(train_bz2)
    test_data = decompress_and_process_bz2(test_bz2)

    save_to_csv(train_data, EXTERNAL_DIR / "amazon_train.csv")
    save_to_csv(test_data, EXTERNAL_DIR / "amazon_test.csv")

    print("Conversão concluída. CSVs salvos em data/external/.")


if __name__ == "__main__":
    main()
