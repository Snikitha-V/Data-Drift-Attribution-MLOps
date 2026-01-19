import sys
from pathlib import Path

# ensure imports from src work when running as `python src/split.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from config import TIME_COL, TRAIN_RATIO, VAL_RATIO, TEST_RATIO

DATA_PATH = Path("data/processed/ieee_cis_processed.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(exist_ok=True)


def temporal_split(df):
    df = df.sort_values(TIME_COL)

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    train, val, test = temporal_split(df)

    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)

    print("Temporal split complete")
