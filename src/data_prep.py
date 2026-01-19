import sys
from pathlib import Path

# ensure imports from src work when running as `python src/data_prep.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from tqdm.auto import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_and_merge(chunksize=200_000):
    trans_path = RAW_DIR / "train_transaction.csv"
    ident_path = RAW_DIR / "train_identity.csv"

    # Read transactions in chunks and show progress
    print(f"Reading {trans_path} in chunks (chunksize={chunksize})...")
    chunks = []
    for chunk in tqdm(pd.read_csv(trans_path, chunksize=chunksize), desc="trans chunks"):
        chunks.append(chunk)
    trans = pd.concat(chunks, ignore_index=True)

    print(f"Reading {ident_path}...")
    ident = pd.read_csv(ident_path)

    print("Merging transaction and identity data...")
    df = trans.merge(ident, on="TransactionID", how="left")
    return df


def basic_cleaning(df):
    # Drop high-cardinality IDs (keep meaningful ones)
    drop_cols = ["TransactionID"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Fill missing with progress
    print("Cleaning columns and filling missing values...")
    for col in tqdm(df.columns, desc="columns"):
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].fillna(df[col].median())

    return df


def save_processed(df):
    path = PROCESSED_DIR / "ieee_cis_processed.csv"
    df.to_csv(path, index=False)
    print(f"Saved processed data to {path}")


if __name__ == "__main__":
    df = load_and_merge()
    df = basic_cleaning(df)
    save_processed(df)
