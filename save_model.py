import sys
from pathlib import Path
import json

# allow importing src.config when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import TARGET_COL

import mlflow
import joblib
import pandas as pd


def main():
    DATA_DIR = Path("data/processed")
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(exist_ok=True)

    # find latest run
    runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
    if runs is None or len(runs) == 0:
        raise RuntimeError("No MLflow runs found. Train a model first.")
    run_id = runs.iloc[0]["run_id"]

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    model_path = MODELS_DIR / "lgbm_model.pkl"
    joblib.dump(model, model_path)

    # save feature columns
    train = pd.read_csv(DATA_DIR / "train.csv")
    feature_cols = train.drop(columns=[TARGET_COL]).columns.tolist()
    with open(MODELS_DIR / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f)

    print(f"Saved model to {model_path}")
    print(f"Saved feature columns to {MODELS_DIR / 'feature_cols.json'}")


if __name__ == "__main__":
    main()
