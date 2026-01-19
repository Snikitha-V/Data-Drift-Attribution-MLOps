import sys
from pathlib import Path

# ensure imports from src work when running as `python src/evaluate.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import mlflow
from sklearn.metrics import roc_auc_score
from config import TARGET_COL

DATA_DIR = Path("data/processed")

if __name__ == "__main__":
    # Load data
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    # Prepare X_test using category mappings built from train+val (same as training)
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]

    # Identify categorical columns from training data
    cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        combined = pd.concat([train[col].astype(str), val[col].astype(str)])
        categories = pd.Categorical(combined).categories
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=categories).codes

    # Ensure feature ordering matches training
    feature_cols = train.drop(columns=[TARGET_COL]).columns.tolist()
    X_test = X_test[feature_cols]

    # Find the most recent run and load its logged model
    runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
    if runs is None or len(runs) == 0:
        raise RuntimeError("No MLflow runs found. Train a model first.")
    run_id = runs.iloc[0]["run_id"]
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    print(f"Test AUC: {auc:.4f}")
