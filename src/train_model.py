import sys
from pathlib import Path

# ensure imports from src work when running as `python src/train_model.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import lightgbm as lgb
import mlflow
from sklearn.metrics import roc_auc_score
from config import TARGET_COL, MODEL_PARAMS

DATA_DIR = Path("data/processed")


def load(split):
    return pd.read_csv(DATA_DIR / f"{split}.csv")


if __name__ == "__main__":
    train = load("train")
    val = load("val")

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]

    X_val = val.drop(columns=[TARGET_COL])
    y_val = val[TARGET_COL]

    # Encode object dtypes to numeric codes (align categories between train and val)
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        # build categories from both train and val to ensure consistent mapping
        combined = pd.concat([X_train[col].astype(str), X_val[col].astype(str)])
        categories = pd.Categorical(combined).categories
        X_train[col] = pd.Categorical(X_train[col].astype(str), categories=categories).codes
        X_val[col] = pd.Categorical(X_val[col].astype(str), categories=categories).codes

    model = lgb.LGBMClassifier(**MODEL_PARAMS)

    mlflow.start_run(run_name="ieee_cis_baseline")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.log_evaluation(50)],
    )

    val_preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)

    mlflow.log_params(MODEL_PARAMS)
    mlflow.log_metric("val_auc", auc)

    mlflow.sklearn.log_model(model, "model")

    print(f"Validation AUC: {auc:.4f}")
    mlflow.end_run()
