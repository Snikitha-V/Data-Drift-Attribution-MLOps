import json
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
from scipy.stats import entropy
import mlflow
import sys
# ensure src/ is on sys.path so relative imports work when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# local imports
from drift.windows import load_data, create_time_windows

REPORTS_DIR = Path("reports")
OUT_DIR = Path("policy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEVERE_PSI_THRESHOLD = 0.25

MODEL_FEATURES_PATH = Path("models/feature_cols.json")


def load_psi(window_id: int) -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / f"drift_window_{window_id}.csv")


def load_attribution(window_id: int) -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / f"attribution_window_{window_id}.csv")


import joblib


def get_latest_model():
    # Try MLflow first, but gracefully fall back to a local model file if needed
    runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
    if runs is None or len(runs) == 0:
        print("No MLflow runs found. Will try to load local model if available.")
    else:
        run_id = runs.iloc[0]["run_id"]
        try:
            # check artifacts first to avoid attempted downloads that will fail
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            artifacts = client.list_artifacts(run_id, path="")
            artifact_names = [a.path for a in artifacts]
            if "model" in artifact_names:
                model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
                return model
            else:
                print(f"No 'model' artifact found in MLflow run {run_id}; will try local model.")
        except Exception as e:
            print(f"Warning: failed to load model from MLflow run {run_id}: {e}")

    # Fallback: try to load a local model file
    local_model_path = Path("models/lgbm_model.pkl")
    if local_model_path.exists():
        try:
            model = joblib.load(local_model_path)
            print(f"Loaded local model from {local_model_path}")
            return model
        except Exception as e:
            print(f"Warning: failed to load local model {local_model_path}: {e}")

    print("No model available (MLflow or local). Continuing without model.")
    return None


def get_baseline_auc_from_mlflow():
    runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
    if runs is None or len(runs) == 0:
        return float(np.nan)
    row = runs.iloc[0]
    # common column names in search_runs: metrics.<metric_name>
    for k in ["metrics.val_auc", "val_auc", "metrics.val-auc"]:
        if k in row.index and not pd.isna(row[k]):
            return float(row[k])
    # fallback to client
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        run_info = client.get_run(row["run_id"])
        metrics = run_info.data.metrics
        if "val_auc" in metrics:
            return float(metrics["val_auc"])
    except Exception:
        pass
    return float(np.nan)


def _prepare_features(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    X = df.copy()

    # Encode object dtypes to numeric codes based on train+val categories
    cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if col in X.columns:
            combined = pd.concat([train_df[col].astype(str), val_df[col].astype(str)])
            categories = pd.Categorical(combined).categories
            X[col] = pd.Categorical(X[col].astype(str), categories=categories).codes

    # Ensure ordering and numeric types
    X = X.reindex(columns=feature_cols)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(float)

    return X


def compute_window_signals(window_id: int, baseline_df: pd.DataFrame, windows: list, model, feature_cols: list, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
    """Compute signals for a specific window using provided windows list (no re-loading)."""
    psi_df = load_psi(window_id)
    attr_df = load_attribution(window_id)

    # ---- PSI signals ----
    max_psi = float(psi_df["psi"].max())
    mean_psi = float(psi_df["psi"].mean())
    num_severe = int((psi_df["psi"] > SEVERE_PSI_THRESHOLD).sum())

    # ---- Attribution signals ----
    total_impact = float(attr_df["impact"].sum())

    if total_impact > 0:
        attr_df = attr_df.assign(share=attr_df["impact"] / total_impact)
        top_row = attr_df.sort_values("share", ascending=False).iloc[0]
        top_group = str(top_row["group"])
        top_group_share = float(top_row["share"])
        shares = attr_df["share"].clip(lower=0).values
        if shares.sum() > 0:
            attr_entropy = float(entropy(shares))
        else:
            attr_entropy = 0.0
    else:
        top_group = "none"
        top_group_share = 0.0
        attr_entropy = 0.0

    # ---- Performance signal ----
    baseline_auc = float(np.nan)
    window_auc = float(np.nan)

    if window_id >= len(windows):
        raise IndexError(f"Window id {window_id} out of range (0..{len(windows)-1})")

    this_win = windows[window_id]

    if model is not None:
        try:
            from sklearn.metrics import roc_auc_score
            y_baseline = baseline_df["isFraud"]
            X_baseline = _prepare_features(baseline_df, train_df, val_df, feature_cols)
            baseline_preds = model.predict_proba(X_baseline)[:, 1]
            baseline_auc = float(roc_auc_score(y_baseline, baseline_preds))
        except Exception:
            baseline_auc = float(np.nan)

        try:
            y_win = this_win["isFraud"]
            X_win = _prepare_features(this_win, train_df, val_df, feature_cols)
            window_preds = model.predict_proba(X_win)[:, 1]
            window_auc = float(roc_auc_score(y_win, window_preds))
        except Exception:
            window_auc = float(np.nan)
    else:
        baseline_auc = get_baseline_auc_from_mlflow()
        window_auc = float(np.nan)

    perf_drop = baseline_auc - window_auc if (not np.isnan(baseline_auc) and not np.isnan(window_auc)) else float(np.nan)

    return {
        "window": int(window_id),
        "max_psi": max_psi,
        "mean_psi": mean_psi,
        "num_severe": num_severe,
        "top_group": top_group,
        "top_group_share": top_group_share,
        "attribution_entropy": attr_entropy,
        "perf_drop": perf_drop,
        "baseline_auc": baseline_auc,
        "window_auc": window_auc,
    }


def build_window_signal_table(baseline_df: pd.DataFrame, windows: list, model, feature_cols: list, train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w in range(len(windows)):
        rows.append(compute_window_signals(w, baseline_df, windows, model, feature_cols, train_df, val_df))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Load baseline & windows
    df = load_data()
    baseline_df, windows = create_time_windows(df, time_col="TransactionDT", baseline_days=7, window_days=7)

    # Load model and feature cols
    model = get_latest_model()

    if not MODEL_FEATURES_PATH.exists():
        raise RuntimeError("models/feature_cols.json not found; run save_model.py first.")

    with open(MODEL_FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    # Load training data for category mappings
    train_df = pd.read_csv(Path('data/processed') / 'train.csv')
    val_df = pd.read_csv(Path('data/processed') / 'val.csv')

    df_signals = build_window_signal_table(baseline_df, len(windows), model, feature_cols, train_df, val_df)
    (OUT_DIR / "window_signals.csv").parent.mkdir(parents=True, exist_ok=True)
    df_signals.to_csv(OUT_DIR / "window_signals.csv", index=False)

    print(f"Saved {OUT_DIR / 'window_signals.csv'}")
    print(df_signals.head())