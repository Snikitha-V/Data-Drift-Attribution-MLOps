import sys
from pathlib import Path
import time
import json

# ensure src imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import joblib
import yaml
import os

from data.paysim_prep import preprocess_paysim
from data.cc_prep import preprocess_cc
from split import temporal_split
from drift.windows import create_time_windows
from drift.detect import compute_feature_drift
from drift.attribution import compute_shap_importance, compute_attribution_for_window, load_feature_groups
from policy.window_signals import build_window_signal_table
from policy.retraining_policy import apply_policy

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def _train_model(X_train, y_train, X_val, y_val, model_params=None):
    import lightgbm as lgb
    from config import MODEL_PARAMS

    params = MODEL_PARAMS if model_params is None else model_params
    model = lgb.LGBMClassifier(**params)
    start = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc", callbacks=[lgb.log_evaluation(50)])
    duration = time.time() - start
    return model, duration


def _prepare_features_for_train(df, train_df, val_df, feature_cols):
    X = df.copy()
    cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if col in X.columns:
            combined = pd.concat([train_df[col].astype(str), val_df[col].astype(str)])
            categories = pd.Categorical(combined).categories
            X[col] = pd.Categorical(X[col].astype(str), categories=categories).codes
    X = X.reindex(columns=feature_cols)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(float)
    return X


def run_pipeline(dataset_name: str, data_path: str, feature_groups_path: str, preprocess: str, baseline_days: int = 7, window_days: int = 7):
    print(f"Running pipeline for {dataset_name}")

    data_raw = pd.read_csv(data_path)

    # preprocess
    if preprocess == "paysim":
        df = preprocess_paysim(data_path)
    elif preprocess == "cc":
        df = preprocess_cc(data_path)
    else:
        raise ValueError("Unsupported preprocess type")

    # basic checks
    if "isFraud" not in df.columns:
        raise RuntimeError("Dataset must contain 'isFraud' target")
    if "TransactionDT" not in df.columns:
        raise RuntimeError("Dataset must contain 'TransactionDT' column after preprocessing")

    # temporal split
    train, val, test = temporal_split(df)

    # save processed files for traceability
    proc_dir = Path("data/processed") / dataset_name
    proc_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(proc_dir / "train.csv", index=False)
    val.to_csv(proc_dir / "val.csv", index=False)
    test.to_csv(proc_dir / "test.csv", index=False)

    # prepare features and train model
    feature_cols = train.drop(columns=["isFraud"]).columns.tolist()
    X_train = _prepare_features_for_train(train, train, val, feature_cols)
    y_train = train["isFraud"]
    X_val = _prepare_features_for_train(val, train, val, feature_cols)
    y_val = val["isFraud"]

    model, train_time = _train_model(X_train, y_train, X_val, y_val)

    # save model locally (models/lgbm_model.pkl) so downstream code finds it
    joblib.dump(model, MODELS_DIR / "lgbm_model.pkl")
    with open(MODELS_DIR / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f)

    # windows creation (use train+val to create baseline+windows)
    df_tv = pd.concat([train, val], axis=0).sort_values("TransactionDT")
    baseline, windows = create_time_windows(df_tv, time_col="TransactionDT", baseline_days=baseline_days, window_days=window_days)

    # compute drift per window and save
    drift_dir = REPORTS_DIR
    drift_dir.mkdir(exist_ok=True)

    for i, win in enumerate(windows):
        drift = compute_feature_drift(baseline[feature_cols], win[feature_cols], exclude_cols=["isFraud", "TransactionDT"], bins=10)
        drift.to_csv(drift_dir / f"drift_window_{i}.csv", index=False)

    # attribution
    feature_groups = load_feature_groups(Path(feature_groups_path))

    # compute shap on baseline (ensure categorical encoding as in training)
    X_baseline = _prepare_features_for_train(baseline, train, val, feature_cols)
    shap_imp = compute_shap_importance(model, X_baseline)
    shap_imp.to_csv(drift_dir / "shap_importance.csv")

    all_attr = []
    for i in range(len(windows)):
        drift_df = pd.read_csv(drift_dir / f"drift_window_{i}.csv")
        attr_df = compute_attribution_for_window(drift_df, shap_imp, feature_groups)
        attr_df.to_csv(drift_dir / f"attribution_window_{i}.csv", index=False)
        all_attr.append(attr_df.assign(window=i))

    if all_attr:
        pd.concat(all_attr, axis=0).to_csv(drift_dir / "attribution_all_windows.csv", index=False)

    # build signals (using existing module functions)
    # we need to provide baseline_auc to the builder; compute baseline AUC on val
    try:
        baseline_preds = model.predict_proba(X_val)[:, 1]
        baseline_auc = float(roc_auc_score(y_val, baseline_preds))
    except Exception:
        baseline_auc = float(np.nan)

    from policy.window_signals import build_window_signal_table
    ws = build_window_signal_table(baseline, windows, model, feature_cols, train, val)
    Path("policy").mkdir(exist_ok=True)
    ws.to_csv(Path("policy") / "window_signals.csv", index=False)

    # apply frozen policy
    from policy.retraining_policy import apply_policy
    decisions = apply_policy(ws)
    decisions.to_csv(Path("policy") / "window_policy_decisions.csv", index=False)

    # simulate strategies and collect metrics
    summary = {}
    # strategy 1: periodic every 5 windows
    def simulate_strategy(strategy):
        model_curr = model
        retrains = 0
        aucs = []
        last_retrain = -10  # for cooldown
        # create copy of model to retrain as needed
        for i, win in enumerate(windows):
            X_win = _prepare_features_for_train(win, train, val, feature_cols)
            y_win = win["isFraud"]
            try:
                preds = model_curr.predict_proba(X_win)[:, 1]
                auc = roc_auc_score(y_win, preds)
            except Exception:
                auc = float(np.nan)
            aucs.append(auc)

            do_retrain = False
            if strategy == "periodic":
                if i > 0 and (i % 5 == 0):
                    do_retrain = True
            elif strategy == "psi_only":
                if i - last_retrain > 5:  # cooldown to prevent too frequent retrains
                    drift = pd.read_csv(drift_dir / f"drift_window_{i}.csv")
                    if drift["psi"].max() > 0.25:
                        do_retrain = True
            elif strategy == "policy":
                r = decisions.iloc[i]
                do_retrain = bool(r["retrain"])

            if do_retrain:
                retrains += 1
                last_retrain = i
                # retrain on all data up to end of this window
                data_until = pd.concat([train, val] + windows[: i + 1], axis=0)
                X_re = _prepare_features_for_train(data_until.drop(columns=["isFraud"]), train, val, feature_cols)
                y_re = data_until["isFraud"]
                try:
                    new_model, _ = _train_model(X_re, y_re, X_val, y_val)
                    model_curr = new_model
                except Exception:
                    pass
        return {
            "retrain_count": retrains,
            "mean_auc": float(np.nanmean(aucs)),
            "std_auc": float(np.nanstd(aucs)),
        }

    summary["periodic"] = simulate_strategy("periodic")
    summary["psi_only"] = simulate_strategy("psi_only")
    summary["policy"] = simulate_strategy("policy")

    # save summary
    with open(OUT_DIR / f"{dataset_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Completed pipeline for {dataset_name}. Summary saved to {OUT_DIR / f'{dataset_name}_summary.json'}")
    return summary
