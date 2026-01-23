# src/drift/attribution.py

import json
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

import sys
# ensure src/ is on sys.path so imports work when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drift.windows import load_data, create_time_windows

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

FEATURE_GROUPS_PATH = Path("src/features/feature_groups.yaml")
MODEL_PATH = Path("models/lgbm_model.pkl")
FEATURE_COLS_PATH = Path("models/feature_cols.json")

EXCLUDE_COLS = ["isFraud", "TransactionDT"]


def load_feature_groups(path=FEATURE_GROUPS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_shap_importance(model, X, sample_size=10000):
    # sample if too large
    if len(X) > sample_size:
        Xs = X.sample(sample_size, random_state=42)
    else:
        Xs = X

    # Ensure numeric numpy input for SHAP / LightGBM
    Xs = Xs.fillna(0).astype(float)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(Xs)

    # shap_values for binary classification returns [neg, pos]
    if isinstance(shap_vals, list):
        vals = np.abs(shap_vals[1])
    else:
        vals = np.abs(shap_vals)

    mean_abs = np.mean(vals, axis=0)
    return pd.Series(mean_abs, index=Xs.columns)


def compute_attribution_for_window(drift_df, shap_imp, feature_groups):
    # drift_df has columns: feature, psi, ks_pvalue
    df = drift_df.set_index("feature")["psi"].reindex(shap_imp.index).fillna(0)

    impacts = df * shap_imp.abs()

    # group aggregation
    results = []
    for group, members in feature_groups.items():
        members_in_model = [m for m in members if m in impacts.index]
        group_impact = impacts.loc[members_in_model].sum() if members_in_model else 0.0
        results.append({"group": group, "impact": float(group_impact)})

    res_df = pd.DataFrame(results).set_index("group")
    total = res_df["impact"].sum()
    res_df["normalized_impact"] = (res_df["impact"] / total).fillna(0.0)
    return res_df.reset_index()


if __name__ == "__main__":
    # Load groups
    feature_groups = load_feature_groups()

    # Load baseline and windows
    df = load_data()
    baseline, windows = create_time_windows(df, time_col="TransactionDT", baseline_days=7, window_days=7)

    # Load model and feature columns
    if not MODEL_PATH.exists() or not FEATURE_COLS_PATH.exists():
        raise RuntimeError("Trained model or feature_cols.json not found in models/; run training/saving step first.")

    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    # Work on an explicit copy to avoid SettingWithCopyWarning
    X_baseline = baseline[feature_cols].copy()

    # Ensure categorical columns are encoded consistently with training encoding
    train_df = pd.read_csv(Path('data/processed') / 'train.csv')
    val_df = pd.read_csv(Path('data/processed') / 'val.csv')
    for col in X_baseline.columns:
        if X_baseline[col].dtype == 'object' or pd.api.types.is_string_dtype(X_baseline[col]):
            combined = pd.concat([train_df[col].astype(str), val_df[col].astype(str)])
            categories = pd.Categorical(combined).categories
            X_baseline.loc[:, col] = pd.Categorical(X_baseline[col].astype(str), categories=categories).codes

    # Ensure all dtypes are numeric for LightGBM (final safety cast)
    for col in X_baseline.columns:
        # try numeric conversion first
        coerced = pd.to_numeric(X_baseline[col], errors='coerce')
        if pd.api.types.is_numeric_dtype(coerced):
            X_baseline.loc[:, col] = coerced.astype(float)
        else:
            X_baseline.loc[:, col] = pd.Categorical(X_baseline[col].astype(str)).codes.astype(int)

    # Verify all columns are numeric and print a sample of any remaining problematic columns
    non_numeric_after = [c for c in X_baseline.columns if not pd.api.types.is_numeric_dtype(X_baseline[c])]
    if non_numeric_after:
        print('After coercion, still non-numeric (sample):', non_numeric_after[:20])
        for c in non_numeric_after[:10]:
            print(c, X_baseline[c].head(5), X_baseline[c].dtype)

    # Debug: list non-numeric columns after coercion
    non_numeric_after = [c for c in X_baseline.columns if not pd.api.types.is_numeric_dtype(X_baseline[c])]
    if non_numeric_after:
        print("Non-numeric columns after coercion (sample):", non_numeric_after[:20])
        for c in non_numeric_after[:10]:
            vals = X_baseline[c].dropna().unique()[:5]
            print(f"  {c}: dtype={X_baseline[c].dtype}, sample_values={vals}")

    print("Computing SHAP importances on baseline (this may take some time)...")
    shap_imp = compute_shap_importance(model, X_baseline)

    shap_series = shap_imp.sort_values(ascending=False)
    shap_series.to_csv(REPORTS_DIR / "shap_importance.csv", header=["mean_abs_shap"])
    print(f"Saved SHAP importance to {REPORTS_DIR / 'shap_importance.csv'}")

    # For each window, compute attribution
    all_attr = []
    for i, window in enumerate(windows):
        drift_path = REPORTS_DIR / f"drift_window_{i}.csv"
        if not drift_path.exists():
            print(f"Warning: missing drift report for window {i}")
            continue
        drift_df = pd.read_csv(drift_path)
        attr_df = compute_attribution_for_window(drift_df, shap_imp, feature_groups)
        attr_df.to_csv(REPORTS_DIR / f"attribution_window_{i}.csv", index=False)
        all_attr.append(attr_df.assign(window=i))

        # plot group impacts
        plt.figure(figsize=(8, 4))
        plt.bar(attr_df["group"], attr_df["impact"])
        plt.title(f"Drift Attribution - Window {i}")
        plt.ylabel("Impact (PSI * |SHAP|)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"attribution_window_{i}.png")
        plt.close()

    if all_attr:
        combined = pd.concat(all_attr, axis=0)
        # total impact per window
        summary_total = combined.groupby('window')['impact'].sum().reset_index(name='total_impact')
        # pivot group impacts into columns
        pivot = (
            combined.pivot_table(index='window', columns='group', values='impact', aggfunc='sum')
            .fillna(0)
            .reset_index()
        )
        summary = summary_total.merge(pivot, on='window')

        # Save the normalized group impacts per window
        combined.to_csv(REPORTS_DIR / "attribution_all_windows.csv", index=False)
        summary.to_csv(REPORTS_DIR / "attribution_summary.csv", index=False)
        print(f"Saved attribution reports to {REPORTS_DIR}")

    # Also save a timeline plot (stacked) for normalized impacts
    timeline = []
    for i, window in enumerate(windows):
        p = REPORTS_DIR / f"attribution_window_{i}.csv"
        if p.exists():
            tmp = pd.read_csv(p).set_index('group')
            timeline.append(tmp['normalized_impact'].rename(i))

    if timeline:
        timeline_df = pd.concat(timeline, axis=1).fillna(0).T
        timeline_df.plot(kind='bar', stacked=True, figsize=(12, 5))
        plt.title('Normalized Group Attribution Over Time')
        plt.xlabel('Window')
        plt.ylabel('Normalized Impact')
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / 'attribution_timeline.png')
        plt.close()
        print(f"Saved attribution timeline to {REPORTS_DIR / 'attribution_timeline.png'}")
