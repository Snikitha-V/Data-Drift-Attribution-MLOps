import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drift.windows import create_time_windows
from drift.detect import compute_feature_drift
from drift.attribution import compute_attribution_for_window, load_feature_groups
from policy.window_signals import compute_window_signals, build_window_signal_table
from policy.retraining_policy import apply_policy

# Load processed PaySim data
proc_dir = Path("data/processed/paysim")
train = pd.read_csv(proc_dir / "train.csv")
val = pd.read_csv(proc_dir / "val.csv")
df_tv = pd.concat([train, val], axis=0).sort_values("TransactionDT")

# Create windows
baseline, windows = create_time_windows(df_tv, time_col="TransactionDT", baseline_days=7, window_days=7)

# Choose windows 9 and 10 for stress test
stress_window_idxs = [9, 10]
for idx in stress_window_idxs:
    if idx >= len(windows):
        raise ValueError("Not enough windows")

# Inject drift into amount for both
for idx in stress_window_idxs:
    window_df = windows[idx].copy()
    window_df["amount"] *= 100  # Strong drift
    windows[idx] = window_df

print(f"Injected drift into windows {stress_window_idxs}: amount multiplied by 100")

# Recompute drift and attribution for both
for idx in stress_window_idxs:
    window_df = windows[idx]
    feature_cols = [c for c in baseline.columns if c not in ["isFraud", "TransactionDT"]]
    drift_df = compute_feature_drift(baseline[feature_cols], window_df[feature_cols], exclude_cols=["isFraud", "TransactionDT"], bins=10)
    drift_df.to_csv(Path("reports") / f"drift_window_{idx}.csv", index=False)

    feature_groups = load_feature_groups(Path("src/features/feature_groups_paysim.yaml"))
    shap_imp = pd.read_csv(Path("reports") / "shap_importance.csv", index_col=0)["0"]
    attr_df = compute_attribution_for_window(drift_df, shap_imp, feature_groups)
    attr_df.to_csv(Path("reports") / f"attribution_window_{idx}.csv", index=False)

# Recompute signals for all windows (since persistence depends on sequence)
model = joblib.load(Path("models") / "lgbm_model.pkl")
with open(Path("models") / "feature_cols.json", "r") as f:
    feature_cols = json.load(f)

ws = build_window_signal_table(baseline, windows[:20], model, feature_cols, train, val)
ws.to_csv(Path("policy") / "window_signals_stressed.csv", index=False)

# Reapply policy
decisions = apply_policy(ws)
decisions.to_csv(Path("policy") / "window_policy_decisions_stressed.csv", index=False)

# Check retrain for window 10
retrain = decisions.loc[10, "retrain"]
print(f"Stress test result: Window 10 retrain = {retrain}")