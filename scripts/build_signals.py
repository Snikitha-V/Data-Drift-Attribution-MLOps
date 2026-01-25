import pandas as pd
import json
import joblib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drift.windows import create_time_windows
from policy.window_signals import build_window_signal_table

# Load processed PaySim data
proc_dir = Path("data/processed/paysim")
train = pd.read_csv(proc_dir / "train.csv")
val = pd.read_csv(proc_dir / "val.csv")
df_tv = pd.concat([train, val], axis=0).sort_values("TransactionDT")

# Create windows
baseline, windows = create_time_windows(df_tv, time_col="TransactionDT", baseline_days=7, window_days=7)

print(f"Number of windows: {len(windows)}")

# Load model and feature cols
model = joblib.load(Path("models") / "lgbm_model.pkl")
with open(Path("models") / "feature_cols.json", "r") as f:
    feature_cols = json.load(f)

# Build signals for first 20 windows to test
ws = build_window_signal_table(baseline, windows[:20], model, feature_cols, train, val)
Path("policy").mkdir(exist_ok=True)
ws.to_csv(Path("policy") / "window_signals_partial.csv", index=False)

print("Saved partial signals")