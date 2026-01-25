import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policy.retraining_policy import apply_policy

# Load window signals
ws_df = pd.read_csv(Path("policy") / "window_signals_partial.csv")

# Full policy
decisions_full = apply_policy(ws_df.copy(), use_persistence=True, use_num_severe=True)
retrain_count_full = decisions_full["retrain"].sum()
mean_auc_full = decisions_full["window_auc"].mean()

# No persistence
decisions_no_persist = apply_policy(ws_df.copy(), use_persistence=False, use_num_severe=True)
retrain_count_no_persist = decisions_no_persist["retrain"].sum()
mean_auc_no_persist = decisions_no_persist["window_auc"].mean()

# No attribution
decisions_no_attr = apply_policy(ws_df.copy(), use_persistence=True, use_num_severe=False)
retrain_count_no_attr = decisions_no_attr["retrain"].sum()
mean_auc_no_attr = decisions_no_attr["window_auc"].mean()

# Print table
print("| Policy Variant        | Retrains | Mean AUC |")
print("|-----------------------|----------|----------|")
print(f"| Full policy          | {retrain_count_full}        | {mean_auc_full:.2f}     |")
print(f"| No persistence       | {retrain_count_no_persist}        | {mean_auc_no_persist:.2f}     |")
print(f"| No attribution       | {retrain_count_no_attr}        | {mean_auc_no_attr:.2f}     |")