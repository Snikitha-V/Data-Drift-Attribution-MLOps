# src/drift/run_drift.py

from pathlib import Path
import sys
# ensure src/ is on sys.path so we can import the drift package when running the script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drift.windows import load_data, create_time_windows
from drift.detect import compute_feature_drift
import pandas as pd
from tqdm import tqdm

EXCLUDE_COLS = ["isFraud", "TransactionDT"]
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


if __name__ == "__main__":
    df = load_data()

    baseline, windows = create_time_windows(
        df,
        time_col="TransactionDT",
        baseline_days=7,
        window_days=7
    )

    all_summaries = []
    summary_rows = []

    for i, window in enumerate(tqdm(windows, desc="Windows", unit="win")):
        drift_df = compute_feature_drift(
            baseline,
            window,
            exclude_cols=EXCLUDE_COLS
        )

        drift_df = drift_df.sort_values("psi", ascending=False)

        # Save per-window report
        out_path = REPORTS_DIR / f"drift_window_{i}.csv"
        drift_df.to_csv(out_path, index=False)

        # Print top-10
        print(f"\n=== Drift Report | Window {i} ===")
        print(drift_df.head(10))

        summary = drift_df.assign(window=i)
        all_summaries.append(summary)

        # window-level counts
        psi_count = int((drift_df["psi"] > 0.25).sum())
        ks_count = int((drift_df["ks_pvalue"] < 0.05).sum())
        summary_rows.append({
            "window": i,
            "psi_drift_count": psi_count,
            "ks_drift_count": ks_count,
        })

    # Save combined
    if all_summaries:
        combined = pd.concat(all_summaries, axis=0)
        combined.to_csv(REPORTS_DIR / "drift_all_windows.csv", index=False)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(REPORTS_DIR / "drift_summary.csv", index=False)

        print(f"Saved drift reports to {REPORTS_DIR}")
