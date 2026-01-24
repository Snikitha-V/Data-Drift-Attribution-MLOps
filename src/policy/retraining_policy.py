import pandas as pd
import numpy as np
from pathlib import Path

REPORT_DIR = Path("policy")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PERSISTENCE_WINDOWS = 2
PSI_THRESHOLD = 0.25
MIN_SEVERE_FEATURES = 5
PERF_DROP_THRESHOLD = 0.01


def apply_policy(window_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a `retrain` decision column to the window signals dataframe.
    Decision rule:
      - max_psi > PSI_THRESHOLD
      - num_severe >= MIN_SEVERE_FEATURES
      - perf_drop >= PERF_DROP_THRESHOLD (must be a finite number)
    Additionally, the condition must persist for PERSISTENCE_WINDOWS consecutive windows.
    """

    # ensure sorted by window
    window_df = window_df.sort_values("window").reset_index(drop=True)

    decisions = []

    for i in range(len(window_df)):
        row = window_df.iloc[i]

        cond_max_psi = row.get("max_psi", 0) > PSI_THRESHOLD
        cond_num_severe = row.get("num_severe", 0) >= MIN_SEVERE_FEATURES
        perf_drop = row.get("perf_drop", np.nan)
        cond_perf_drop = pd.notna(perf_drop) and (perf_drop >= PERF_DROP_THRESHOLD)

        condition = cond_max_psi and cond_num_severe and cond_perf_drop

        # persistence check
        if condition and i >= PERSISTENCE_WINDOWS - 1:
            persistent = True
            for j in range(i - PERSISTENCE_WINDOWS + 1, i + 1):
                r = window_df.iloc[j]
                pj = (
                    (r.get("max_psi", 0) > PSI_THRESHOLD)
                    and (r.get("num_severe", 0) >= MIN_SEVERE_FEATURES)
                    and (pd.notna(r.get("perf_drop", np.nan)) and (r.get("perf_drop") >= PERF_DROP_THRESHOLD))
                )
                if not pj:
                    persistent = False
                    break
            decisions.append(persistent)
        else:
            decisions.append(False)

    window_df["retrain"] = decisions
    return window_df


def main():
    signals_path = REPORT_DIR / "window_signals.csv"
    if not signals_path.exists():
        raise RuntimeError(f"Signals file not found: {signals_path}. Run src/policy/window_signals.py first.")

    df = pd.read_csv(signals_path)

    # Sanity checks
    assert "window" in df.columns, "window column missing from signals"

    out = apply_policy(df)

    out_path = REPORT_DIR / "window_policy_decisions.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved policy decisions to {out_path}")
    print(out[["window", "retrain", "max_psi", "num_severe", "perf_drop"]].head(20))


if __name__ == "__main__":
    main()
