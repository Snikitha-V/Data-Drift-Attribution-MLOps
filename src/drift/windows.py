# src/drift/windows.py

# src/drift/windows.py

import pandas as pd
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm
import matplotlib.pyplot as plt


SECONDS_IN_DAY = 24 * 60 * 60


def load_data():
    """
    Load TRAIN + VAL data only.
    We NEVER use test data for drift monitoring.
    """
    base_path = Path("data/processed")
    train = pd.read_csv(base_path / "train.csv")
    val = pd.read_csv(base_path / "val.csv")

    df = pd.concat([train, val], axis=0)
    return df


def create_time_windows(
    df: pd.DataFrame,
    time_col: str,
    baseline_days: int = 7,
    window_days: int = 7,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    Creates:
    - One fixed baseline window
    - Sequential rolling monitoring windows

    Returns:
    baseline_df, list_of_window_dfs
    """

    # Sort by time (CRITICAL)
    df = df.sort_values(time_col).reset_index(drop=True)

    start_time = df[time_col].min()

    baseline_end_time = start_time + baseline_days * SECONDS_IN_DAY

    baseline_df = df[df[time_col] < baseline_end_time]

    windows = []
    current_start = baseline_end_time

    max_time = df[time_col].max()

    # estimate number of windows for a progress bar
    if current_start >= max_time:
        est_windows = 0
    else:
        est_windows = int((max_time - current_start) // (window_days * SECONDS_IN_DAY)) + 1

    # keep the progress bar visible after completion and print per-window messages
    window_idx = 0
    with tqdm(total=est_windows, desc="Creating windows", unit="win", leave=True, ascii=True) as pbar:
        while current_start < max_time:
            current_end = current_start + window_days * SECONDS_IN_DAY

            window_df = df[
                (df[time_col] >= current_start)
                & (df[time_col] < current_end)
            ]

            if len(window_df) > 0:
                windows.append(window_df)
                tqdm.write(f"Created window {window_idx}: {len(window_df)} samples")
                window_idx += 1

            current_start = current_end
            pbar.update(1)

    return baseline_df, windows


def summarize_windows(baseline_df, windows):
    """
    Print quick sanity checks.
    """
    print("====== TIME WINDOW SUMMARY ======")
    print(f"Baseline samples: {len(baseline_df)}")

    for i, w in enumerate(windows):
        print(f"Window {i}: {len(w)} samples")

    print("Total windows:", len(windows))
    print("================================")


def plot_window_sizes(baseline_df, windows, save_path: str = "reports/window_sizes.png"):
    sizes = [len(baseline_df)] + [len(w) for w in windows]
    labels = ["Baseline"] + [f"W{i}" for i in range(len(windows))]

    plt.figure(figsize=(12, 4))
    plt.plot(labels, sizes, marker="o")
    plt.title("Samples per Time Window")
    plt.xlabel("Window")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # Ensure reports dir exists and save figure to avoid blocking interactive backends
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    import time

    start = time.perf_counter()

    df = load_data()

    baseline, windows = create_time_windows(
        df,
        time_col="TransactionDT",
        baseline_days=7,
        window_days=7,
    )

    summarize_windows(baseline, windows)
    try:
        plot_window_sizes(baseline, windows)
    except Exception:
        print("Plotting skipped or failed.")

    elapsed = time.perf_counter() - start
    print(f"Done. Script runtime: {elapsed:.2f} seconds")

import pandas as pd
from pathlib import Path
from typing import List, Tuple

SECONDS_IN_DAY = 24 * 60 * 60


def load_data():
    """
    Load TRAIN + VAL data only.
    We NEVER use test data for drift monitoring.
    """
    base_path = Path("data/processed")
    train = pd.read_csv(base_path / "train.csv")
    val = pd.read_csv(base_path / "val.csv")

    df = pd.concat([train, val], axis=0)
    return df


def create_time_windows(
    df: pd.DataFrame,
    time_col: str,
    baseline_days: int = 7,
    window_days: int = 7,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    Creates:
    - One fixed baseline window
    - Sequential rolling monitoring windows

    Returns:
    baseline_df, list_of_window_dfs
    """

    # Sort by time (CRITICAL)
    df = df.sort_values(time_col).reset_index(drop=True)

    start_time = df[time_col].min()

    baseline_end_time = start_time + baseline_days * SECONDS_IN_DAY

    baseline_df = df[df[time_col] < baseline_end_time]

    windows = []
    current_start = baseline_end_time

    max_time = df[time_col].max()

    while current_start < max_time:
        current_end = current_start + window_days * SECONDS_IN_DAY

        window_df = df[
            (df[time_col] >= current_start)
            & (df[time_col] < current_end)
        ]

        if len(window_df) > 0:
            windows.append(window_df)

        current_start = current_end

    return baseline_df, windows


def summarize_windows(baseline_df, windows):
    """
    Print quick sanity checks.
    """
    print("====== TIME WINDOW SUMMARY ======")
    print(f"Baseline samples: {len(baseline_df)}")

    for i, w in enumerate(windows):
        print(f"Window {i}: {len(w)} samples")

    print("Total windows:", len(windows))
    print("================================")


if __name__ == "__main__":
    df = load_data()

    baseline, windows = create_time_windows(
        df,
        time_col="TransactionDT",
        baseline_days=7,
        window_days=7,
    )

    summarize_windows(baseline, windows)

    # Optional: quick plot to visualize window sizes
    try:
        import matplotlib.pyplot as plt

        def plot_window_sizes(baseline_df, windows):
            sizes = [len(baseline_df)] + [len(w) for w in windows]
            labels = ["Baseline"] + [f"W{i}" for i in range(len(windows))]

            plt.figure(figsize=(12, 4))
            plt.plot(labels, sizes, marker="o")
            plt.title("Samples per Time Window")
            plt.xlabel("Window")
            plt.ylabel("Number of Samples")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        plot_window_sizes(baseline, windows)
    except Exception:
        pass
