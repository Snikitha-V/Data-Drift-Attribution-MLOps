# src/drift/detect.py

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict
from tqdm import tqdm


def _safe_divide(a, b):
    return np.where(b == 0, 0, a / b)


def calculate_psi_numeric(
    baseline: pd.Series,
    current: pd.Series,
    bins: int = 10
) -> float:
    """
    Compute Population Stability Index (PSI) for numeric data
    """
    baseline = baseline.dropna()
    current = current.dropna()

    if len(baseline) == 0 or len(current) == 0:
        return np.nan

    # Use baseline quantiles as bin edges
    quantiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(baseline, quantiles)

    # ensure unique breakpoints (np.histogram requires increasing edges)
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) <= 1:
        return 0.0

    baseline_counts, _ = np.histogram(baseline, bins=breakpoints)
    current_counts, _ = np.histogram(current, bins=breakpoints)

    baseline_perc = _safe_divide(baseline_counts, len(baseline))
    current_perc = _safe_divide(current_counts, len(current))

    # clip to avoid log(0) / divide-by-zero issues
    eps = 1e-8
    baseline_perc = np.clip(baseline_perc, eps, 1.0)
    current_perc = np.clip(current_perc, eps, 1.0)

    psi_values = (current_perc - baseline_perc) * np.log(current_perc / baseline_perc)

    return float(np.sum(psi_values))


def calculate_psi_categorical(
    baseline: pd.Series,
    current: pd.Series
) -> float:
    """
    Compute PSI for categorical data by aligning category bins
    """
    baseline = baseline.dropna().astype(str)
    current = current.dropna().astype(str)

    if len(baseline) == 0 or len(current) == 0:
        return np.nan

    base_counts = baseline.value_counts()
    cur_counts = current.value_counts()

    categories = list(set(base_counts.index).union(set(cur_counts.index)))

    base_perc = np.array([base_counts.get(cat, 0) for cat in categories], dtype=float)
    cur_perc = np.array([cur_counts.get(cat, 0) for cat in categories], dtype=float)

    base_perc = _safe_divide(base_perc, base_perc.sum())
    cur_perc = _safe_divide(cur_perc, cur_perc.sum())

    # clip to avoid log(0) / divide-by-zero issues
    eps = 1e-8
    base_perc = np.clip(base_perc, eps, 1.0)
    cur_perc = np.clip(cur_perc, eps, 1.0)

    psi_values = (cur_perc - base_perc) * np.log(cur_perc / base_perc)

    return float(np.sum(psi_values))


def calculate_psi(
    baseline: pd.Series,
    current: pd.Series,
    bins: int = 10
) -> float:
    """
    Wrapper: choose numeric or categorical implementation
    """
    if pd.api.types.is_numeric_dtype(baseline) and pd.api.types.is_numeric_dtype(current):
        return calculate_psi_numeric(baseline, current, bins=bins)
    else:
        return calculate_psi_categorical(baseline, current)


def compute_feature_drift(
    baseline_df: pd.DataFrame,
    window_df: pd.DataFrame,
    exclude_cols=None,
    bins: int = 10
) -> pd.DataFrame:
    """
    Compute PSI + KS per feature for one window.
    Returns a dataframe with columns: feature, psi, ks_pvalue
    """

    if exclude_cols is None:
        exclude_cols = []

    results = []

    cols = [c for c in baseline_df.columns if c not in exclude_cols]

    for col in tqdm(cols, desc="Computing feature drift", unit="feat"):
        base_col = baseline_df[col]
        win_col = window_df[col]

        result = {"feature": col}

        # PSI works for numeric & categorical (after encoding)
        try:
            psi = calculate_psi(base_col, win_col, bins=bins)
        except Exception:
            psi = np.nan

        result["psi"] = psi

        # KS only for numeric
        if pd.api.types.is_numeric_dtype(base_col):
            try:
                ks_stat, ks_p = ks_2samp(
                    base_col.dropna(),
                    win_col.dropna()
                )
            except Exception:
                ks_p = np.nan
            result["ks_pvalue"] = ks_p
        else:
            result["ks_pvalue"] = np.nan

        results.append(result)

    return pd.DataFrame(results)
