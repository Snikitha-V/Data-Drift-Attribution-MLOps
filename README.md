# Drift Attribution & Intelligent Retraining Trigger for Fraud Models

This project presents an end-to-end MLOps system that goes beyond drift detection to explain drift, quantify its impact, and make intelligent retraining decisions. Unlike naïve retrain-on-drift approaches, the system retrains only when drift is harmful, persistent, and performance-degrading.

The core contribution is a generalizable retraining policy trained on one dataset and validated unchanged across multiple fraud domains.

## Key Contributions

- Statistical Drift Detection using PSI + KS over rolling time windows
- Root-Cause Drift Attribution via PSI × SHAP importance
- Window-Level Signal Compression for decision readiness
- Interpretable Intelligent Retraining Policy (rule-based, persistence-aware)
- Cross-Dataset Generalization without re-tuning
- Stress Test + Ablation Study for robustness and design validation

## Datasets Used

| Dataset                  | Purpose                          |
|--------------------------|----------------------------------|
| IEEE-CIS Fraud Detection | Policy development & primary experiments |
| PaySim                   | High-drift stress testing & generalization |
| Credit Card Fraud (European) | Low-drift stability validation |

All datasets are treated as tabular, temporally ordered streams.

## System Overview

```
Data → Baseline Model
     → Time Windows
     → Drift Detection (PSI, KS)
     → Drift Attribution (SHAP × PSI)
     → Window Signal Compression
     → Intelligent Retraining Policy
     → Cross-Dataset Evaluation
```

The baseline window is fixed. All monitoring windows are compared against it, mimicking production monitoring.

## Project Structure

```
fraud-drift-retraining/
├── data/
│   ├── raw/                # Original datasets
│   └── processed/          # Cleaned & split data
├── drift/
│   ├── windows.py          # Time window construction
│   ├── detect.py           # PSI + KS drift detection
│   └── attribution.py      # SHAP-based drift attribution
├── policy/
│   ├── window_signals.py   # Window-level signal compression
│   └── retraining_policy.py# Intelligent retraining rules
├── pipelines/
│   ├── run_ieee.py
│   ├── run_paysim.py
│   └── run_credit_card.py
├── reports/
│   ├── drift_*.csv/png
│   ├── attribution_*.csv/png
│   └── summaries.json
└── README.md
```

## Drift Detection

**PSI (Population Stability Index)**: Quantifies magnitude of distributional shift. PSI > 0.25 flagged as severe.

**KS Test**: Tests statistical significance for numeric features.

Drift is computed per feature per window against a fixed baseline.

*Results*: Generated drift reports (e.g., `drift_window_0.csv`) showing PSI and KS values for each feature across windows.

## Drift Attribution

Drift alone does not imply action. We compute drift impact as:

`Drift Impact(feature) = PSI × |SHAP importance|`

Impacts are aggregated into semantic feature groups (e.g., monetary, behavioral, identity), enabling root-cause analysis.

Outputs include:
- Per-window attribution CSVs
- Attribution heatmaps
- Temporal attribution timeline

*Results*: Attribution reports (e.g., `attribution_window_0.csv`) with group-level impacts, identifying which feature categories are most affected by drift.

## Window-Level Signal Compression

Rich attribution outputs are compressed into one row per window, e.g.:

| window | max_psi | num_severe | top_group | entropy | perf_drop |
|--------|---------|------------|-----------|---------|-----------|

These signals are low-dimensional, dataset-agnostic, and suitable for automated decision logic.

*Results*: `window_signals.csv` with compressed signals for all windows, ready for policy application.

## Intelligent Retraining Policy

An interpretable, frozen rule-based policy decides when to retrain:

**RETRAIN if:**
- max_psi > 0.25
- num_severe ≥ 5
- perf_drop ≥ 0.01
- conditions persist ≥ 2 consecutive windows

Key properties: Avoids retraining on harmless drift, anchored to performance impact, generalizes across datasets, serves as a safe fallback.

*Results*: `window_policy_decisions.csv` with retrain flags per window. In IEEE-CIS, policy triggered retrains only when necessary.

## Cross-Dataset Validation

The same policy (unchanged) is applied to PaySim and Credit Card Fraud.

### Results Summary

**PaySim** (High-drift scenario):
| Strategy  | Retrains | Mean AUC | Std AUC |
|-----------|----------|----------|---------|
| Periodic  | 10       | 0.94     | 0.07    |
| PSI-only  | 0        | 0.96     | 0.05    |
| Policy    | 0        | 0.96     | 0.05    |

**Credit Card Fraud** (Low-drift scenario):
| Strategy  | Retrains | Mean AUC | Std AUC |
|-----------|----------|----------|---------|
| Periodic  | 0        | 0.81     | 0.00    |
| PSI-only  | 1        | 0.81     | 0.00    |
| Policy    | 0        | 0.81     | 0.00    |

*Conclusion*: The policy achieves equal or better performance with fewer retrains, demonstrating robust generalization.

## Stress Test & Ablation

### Stress Test
A strong distributional shift was injected into high-impact monetary features without performance degradation. The policy correctly suppressed retraining, demonstrating robustness to false positives.

*Results*: No retrain triggered, confirming conservative behavior.

### Ablation Study
| Variant          | Retrains | Mean AUC |
|------------------|----------|----------|
| Full policy      | 0        | 0.96     |
| No persistence   | 3        | 0.95     |
| No attribution   | 5        | 0.94     |

*Results*: Removing persistence or attribution increases unnecessary retraining, validating each component's necessity.

## Key Takeaways

- Drift ≠ retrain
- Attribution matters
- Persistence matters
- Performance impact is essential
- One policy can generalize across domains

## Status

✅ End-to-end system complete  
✅ Stress-tested & ablated  
✅ Cross-dataset validated  
✅ Conference / industry-ready

