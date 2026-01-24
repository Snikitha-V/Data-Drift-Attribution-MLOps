# Fraud-Drift-Retraining — Clear Starter

## TL;DR ✅
A minimal, production-minded starter that trains a temporal baseline on IEEE‑CIS Fraud data, detects feature drift (PSI + KS), attributes drift with SHAP, and produces compact window-level signals used by an interpretable retraining policy.

---

## Quick start (no nonsense)
1. Put dataset CSVs into `data/raw/`: `train_transaction.csv`, `train_identity.csv`.
2. Create & activate the project venv and install deps:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

3. Run the pipeline (from repo root):

```bash
python src/data_prep.py      # merge & clean
python src/split.py          # baseline / windows
python src/train_model.py    # train + log to MLflow
python src/drift/run_drift.py   # compute PSI/KS per window
python src/drift/attribution.py # compute SHAP × PSI group attributions
python src/policy/window_signals.py   # synthesize 1-row/window signals
python src/policy/retraining_policy.py # produce retrain decisions
```

> Tip: run `mlflow ui` to inspect runs locally (optional).

---

## Where outputs go
- `reports/` — per-window drift CSVs and attribution CSVs + PNGs (visuals and raw data)
- `policy/window_signals.csv` — compact signals, one row per window (Step 5)
- `policy/window_policy_decisions.csv` — rule-based retraining decisions (Step 6)
- `models/` — exported `lgbm_model.pkl` and `feature_cols.json` (git-ignored)

---

## What the policy does (short)
- Computes 7 window-level signals (max_psi, mean_psi, num_severe, top_group, top_group_share, attribution_entropy, perf_drop).
- Rule-based retrain decision = True iff:
  - max_psi > 0.25 AND num_severe >= 5 AND perf_drop >= 0.01 AND condition persists for ≥ 2 windows.

This is conservative, interpretable, and dataset-agnostic.

---

## Repro & testing
- `scripts/mlflow_list_runs.py` — helper to inspect MLflow runs and artifacts.
- Consider adding a quick CI job to run `src/policy/*` on a small sample and fail on exceptions.

---

## Next steps
- Add CI and tests, or a retraining orchestration step to trigger automated retrains when policy fires.
- I can add these and open a PR if you prefer a feature-branch workflow.

---

Minimal, actionable, and focused — no fluff.
