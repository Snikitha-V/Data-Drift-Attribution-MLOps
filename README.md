# Fraud Drift Retraining — Starter

This repository contains a minimal, runnable baseline for the IEEE-CIS Fraud Detection dataset. It trains a temporal baseline, logs runs with MLflow, and is structured for adding drift detection and retraining later.

Quick start

1. Place the two required files in `data/raw/`:

- train_transaction.csv
- train_identity.csv

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the pipeline steps (from project root):

```bash
python src/data_prep.py
python src/split.py
mlflow ui   # optional: view runs
python src/train_model.py
python src/evaluate.py
```

Files of interest

- [src/config.py](src/config.py): central configuration
- [src/data_prep.py](src/data_prep.py): load/merge/clean
- [src/split.py](src/split.py): temporal split
- [src/train_model.py](src/train_model.py): train baseline + MLflow
- [src/evaluate.py](src/evaluate.py): test evaluation

If you want, I can run the preprocessing and split steps now (I already created `data/processed/`).
