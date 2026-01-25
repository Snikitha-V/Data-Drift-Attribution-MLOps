import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipelines.run_dataset_pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline(
        dataset_name="credit_card",
        data_path="data/raw/creditcard.csv",
        feature_groups_path="src/features/feature_groups_cc.yaml",
        preprocess="cc",
        baseline_days=1,
        window_days=1,
    )
