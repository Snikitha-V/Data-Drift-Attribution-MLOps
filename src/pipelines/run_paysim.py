import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipelines.run_dataset_pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline(
        dataset_name="paysim",
        data_path="data/raw/PS_20174392719_1491204439457_log.csv",
        feature_groups_path="src/features/feature_groups_paysim.yaml",
        preprocess="paysim",
        baseline_days=7,
        window_days=7,
    )
