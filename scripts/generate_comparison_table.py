import json
import pandas as pd

# Load summaries
with open("results/paysim_summary.json", "r") as f:
    paysim = json.load(f)

with open("results/credit_card_summary.json", "r") as f:
    cc = json.load(f)

# Create comparison table
rows = []
for dataset, data in [("PaySim", paysim), ("Credit Card", cc)]:
    for strategy in ["periodic", "psi_only", "policy"]:
        row = {
            "Dataset": dataset,
            "Strategy": strategy.replace("_", " ").title(),
            "Retrain Count": data[strategy]["retrain_count"],
            "Mean AUC": round(data[strategy]["mean_auc"], 3),
            "Std AUC": round(data[strategy]["std_auc"], 3),
        }
        rows.append(row)

df = pd.DataFrame(rows)
print("Cross-Dataset Retraining Policy Comparison")
print("=" * 50)
print(df.to_string(index=False))