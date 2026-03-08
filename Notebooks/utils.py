import json
import os

def save_baseline_stats(train_inputs):
    os.makedirs("reports", exist_ok=True)
    stats={}
    for col in train_inputs.columns:
        stats[col]={
            "mean": float(train_inputs[col].mean()),
            "std": float(train_inputs[col].std()),
            "min": float(train_inputs[col].min()),
            "max": float(train_inputs[col].max())
        }
    with open("reports/baseline_stats.json","w") as f:
        json.dump(stats, f, indent=4)