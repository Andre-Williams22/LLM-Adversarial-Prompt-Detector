# rank_models.py

import wandb
import pandas as pd

# Config
ENTITY = "awjrs22"
PROJECT = "adversarial-prompt-detector"
METRIC = "f1"  # Change to "accuracy" if you want to rank by accuracy

# Initialize W&B API
api = wandb.Api()

# Fetch all runs
runs = api.runs(f"{ENTITY}/{PROJECT}")

# Extract run info
data = []
for run in runs:
    config = run.config
    summary = run.summary
    data.append({
        "name": run.name,
        "model": config.get("model", "unknown"),
        "accuracy": summary.get("accuracy"),
        "f1": summary.get("f1"),
        "precision": summary.get("precision"),
        "recall": summary.get("recall"),
        "url": run.url
    })

# Convert to DataFrame
df = pd.DataFrame(data)
df = df.sort_values(by=METRIC, ascending=False)

# Show leaderboard
print("\n🏆 Model Leaderboard:")
print(df[["name", "model", "accuracy", "f1", "precision", "recall", "url"]])

# Optionally save
df.to_csv("model_leaderboard.csv", index=False)
print("\n📁 Saved as model_leaderboard.csv")
