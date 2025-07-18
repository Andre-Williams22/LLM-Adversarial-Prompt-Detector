import pandas as pd

# Load and process eval.tsv
df_eval = pd.read_csv("data/raw/eval.tsv", sep="\t")
df_eval = df_eval.rename(columns={"adversarial": "prompt"})
df_eval["label"] = df_eval["label"].map({0: "safe", 1: "adversarial"})

# Load and process train.tsv
df_train = pd.read_csv("data/raw/train.tsv", sep="\t")

# The "adversarial" column here is actually under different headers (vanilla, adversarial, completion)
# So we treat "adversarial" column as the prompt input
df_train = df_train.rename(columns={"adversarial": "prompt"})
df_train["label"] = "adversarial"  # All of train.tsv entries are harmful prompts

# Some entries may have NaNs in "prompt" due to misalignment — drop those
df_train = df_train[df_train["prompt"].notnull()]

# Align both
df_eval_clean = df_eval[["prompt", "label"]]
df_train_clean = df_train[["prompt", "label"]]

# Combine and shuffle
df_combined = pd.concat([df_eval_clean, df_train_clean], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to processed folder
df_combined.to_csv("data/processed/wildjailbreak_cleaned.csv", index=False)
print("✅ Combined and cleaned WildJailbreak dataset saved to wildjailbreak_cleaned.csv")
