import pandas as pd

# Load the TSV (from your merged file)
df = pd.read_csv("data/processed/wildjailbreak_combined.csv")


print("Initial DataFrame shape:", df.shape)

print(df["label"].value_counts())