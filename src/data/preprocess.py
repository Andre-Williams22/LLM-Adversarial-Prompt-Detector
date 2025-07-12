import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 📁 Define folder path
processed_folder = '.p/data/preprocessed'

# Ensure the directory exists
os.makedirs(processed_folder, exist_ok=True)

# 📦 Load all CSVs in the folder
csv_files = [f for f in os.listdir(processed_folder) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {processed_folder}. Please add the necessary files.")

# 🧬 Combine all datasets
datasets = [pd.read_csv(os.path.join(processed_folder, file)) for file in csv_files]
combined_dataset = pd.concat(datasets, ignore_index=True)
print(f"✅ Combined dataset shape: {combined_dataset.shape}")

# 🧹 Standardize labels
combined_dataset["label"] = combined_dataset["label"].replace("safe", "benign")

# 🔍 Check label distribution
print("🔢 Label distribution (original):")
print(combined_dataset["label"].value_counts())

# ❌ Drop duplicate prompts to prevent leakage
combined_dataset = combined_dataset.drop_duplicates(subset=["prompt"])
print(f"🧹 After deduplication: {combined_dataset.shape}")

# 🔁 Stratified train-test split
X = combined_dataset["prompt"]
y = combined_dataset["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🧪 Verify no overlap
overlap = set(X_train) & set(X_test)
print(f"🚨 Overlap between train and test: {len(overlap)}")  # Should be 0

# 📈 Check class balance
print("✅ Train label distribution:")
print(y_train.value_counts(normalize=True))
print("✅ Test label distribution:")
print(y_test.value_counts(normalize=True))

# 💾 Save clean datasets
pd.DataFrame({'prompt': X_train, 'label': y_train}).to_csv(os.path.join(processed_folder, 'train_data.csv'), index=False)
pd.DataFrame({'prompt': X_test, 'label': y_test}).to_csv(os.path.join(processed_folder, 'test_data.csv'), index=False)

print("📁 Clean train/test splits saved to:")
print(f" - {os.path.join(processed_folder, 'train_data.csv')}")
print(f" - {os.path.join(processed_folder, 'test_data.csv')}")
