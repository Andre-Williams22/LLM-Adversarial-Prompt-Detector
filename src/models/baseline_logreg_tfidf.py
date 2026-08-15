# models/baseline_logreg_tfidf.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import wandb

# Init W&B
wandb.init(project="adversarial-prompt-detector", entity="awjrs22", name="logreg-baseline")

# Load data
TRAIN_PATH = "./data/preprocessed/train_data.csv"
TEST_PATH = "./data/preprocessed/test_data.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Encode labels
label_map = {label: i for i, label in enumerate(train_df["label"].unique())}

print("Label Map:", label_map)
train_df["label"] = train_df["label"].map(label_map)
test_df["label"] = test_df["label"].map(label_map)

# Build model pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced"))
])

# Train
pipeline.fit(train_df["prompt"], train_df["label"])

# Evaluate
preds = pipeline.predict(test_df["prompt"])
acc = accuracy_score(test_df["label"], preds)
f1 = f1_score(test_df["label"], preds, average="macro")
recall = recall_score(test_df["label"], preds, average="macro")
precision = precision_score(test_df["label"], preds, average="macro")

print("Accuracy:", acc)
print("F1 Score:", f1)
print("Recall:", recall)
print("Precision:", precision)

# Log to W&B
wandb.log({"accuracy": acc, "f1": f1, "recall": recall, "precision": precision})
wandb.finish()
