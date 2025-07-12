import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import wandb

# 🟡 Initialize W&åB
wandb.init(project="adversarial-prompt-detector", name="distilbert-hard-test-eval")


# 📁 Paths
MODEL_PATH = "./outputs/electra/best_model"
TEST_PATH = "./data/preprocessed/hard_test.csv"

# 🔢 Load & encode labels
df = pd.read_csv(TEST_PATH)
label_map = {"adversarial": 0, "benign": 1}
df["label"] = df["label"].map(label_map)

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df[["prompt", "label"]])

# 🧪 Tokenize
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
def tokenize(example):
    return tokenizer(example["prompt"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize, batched=True)

# 🧠 Load Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# 🧑‍🏫 Load Trainer for prediction
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

# 🔮 Predict
predictions = trainer.predict(dataset)
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids

# 📈 Evaluation
acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds)
precision = precision_score(labels, preds)
recall = recall_score(labels, preds)

print("📊 Evaluation on Hard Test Set:")
print(f"Accuracy:  {acc:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

wandb.log({
    "hard_test_accuracy": acc,
    "hard_test_f1": f1,
    "hard_test_precision": precision,
    "hard_test_recall": recall
})
wandb.finish()