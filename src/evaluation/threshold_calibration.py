# scripts/calibrate_threshold.py

import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
import torch

# ==== Config ==== 
MODEL_DIR = "outputs/electra/best_model"  # path to your fine-tuned model checkpoint
TEST_CSV  = "data/preprocessed/hard_test.csv"  # CSV with 'prompt','label' columns
MAX_LENGTH = 128  # tokenizer max length

# ==== Load model and tokenizer ==== 
print("Loading model from:", MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ==== Prepare dataset ==== 
df = pd.read_csv(TEST_CSV)
label_map = {"adversarial": 0, "benign": 1}
if df['label'].dtype == object:
    df['label'] = df['label'].map(label_map)

# HuggingFace Dataset 
ds = Dataset.from_pandas(df[['prompt','label']])

# Tokenize

def tokenize_fn(batch):
    return tokenizer(batch['prompt'], padding='max_length', truncation=True, max_length=MAX_LENGTH)

ds = ds.map(tokenize_fn, batched=True)
ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])

# ==== Use Trainer to get probabilities ==== 
trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer))
pred_out = trainer.predict(ds)

# Get predicted probabilities for positive class (benign=1)
logits = pred_out.predictions
probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:,1].numpy()
labels = pred_out.label_ids

# ==== Precision-Recall Curve & Best Threshold ==== 
precision, recall, thresholds = precision_recall_curve(labels, probs)
f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
best_idx = np.nanargmax(f1_scores[:-1])  # exclude last threshold
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"Best threshold for max F1: {best_threshold:.4f}")
print(f"Max F1 at this threshold:  {best_f1:.4f}")

# ==== Compute metrics at best threshold ==== 
bin_preds = (probs >= best_threshold).astype(int)
acc      = (bin_preds == labels).mean()
prec_thr = precision_score(labels, bin_preds)
recall_thr = recall_score(labels, bin_preds)
f1_thr   = f1_score(labels, bin_preds)

print("\nMetrics at threshold {:.4f}:".format(best_threshold))
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec_thr:.4f}")
print(f"  Recall   : {recall_thr:.4f}")
print(f"  F1 Score : {f1_thr:.4f}")

# ==== Optional: save threshold for deployment ==== 
config_path = os.path.join(MODEL_DIR, 'threshold.txt')
with open(config_path, 'w') as f:
    f.write(str(best_threshold))
print(f"Saved best threshold to {config_path}")
