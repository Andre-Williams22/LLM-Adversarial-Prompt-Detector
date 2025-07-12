# models/fine_tune_electra.py (Manual F1-based Checkpointing)

import os
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig, TaskType
import wandb
from datetime import datetime
import time

# === Custom callback to save best F1 checkpoint ===
class SaveBestF1Callback(TrainerCallback):
    """Save the model checkpoint with highest eval_f1"""
    def __init__(self, output_dir):
        self.best_f1 = 0.0
        self.output_dir = output_dir

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        f1 = metrics.get("eval_f1")
        if f1 is not None and f1 > self.best_f1:
            self.best_f1 = f1
            # Mark that we should save checkpoint at this evaluation
            control.should_save = True
        else:
            control.should_save = False
        return control

# === Start training ===
start = time.time()

# 📁 Paths and config
MODEL_NAME      = "google/electra-small-discriminator"
OUTPUT_DIR      = "./outputs/electra"
TRAIN_PATH      = "./data/preprocessed/train_data.csv"
VALID_PATH      = "./data/preprocessed/valid_data.csv"
HARD_TEST_PATH  = "./data/preprocessed/hard_test.csv"
NUM_EPOCHS      = 2
LR              = 1e-5
BATCH_SIZE      = 16
SAVE_STEPS      = 1000
LABEL_SMOOTHING = 0.1

# 🕒 W&B init
run_name = f"electra-f1-checkpoint-{datetime.now():%Y%m%d-%H%M%S}"
wandb.init(
    project="adversarial-prompt-detector",
    entity="awjrs22",
    name=run_name,
    config={"model": MODEL_NAME, "epochs": NUM_EPOCHS, "lr": LR, "batch_size": BATCH_SIZE}
)

# 🧹 Load datasets
def load_dataset(path):
    df = pd.read_csv(path)
    label_map = {label: i for i, label in enumerate(df["label"].unique())}
    df["label"] = df["label"].map(label_map)
    return Dataset.from_pandas(df[["prompt", "label"]]), label_map

train_ds, label_map = load_dataset(TRAIN_PATH)
valid_ds, _       = load_dataset(VALID_PATH)
NUM_LABELS        = len(label_map)

# 🧪 Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(ex): return tokenizer(ex["prompt"], truncation=True, padding="max_length")
train_ds = train_ds.map(tokenize, batched=True)
valid_ds = valid_ds.map(tokenize, batched=True)

# 🧠 Model setup
torch_dtype = "float32"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
peft_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1,
    bias="none", task_type=TaskType.SEQ_CLS,
    target_modules=["query", "value"]
)
model = get_peft_model(model, peft_cfg)

# ⚙️ TrainingArguments (no load_best_model...)
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    weight_decay=0.1,
    max_grad_norm=1.0,
    label_smoothing_factor=LABEL_SMOOTHING,
    logging_dir="./logs",
    logging_steps=10,
    report_to="wandb",
    run_name=run_name,
    save_total_limit=3,
    logging_first_step=True,
    label_names=["label"]
)

# 📈 Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    loss   = pred.loss
    # Log eval_loss to W&B manually
    wandb.log({"eval_loss": loss})
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
        "f1": f1_score(labels, preds, average="macro")
    }

# 🧑‍🏫 Trainer with F1 callback
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[SaveBestF1Callback(OUTPUT_DIR)]
)

# 🔁 Train & Eval
trainer.train()
val_metrics = trainer.evaluate()
print("✅ Validation:", val_metrics)

# 🧪 Hard Test Eval
hard_df = pd.read_csv(HARD_TEST_PATH)
test_ds = Dataset.from_pandas(hard_df)
test_ds = test_ds.map(tokenize, batched=True)
test_ds = test_ds.map(lambda ex: {"label": label_map[ex["label"]]}, batched=False)
hard_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="hard_test")
print("🧪 Hard Test:", hard_metrics)

# 💾 Model saved by callback; final path:
print("🥇 Best F1 checkpoint in:", OUTPUT_DIR)
print("🕒 Total time:", time.time() - start)
