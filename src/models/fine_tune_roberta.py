# fine_tune_roberta.py

import os
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
import wandb
from datetime import datetime

# 📁 Paths & Config
MODEL_NAME = "roberta-base"
OUTPUT_DIR = "./outputs/roberta"
TRAIN_PATH = "./data/preprocessed/train_data.csv"
TEST_PATH = "./data/preprocessed/test_data.csv"
NUM_EPOCHS = 3
LR = 2e-5
BATCH_SIZE = 16

# 🕒 Generate unique run name
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"{MODEL_NAME.replace('/', '-')}-{timestamp}"

# 🟡 Initialize W&B for tracking (uses your existing project)
wandb.init(
    project="adversarial-prompt-detector",  # your existing project
    entity="awjrs22",                        # your W&B username
    name=run_name,
    config={
        "model": MODEL_NAME,
        "epochs": NUM_EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE
    }
)

# 🧹 Load and preprocess data
def load_dataset(path):
    df = pd.read_csv(path)
    label_map = {label: i for i, label in enumerate(df["label"].unique())}
    df["label"] = df["label"].map(label_map)
    return Dataset.from_pandas(df[["prompt", "label"]]), label_map

train_dataset, label_map = load_dataset(TRAIN_PATH)
test_dataset, _ = load_dataset(TEST_PATH)
NUM_LABELS = len(label_map)

# 🧪 Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(example["prompt"], truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# 🧠 Load model + PEFT config
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)
model = get_peft_model(model, peft_config)

# ⚙️ Training setup
# args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     save_strategy="epoch",
#     per_device_train_batch_size=BATCH_SIZE,
#     per_device_eval_batch_size=BATCH_SIZE,
#     num_train_epochs=NUM_EPOCHS,
#     learning_rate=LR,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=10,
#     report_to="wandb",
#     run_name=run_name,
#     save_total_limit=1,
# )
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    save_strategy="epoch",
    eval_strategy="epoch",  # Ensure evaluation matches save strategy
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="wandb",
    run_name=run_name,
    save_total_limit=2,  # Increased limit to keep more checkpoints
    load_best_model_at_end=True,  # Added to load the best model
    metric_for_best_model="accuracy",  # Specify metric for best model
    greater_is_better=True,  # Ensure correct comparison
)

# 🧪 Metrics
def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),  # Added precision
        "recall": recall_score(labels, preds, average="macro"),  # Added recall
    }

# 🧑‍🏫 Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# 🔁 Train + Evaluate + Save
trainer.train()
metrics = trainer.evaluate()
print("✅ Final Evaluation:", metrics)

# Log final metrics to W&B
wandb.log(metrics)

# 💾 Save best model for reuse
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model")
trainer.save_model(BEST_MODEL_PATH)
tokenizer.save_pretrained(BEST_MODEL_PATH)
