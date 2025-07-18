# scripts/log_model_mlflow.py
import os
from dotenv import load_dotenv

import mlflow
import mlflow.pytorch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

load_dotenv()

# No set_tracking_uri() → default is file://./mlruns
mlflow.set_experiment("adversarial_prompt_detector")

with mlflow.start_run(run_name="register_adapter_checkpoint"):
    # load base + adapter
    base = "google/electra-small-discriminator"
    model = AutoModelForSequenceClassification.from_pretrained(base)
    model = PeftModel.from_pretrained(model, "outputs/electra/best_model")
    tokenizer = AutoTokenizer.from_pretrained("outputs/electra/best_model")

    # log it all
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="electra_adversarial_detector",
        registered_model_name="Electra_Adversarial_Detector",
    )
    print("✅ Model registered in MLflow!")
