import time 
import mlflow 
import datetime 
import numpy as np
from transformers import pipeline
from prometheus_client import Counter, Histogram

# Don't start HTTP server here - it's handled by FastAPI

prediction_counter = Counter('model_prediction_total', 'Total predictions per model', ['model_name'])
model_latency = Histogram('model_latency_seconds', 'Latency of each model', ['model_name'])
# global detectors dict 
detectors = {}

def load_models():
    """
    Load models in a separate thread to avoid blocking the server startup.
    """
    global detectors
    detectors = {
        "electra_small": pipeline("text-classification", model="google/electra-small-discriminator", return_all_scores=True),
        "tox_bert": pipeline("text-classification", model="unitary/toxic-bert", return_all_scores=True),
        "offensive_roberta": pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-offensive", return_all_scores=True),
        "bart_mnli": pipeline("zero-shot-classification", model="facebook/bart-large-mnli"),
    }

    print("Models loaded successfully!")
    return detectors 

def predict_with_electra(prompt, detector, name):
    """Predict using Electra Small model."""
    start_time = time.time()
    # Map LABEL_0 and LABEL_1 to meaningful labels
    label_mapping = {"LABEL_0": "safe", "LABEL_1": "adversarial"}
    # Pass the prompt directly as a string
    prediction = detector(prompt)
    latency = time.time() - start_time

    # Log Prometheus metrics
    prediction_counter.labels(model_name="electra_small").inc()
    model_latency.labels(model_name="electra_small").observe(latency)

    # extract prediction 
     # Ensure proper slicing of nested list structure
    if isinstance(prediction, list) and len(prediction) > 0 and isinstance(prediction[0], list):
        prediction = prediction[0]
    # Extract the score for the adversarial label (assuming label index 1 is adversarial)
    adversarial_score = next((item["score"] for item in prediction if label_mapping.get(item["label"]) == "adversarial"), 0)
    print(name, "adversarial_score:", adversarial_score)
    print(" ")

    # Log MLflow metrics in nested run (with error handling)
    try:
        with mlflow.start_run(nested=True):
            mlflow.log_param("model_name", "electra_small")
            mlflow.log_metric("latency", latency)
            mlflow.log_metric("detection_score", adversarial_score)
            mlflow.log_param("is_adversarial", adversarial_score > 0.5)
            mlflow.log_param("prompt", prompt)  # Log the prompt
    except Exception as e:
        print(f"MLflow logging failed for electra_small: {e}")

    return adversarial_score

def predict_with_tox_bert(prompt, detector, name):
    """Predict using Toxic BERT model."""
    start_time = time.time()
    prediction = detector(prompt)
    latency = time.time() - start_time
    # Log Prometheus metrics
    prediction_counter.labels(model_name="tox_bert").inc()
    model_latency.labels(model_name="tox_bert").observe(latency)
    # Extract prediction Ensure proper slicing of nested list structure
    # Handle nested list structure
    if isinstance(prediction, list) and len(prediction) > 0 and isinstance(prediction[0], list):
        prediction = prediction[0]
    # Extract the score for the "toxic" label
    adversarial_score = sum(item["score"] for item in prediction if item["label"] in ["toxic", "threat", "identity_hate"])
    print(name, "adversarial_score:", adversarial_score)
    print(" ")
    
    # Log MLflow metrics in nested run (with error handling)
    try:
        with mlflow.start_run(nested=True):
            mlflow.log_param("model_name", "tox_bert")
            mlflow.log_metric("latency", latency)
            mlflow.log_metric("detection_score", adversarial_score)
            mlflow.log_param("is_adversarial", adversarial_score > 0.5)
            mlflow.log_param("prompt", prompt)  # Log the prompt
    except Exception as e:
        print(f"MLflow logging failed for tox_bert: {e}")

    return adversarial_score

def predict_with_offensive_roberta(prompt, detector, name):
    """Predict using Offensive Roberta model."""
    start_time = time.time()
    prediction = detector(prompt)
    latency = time.time() - start_time

    # Log Prometheus metrics
    prediction_counter.labels(model_name="offensive_roberta").inc()
    model_latency.labels(model_name="offensive_roberta").observe(latency)

    # Extract prediction Ensure proper slicing of nested list structure
    if isinstance(prediction, list) and isinstance(prediction[0], list):
        prediction = prediction[0]
    adversarial_score = next((item["score"] for item in prediction if item["label"] == "offensive"), 0.0)
    print(name, "adversarial_score:", adversarial_score)
    print(" ")

    # Log MLflow metrics in nested run (with error handling)
    try:
        with mlflow.start_run(nested=True):
            mlflow.log_param("model_name", "offensive_roberta")
            mlflow.log_metric("latency", latency)
            mlflow.log_metric("detection_score", adversarial_score)
            mlflow.log_param("is_adversarial", adversarial_score > 0.5)
            mlflow.log_param("prompt", prompt)  # Log the prompt
    except Exception as e:
        print(f"MLflow logging failed for offensive_roberta: {e}")

    return adversarial_score

def predict_with_bart_mnli(prompt, detector, name):
    """Predict using BART MNLI model."""
    start_time = time.time()
    prediction = detector(prompt, candidate_labels=["toxic", "safe"])
    latency = time.time() - start_time

    # Log Prometheus metrics
    prediction_counter.labels(model_name="bart_mnli").inc()
    model_latency.labels(model_name="bart_mnli").observe(latency)

    # Ensure proper slicing of nested list structure
    if isinstance(prediction, list) and len(prediction) > 0 and isinstance(prediction[0], list):
        prediction = [item for sublist in prediction for item in sublist]  # Flatten the list
    adversarial_score = prediction["scores"][1]
    print(name, "adversarial_score:", adversarial_score)
    print("")

    # Log MLflow metrics in nested run (with error handling)
    try:
        with mlflow.start_run(nested=True):
            mlflow.log_param("model_name", "bart_mnli")
            mlflow.log_metric("latency", latency)
            mlflow.log_metric("detection_score", adversarial_score)
            mlflow.log_param("is_adversarial", adversarial_score > 0.5)
            mlflow.log_param("prompt", prompt)  # Log the prompt
    except Exception as e:
        print(f"MLflow logging failed for bart_mnli: {e}")

    return adversarial_score

@model_latency.labels(model_name="detect_adversarial_prompt").time() # Automatically measures latency
def detect_adversarial_prompt(prompt, detectors):
    """ Detects adversarial prompts using multiple detectors. """
    prediction_counter.labels(model_name="detect_adversarial_prompt").inc() # Increment prediction counter

    if not isinstance(prompt, (str)):
        raise ValueError("Prompt must be a string.")
    
    scores = []
    for name, detector in detectors.items():
        try:
            if name == "electra_small":
                # predict using Electra Small model
                scores.append(predict_with_electra(prompt=prompt, detector=detector, name=name))
            elif name == "tox_bert":
                # Pass the prompt directly as a string
                scores.append(predict_with_tox_bert(prompt=prompt, detector=detector, name=name))
            elif name == "offensive_roberta":
                # Pass the prompt directly as a string
                scores.append(predict_with_offensive_roberta(prompt=prompt, detector=detector, name=name))
            elif name == "bart_mnli":
                # Zero-shot classification with custom labels
                scores.append(predict_with_bart_mnli(prompt=prompt, detector=detector, name=name))
            else:
                # Default score for unsupported detectors
                scores.append(0.0)
                
        except Exception as e:
            print(f"Error with detector {name}: {e}")
            scores.append(0.0)  # Default to 0 if detector fails

    # Apply voting mechanism to determine if the prompt is adversarial
    is_adv, reasoning = is_adversarial(scores)

    # Log only the final voting decision and reasoning (with error handling)
    try:
        with mlflow.start_run(nested=True):
            mlflow.log_param("prompt", prompt)
            mlflow.log_param("final_reasoning", str(reasoning["decision"]))
            mlflow.log_param("is_adversarial", is_adv)
            mlflow.log_metric("final_adversarial_decision", float(is_adv))
    except Exception as e:
        print(f"MLflow logging failed for final decision: {e}")

    return is_adv, reasoning


def is_adversarial(scores, threshold=0.5):
    """
    Determines if any score exceeds the given threshold, indicating adversarial behavior.
    Electra is given less weight, while other models are prioritized for confident decisions.

    Args:
        scores (list[float]): List of scores to evaluate.
        threshold (float): The threshold value to classify as adversarial.

    Returns:
        tuple: (bool, dict) - Classification result and reasoning details.
    """
    vote = []
    reasoning = {"scores": scores, "threshold": threshold, "decision": None}

    # Check Electra model score (less reliable, lower weight)
    electra_score = scores[0]
    if electra_score > threshold:
        vote.append(electra_score)
        reasoning["electra_vote"] = True
    else:
        reasoning["electra_vote"] = False

    # Check other models' scores
    for i in range(1, len(scores)):
        model_score = scores[i]
        if model_score > threshold:
            reasoning["decision"] = True
            reasoning["reason"] = f"Model {i} exceeded threshold confidently."
            return True, reasoning

        if model_score > 0.4:
            vote.append(model_score)

    # Final decision based on votes
    reasoning["votes"] = len(vote)
    if len(vote) > 1:
        # If Electra supports the majority decision, classify as adversarial
        reasoning["decision"] = True
        reasoning["reason"] = "Majority decision with Electra support."
        return True, reasoning

    reasoning["decision"] = False
    reasoning["reason"] = "Insufficient votes for adversarial classification."
    return False, reasoning


