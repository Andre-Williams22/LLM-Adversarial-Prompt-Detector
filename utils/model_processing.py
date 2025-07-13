from main import * 
import time 
import mlflow 
import datetime 
import numpy as np
import whylogs as why
from transformers import pipeline
from prometheus_client import Counter, Histogram, start_http_server


prediction_counter = Counter('model_prediction_total', 'Total number of model predictions made')
prediction_latency = Histogram('model_prediction_latency_seconds', 'Latency of model predictions in seconds')

# Start prometheus HTTP Serer
start_http_server(8000) # Expose metrics on port 8000


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

@prediction_latency.time() # Automatically measures latency
def detect_adversarial_prompt(prompt, detectors):
    """ Detects adversarial prompts using multiple detectors. """
    prediction_counter.inc() # Increment prediction counter

    # Log the prompt using whylogs
    profile_result = why.log({"prompt": prompt})
    print(profile_result.view().to_pandas())


    if not isinstance(prompt, (str)):
        raise ValueError("Prompt must be a string.")
    
    scores = []
    for name, detector in detectors.items():
        try:
            if name == "electra_small":
                # Map LABEL_0 and LABEL_1 to meaningful labels
                label_mapping = {"LABEL_0": "safe", "LABEL_1": "adversarial"}
                # Pass the prompt directly as a string
                result = detector(prompt)
                # Ensure proper slicing of nested list structure
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                    result = result[0]  
                # Extract the score for the adversarial label (assuming label index 1 is adversarial)
                adversarial_score = next((item["score"] for item in result if label_mapping.get(item["label"]) == "adversarial"), 0)

            elif name == "tox_bert":
                # Pass the prompt directly as a string
                result = detector(prompt)
                # Handle nested list structure
                if isinstance(result, list) and isinstance(result[0], list):
                    result = result[0]  # Access the first element of the outer list
                # Extract the score for the "toxic" label
                adversarial_score = sum(item["score"] for item in result if item["label"] in ["toxic", "threat", "identity_hate"])
                print(name, "adversarial_score:", adversarial_score)
                print(" ")
            elif name == "offensive_roberta":
                result = detector(prompt)
                if isinstance(result, list) and isinstance(result[0], list):
                    result = result[0]
                adversarial_score = next((item["score"] for item in result if item["label"] == "offensive"), 0.0)
                print(name, "adversarial_score:", adversarial_score)
                print(" ")
            elif name == "bart_mnli":
                # Zero-shot classification with custom labels
                result = detector(prompt, candidate_labels=["toxic", "safe"])
                if isinstance(result, list) and isinstance(result[0], list):
                    result = [item for sublist in result for item in sublist]  # Flatten the list
                adversarial_score = result["scores"][1]
                print(name, "adversarial_score:", adversarial_score)
            else:
                adversarial_score = 0.0 # Default score for unsupported detectors

            scores.append(adversarial_score)
        except Exception as e:
            print(f"Error with detector {name}: {e}")
            scores.append(0.0)  # Default to 0 if detector fails

    # Apply voting mechanism to determine if the prompt is adversarial
    is_adv, reasoning = is_adversarial(scores)

    # MLflow logging
    with mlflow.start_run():
        mlflow.log_param("prompt", prompt)
        mlflow.log_param("reasoning", reasoning)
        for idx, (name, score) in enumerate(zip(detectors.keys(), scores)):
            # Log the model name alongside the score
            mlflow.log_metric(f"{name}_adversarial_score", score)
            mlflow.log_param("model_name", name)
                # If you want to log reasoning, include it in the data dictionary
            profile = why.log({"model_name": name, "prompt": prompt, "adversarial_score": score, "reasoning": reasoning})
            # Write the profile to a specified path
            profile.view().write("whylogs_adversarial_detection_path")

    # Whylogs logging
    # If you want to log reasoning, include it in the data dictionary
    profile = why.log({"prompt": prompt, "scores": scores, "reasoning": reasoning})
    # Write the profile to a specified path
    profile.view().write("whylogs_adversarial_detection_path")

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


