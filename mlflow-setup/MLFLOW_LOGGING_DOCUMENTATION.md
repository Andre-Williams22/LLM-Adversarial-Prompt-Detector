# MLflow Logging Implementation for Adversarial Prompt Detection

## Overview

This document describes the comprehensive MLflow logging implementation for the adversarial prompt detection system. Every prediction from individual models and ensemble decisions is now properly logged to MLflow for full traceability and analysis.

## Implementation Summary

### ✅ What Was Fixed

1. **Synchronous Detection Logging**: Added complete MLflow logging to `detect_adversarial_sync()` method
2. **Experiment Consistency**: Fixed experiment name inconsistency - all runs now use `adversarial_detection_system`
3. **Individual Model Tracking**: Each model (keyword, toxic, hate, safety) gets its own MLflow run
4. **Ensemble Decision Tracking**: Final voting decisions are logged with detailed voting mechanics
5. **Timing Precision**: Individual model inference times are tracked separately
6. **Error Handling**: Fallback scenarios are also logged to MLflow

### 🏗️ Logging Structure

```
📁 adversarial_detection_system (Single Experiment)
├── 🔍 Individual Model Runs
│   ├── keyword_detector_YYYY-MM-DD_HH-MM-SS
│   ├── toxic_classifier_YYYY-MM-DD_HH-MM-SS
│   ├── hate_classifier_YYYY-MM-DD_HH-MM-SS
│   └── safety_classifier_YYYY-MM-DD_HH-MM-SS
│
└── 🎯 Ensemble Decision Runs
    └── ensemble_decision_YYYY-MM-DD_HH-MM-SS
```

### 📊 Individual Model Run Data

Each individual model logs:

**Parameters:**
- `model_name`: Name of the specific model (e.g., "keyword_detector")
- `timestamp`: Execution timestamp
- `adversarial`: Boolean decision for this model
- `input_length`: Length of input text
- `input_hash`: Hash of input (for tracking without storing sensitive data)
- `prompt`: First 500 characters of input prompt
- `sensitivity_mode`: Current sensitivity configuration
- `detection_type`: "rule_based" or "ml_model"

**Metrics:**
- `score`: Model's confidence score (0.0-1.0)
- `inference_time_ms`: Individual model inference time in milliseconds
- `threshold`: Decision threshold used
- `above_threshold`: Binary indicator if score exceeded threshold

**Tags:**
- `model_type`: "individual"
- `detection_outcome`: "adversarial" or "safe"
- `model_category`: Model type category

### 🎯 Ensemble Decision Run Data

Ensemble decisions log:

**Parameters:**
- `model_name`: "ensemble_voting_system"
- `timestamp`: Execution timestamp
- `adversarial`: Final ensemble decision
- `input_length`: Length of input text
- `input_hash`: Hash of input
- `prompt`: First 500 characters of input prompt
- `sensitivity_mode`: Current sensitivity configuration
- `decision_reason`: Explanation of why this decision was made
- `winning_mechanism`: Which voting mechanism triggered the decision
- Voting configuration parameters (thresholds, etc.)

**Metrics:**
- `total_inference_time_ms`: Total time for entire detection process
- `max_individual_score`: Highest score from any individual model
- `early_exit`: Whether detection exited early (keyword-based)
- Individual model scores: `keyword_score`, `toxic_score`, `hate_score`, `safety_score`
- Threshold indicators: `keyword_above_threshold`, etc.
- Voting mechanism triggers: `voting_high_confidence_trigger`, etc.

**Tags:**
- `model_type`: "ensemble"
- `detection_outcome`: "adversarial" or "safe"
- `decision_type`: "early_exit" or "full_ensemble"

**Artifacts:**
- `ensemble_details.json`: Detailed voting information and scores

## Usage Examples

### Viewing Recent Runs

```python
# Run the summary script to see recent activity
python mlflow_logging_summary.py
```

### MLflow UI

```bash
# Start MLflow UI (using port 5001 since 5000 is used by macOS ControlCenter)

# Open browser to:
# http://localhost:5001
```

### Filtering Runs

In MLflow UI, you can filter by:
- **Individual models**: `tags.model_type = "individual"`
- **Ensemble decisions**: `tags.model_type = "ensemble"`
- **Adversarial detections**: `tags.detection_outcome = "adversarial"`
- **Safe detections**: `tags.detection_outcome = "safe"`
- **Early exits**: `tags.decision_type = "early_exit"`
- **Specific model**: `params.model_name = "keyword_detector"`

## Benefits

1. **Complete Traceability**: Every prediction is logged with full context
2. **Performance Analysis**: Individual model timing and scoring data
3. **Decision Transparency**: Detailed voting mechanism information
4. **Error Tracking**: Fallback scenarios are captured
5. **Comparative Analysis**: Easy comparison between models and configurations
6. **Production Monitoring**: Real-time tracking of detection performance

## Code Integration

Both detection methods now include comprehensive logging:

- `detect_adversarial_sync()`: Synchronous detection with MLflow logging
- `detect_adversarial_fast()`: Asynchronous detection with MLflow logging (existing)

The logging is automatic and doesn't require any changes to the API usage.

## Verification

The implementation has been tested and verified:
- ✅ Individual model runs are logged
- ✅ Ensemble decisions are logged
- ✅ Both sync and async methods include logging
- ✅ Experiment consistency maintained
- ✅ Proper timing and scoring data captured
- ✅ Error scenarios are handled and logged

## Next Steps

1. **Production Monitoring**: Set up alerts based on MLflow metrics
2. **Performance Optimization**: Use MLflow data to identify slow models
3. **A/B Testing**: Compare different sensitivity modes using MLflow experiments
4. **Model Drift Detection**: Monitor score distributions over time
