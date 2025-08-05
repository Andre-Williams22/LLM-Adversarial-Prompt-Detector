#!/usr/bin/env python3
"""
MLflow Logging Summary Script
Shows the complete logging structure for adversarial prompt detection
"""

import mlflow
import mlflow.tracking
import pandas as pd
from datetime import datetime
import os

def show_mlflow_experiments():
    """Display all MLflow experiments"""
    print("📊 MLflow Experiments Overview")
    print("=" * 50)
    
    try:
        experiments = mlflow.search_experiments()
        for exp in experiments:
            print(f"Experiment: {exp.name}")
            print(f"  ID: {exp.experiment_id}")
            print(f"  Lifecycle Stage: {exp.lifecycle_stage}")
            print(f"  Artifact Location: {exp.artifact_location}")
            print()
    except Exception as e:
        print(f"Error accessing experiments: {e}")

def show_recent_runs(experiment_name="adversarial_detection_system", max_results=10):
    """Show recent runs from the adversarial detection experiment"""
    print(f"🔍 Recent Runs from '{experiment_name}' Experiment")
    print("=" * 60)
    
    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"❌ Experiment '{experiment_name}' not found!")
            return
        
        print(f"✅ Found experiment: {experiment.name} (ID: {experiment.experiment_id})")
        print()
        
        # Search for recent runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"]
        )
        
        if runs.empty:
            print("❌ No runs found in this experiment!")
            return
        
        print(f"📈 Showing {len(runs)} most recent runs:")
        print("-" * 60)
        
        # Display run summary
        for idx, (_, run) in enumerate(runs.iterrows(), 1):
            print(f"{idx}. Run: {run.get('tags.mlflow.runName', 'Unknown')}")
            print(f"   Model: {run.get('params.model_name', 'Unknown')}")
            print(f"   Decision: {'🚨 ADVERSARIAL' if run.get('params.adversarial') == 'True' else '✅ SAFE'}")
            print(f"   Score: {run.get('metrics.score', 'N/A')}")
            print(f"   Time: {run.get('metrics.inference_time_ms', 'N/A')} ms")
            print(f"   Started: {run['start_time']}")
            print(f"   Status: {run['status']}")
            print()
            
        # Show model type breakdown
        model_types = runs['params.model_name'].value_counts()
        print("📊 Model Type Breakdown:")
        for model_type, count in model_types.items():
            print(f"   {model_type}: {count} runs")
        print()
        
        # Show detection outcomes
        outcomes = runs['params.adversarial'].value_counts()
        print("🎯 Detection Outcomes:")
        for outcome, count in outcomes.items():
            status = "🚨 ADVERSARIAL" if outcome == "True" else "✅ SAFE"
            print(f"   {status}: {count} runs")
        
    except Exception as e:
        print(f"❌ Error accessing runs: {e}")

def show_logging_structure():
    """Show the complete MLflow logging structure"""
    print("🏗️  MLflow Logging Structure")
    print("=" * 50)
    
    structure = """
📁 adversarial_detection_system (Experiment)
├── 🔍 Individual Model Runs
│   ├── keyword_detector_YYYY-MM-DD_HH-MM-SS
│   │   ├── 📊 Metrics: score, inference_time_ms, threshold, above_threshold
│   │   ├── 📋 Params: model_name, timestamp, adversarial, input_length, 
│   │   │              input_hash, prompt, sensitivity_mode, detection_type
│   │   └── 🏷️  Tags: model_type=individual, detection_outcome, model_category
│   │
│   ├── toxic_classifier_YYYY-MM-DD_HH-MM-SS
│   │   ├── 📊 Metrics: score, inference_time_ms, threshold, above_threshold
│   │   ├── 📋 Params: model_name, timestamp, adversarial, input_length,
│   │   │              input_hash, prompt, sensitivity_mode, detection_type
│   │   └── 🏷️  Tags: model_type=individual, detection_outcome, model_category
│   │
│   ├── hate_classifier_YYYY-MM-DD_HH-MM-SS
│   │   └── (same structure as above)
│   │
│   └── safety_classifier_YYYY-MM-DD_HH-MM-SS
│       └── (same structure as above)
│
└── 🎯 Ensemble Decision Runs
    └── ensemble_decision_YYYY-MM-DD_HH-MM-SS
        ├── 📊 Metrics: total_inference_time_ms, max_individual_score,
        │               early_exit, keyword_score, toxic_score, hate_score,
        │               safety_score, keyword_above_threshold, 
        │               toxic_above_threshold, hate_above_threshold,
        │               safety_above_threshold, voting_*
        ├── 📋 Params: model_name=ensemble_voting_system, timestamp,
        │              adversarial, input_length, input_hash, prompt,
        │              sensitivity_mode, decision_reason, winning_mechanism,
        │              high_confidence_threshold, weak_signals_threshold,
        │              majority_threshold, weighted_threshold
        ├── 🏷️  Tags: model_type=ensemble, detection_outcome, decision_type
        └── 📄 Artifacts: ensemble_details.json
    """
    
    print(structure)
    
    print("\n🔧 Key Features:")
    print("• Each model prediction gets its own MLflow run")
    print("• Ensemble decisions are logged separately with voting details")
    print("• All runs use the same experiment for easy comparison")
    print("• Individual timing and scoring data for each model")
    print("• Complete traceability from input to final decision")
    print("• Proper tagging for filtering and analysis")

def main():
    """Main function to show MLflow logging overview"""
    print("🎯 MLflow Logging Verification for Adversarial Prompt Detection")
    print("=" * 70)
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Show experiments
    show_mlflow_experiments()
    
    # Show recent runs
    show_recent_runs()
    
    # Show logging structure
    show_logging_structure()
    
    print("\n" + "=" * 70)
    print("✅ MLflow logging verification complete!")
    print("\n💡 To view in MLflow UI:")
    print("   mlflow ui --port 5000")
    print("   Then open: http://localhost:5000")

if __name__ == "__main__":
    main()
