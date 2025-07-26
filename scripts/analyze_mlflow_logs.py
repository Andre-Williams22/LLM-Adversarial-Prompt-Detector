#!/usr/bin/env python3
"""
MLflow Analysis Script for Adversarial Detection System
Demonstrates how to query and analyze the industry-standard logging
"""

import mlflow
import pandas as pd
from datetime import datetime, timedelta

def analyze_model_performance():
    """Analyze individual model performance from MLflow logs"""
    print("Individual Model Performance Analysis")
    print("=" * 50)
    
    # Set experiment for all runs (single experiment structure)
    mlflow.set_experiment("adversarial_detection_system")
    
    # Get all individual model runs (excluding ensemble decisions)
    experiment = mlflow.get_experiment_by_name("adversarial_detection_system")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.model_name != 'ensemble_voting_system' AND metrics.inference_time_ms > 0",
        order_by=["start_time DESC"],
        max_results=100
    )
    
    if runs.empty:
        print("No model runs found. Run some detections first.")
        return
    
    print(f"Found {len(runs)} individual model runs")
    
    # Analyze by model type
    model_stats = runs.groupby('params.model_name').agg({
        'metrics.score': ['mean', 'std', 'count'],
        'metrics.inference_time_ms': ['mean', 'max', 'min'],
        'params.adversarial': lambda x: (x == 'True').sum()
    }).round(3)
    
    print("\nModel Performance Summary:")
    print(model_stats)
    
    # Detection rate by model
    print("\nAdversarial Detection Rate by Model:")
    for model_name in runs['params.model_name'].unique():
        model_runs = runs[runs['params.model_name'] == model_name]
        total = len(model_runs)
        adversarial = len(model_runs[model_runs['params.adversarial'] == 'True'])
        rate = (adversarial / total * 100) if total > 0 else 0
        print(f"  {model_name}: {adversarial}/{total} ({rate:.1f}%)")

def analyze_ensemble_decisions():
    """Analyze ensemble decision patterns from MLflow logs"""
    print("\nEnsemble Decision Analysis")
    print("=" * 50)
    
    # Set experiment (same single experiment)
    mlflow.set_experiment("adversarial_detection_system")
    
    # Get ensemble runs only
    experiment = mlflow.get_experiment_by_name("adversarial_detection_system")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.model_name = 'ensemble_voting_system' AND metrics.total_inference_time_ms > 0",
        order_by=["start_time DESC"],
        max_results=100
    )
    
    if runs.empty:
        print("No ensemble runs found. Run some detections first.")
        return
    
    print(f"Found {len(runs)} ensemble decision runs")
    
    # Analyze decision patterns
    print("\n🗳️ Voting Mechanism Effectiveness:")
    voting_mechanisms = [
        'voting_high_confidence_trigger',
        'voting_weak_signals_trigger', 
        'voting_majority_consensus',
        'voting_weighted_ensemble'
    ]
    
    for mechanism in voting_mechanisms:
        if f'metrics.{mechanism}' in runs.columns:
            triggered = runs[runs[f'metrics.{mechanism}'] == 1.0]
            total_triggered = len(triggered)
            adversarial_correct = len(triggered[triggered['params.adversarial'] == 'True'])
            accuracy = (adversarial_correct / total_triggered * 100) if total_triggered > 0 else 0
            print(f"  {mechanism.replace('voting_', '').replace('_', ' ').title()}: {total_triggered} triggers, {accuracy:.1f}% accuracy")
    
    # Early exit analysis
    early_exits = len(runs[runs['metrics.early_exit'] == 1.0])
    total_runs = len(runs)
    early_exit_rate = (early_exits / total_runs * 100) if total_runs > 0 else 0
    
    print(f"\n⚡ Early Exit Performance:")
    print(f"  Early exits: {early_exits}/{total_runs} ({early_exit_rate:.1f}%)")
    
    if early_exits > 0:
        early_exit_runs = runs[runs['metrics.early_exit'] == 1.0]
        avg_early_time = early_exit_runs['metrics.total_inference_time_ms'].mean()
        print(f"  Average early exit time: {avg_early_time:.1f}ms")
    
    # Performance by sensitivity mode
    print(f"\n⚙️ Performance by Sensitivity Mode:")
    for mode in runs['params.sensitivity_mode'].unique():
        mode_runs = runs[runs['params.sensitivity_mode'] == mode]
        avg_time = mode_runs['metrics.total_inference_time_ms'].mean()
        adversarial_rate = len(mode_runs[mode_runs['params.adversarial'] == 'True']) / len(mode_runs) * 100
        print(f"  {mode}: {len(mode_runs)} runs, {avg_time:.1f}ms avg, {adversarial_rate:.1f}% flagged")

def recent_detections_summary():
    """Show recent detection summary with prompts"""
    print("\n📈 Recent Detection Summary (Last 10 Ensemble Decisions)")
    print("=" * 80)
    
    # Get recent ensemble decisions from single experiment
    mlflow.set_experiment("adversarial_detection_system")
    experiment = mlflow.get_experiment_by_name("adversarial_detection_system")
    recent_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.model_name = 'ensemble_voting_system'",
        order_by=["start_time DESC"],
        max_results=10
    )
    
    if recent_runs.empty:
        print("No recent detections found.")
        return
    
    print(f"{'Timestamp':<20} {'Result':<12} {'Reason':<25} {'Time(ms)':<10} {'Prompt Preview':<30}")
    print("-" * 100)
    
    for _, run in recent_runs.iterrows():
        timestamp = run.get('params.timestamp', 'Unknown')
        result = "ADVERSARIAL" if run.get('params.adversarial') == 'True' else "SAFE"
        reason = run.get('params.decision_reason', 'Unknown')[:23]
        time_ms = run.get('metrics.total_inference_time_ms', 0)
        prompt_preview = run.get('params.prompt', 'No prompt logged')[:28] + "..."
        
        print(f"{timestamp:<20} {result:<12} {reason:<25} {time_ms:<10.1f} {prompt_preview:<30}")

def analyze_prompt_patterns():
    """Analyze patterns in adversarial vs safe prompts"""
    print("\n📝 Prompt Pattern Analysis")
    print("=" * 50)
    
    # Get all ensemble decisions with prompts
    mlflow.set_experiment("adversarial_detection_system")
    experiment = mlflow.get_experiment_by_name("adversarial_detection_system")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.model_name = 'ensemble_voting_system'",
        order_by=["start_time DESC"],
        max_results=50
    )
    
    if runs.empty:
        print("No prompt data found.")
        return
    
    # Analyze adversarial vs safe prompts
    adversarial_runs = runs[runs['params.adversarial'] == 'True']
    safe_runs = runs[runs['params.adversarial'] == 'False']
    
    print(f"📊 Dataset Overview:")
    print(f"  Total analyzed: {len(runs)}")
    print(f"  Adversarial: {len(adversarial_runs)} ({len(adversarial_runs)/len(runs)*100:.1f}%)")
    print(f"  Safe: {len(safe_runs)} ({len(safe_runs)/len(runs)*100:.1f}%)")
    
    if len(adversarial_runs) > 0:
        print(f"\n🚨 Adversarial Prompt Characteristics:")
        avg_length_adv = adversarial_runs['params.input_length'].astype(float).mean()
        print(f"  Average length: {avg_length_adv:.0f} characters")
        
        # Show sample adversarial prompts (first 100 chars)
        print(f"  Sample prompts:")
        for i, (_, run) in enumerate(adversarial_runs.head(3).iterrows()):
            prompt = run.get('params.prompt', 'No prompt')[:80]
            print(f"    {i+1}. {prompt}...")
    
    if len(safe_runs) > 0:
        print(f"\n✅ Safe Prompt Characteristics:")
        avg_length_safe = safe_runs['params.input_length'].astype(float).mean()
        print(f"  Average length: {avg_length_safe:.0f} characters")
        
        # Show sample safe prompts (first 100 chars)
        print(f"  Sample prompts:")
        for i, (_, run) in enumerate(safe_runs.head(3).iterrows()):
            prompt = run.get('params.prompt', 'No prompt')[:80]
            print(f"    {i+1}. {prompt}...")
    
    # Analyze prompt length correlation with detection
    if len(runs) > 5:
        print(f"\n📏 Length Analysis:")
        runs['input_length_num'] = runs['params.input_length'].astype(float)
        short_prompts = runs[runs['input_length_num'] < 50]
        medium_prompts = runs[(runs['input_length_num'] >= 50) & (runs['input_length_num'] < 200)]
        long_prompts = runs[runs['input_length_num'] >= 200]
        
        for category, subset in [("Short (<50 chars)", short_prompts), 
                                ("Medium (50-200 chars)", medium_prompts), 
                                ("Long (>200 chars)", long_prompts)]:
            if len(subset) > 0:
                adv_rate = len(subset[subset['params.adversarial'] == 'True']) / len(subset) * 100
                print(f"  {category}: {len(subset)} prompts, {adv_rate:.1f}% flagged as adversarial")

def main():
    """Main analysis function"""
    try:
        print("🔬 MLflow Adversarial Detection Analysis")
        print("=" * 60)
        
        analyze_model_performance()
        analyze_ensemble_decisions()
        recent_detections_summary()
        analyze_prompt_patterns()
        
        print("\n✅ Analysis complete!")
        print("\n💡 Tips:")
        print("  • Check MLflow UI at http://localhost:5000 for detailed views")
        print("  • Filter by tags: model_type='individual' or 'ensemble'")
        print("  • Compare performance across sensitivity modes")
        print("  • Monitor early_exit rates for optimization opportunities")
        print("  • Analyze prompt patterns to improve detection accuracy")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("Make sure MLflow is running and you have logged some detections.")

if __name__ == "__main__":
    main()
