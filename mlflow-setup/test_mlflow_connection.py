#!/usr/bin/env python3
"""
Test MLflow connection and logging to production server
"""

import os
import mlflow
from datetime import datetime

def test_mlflow_connection():
    """Test connection to MLflow server and log a simple test run"""
    
    # Set the production MLflow URI
    mlflow_uri = "https://mlflow-ui.project.andrewilliams.ai"
    print(f"🔗 Testing MLflow connection to: {mlflow_uri}")
    
    try:
        # Set tracking URI
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Test connection by creating/getting experiment
        experiment_name = "adversarial_detection_system"
        mlflow.set_experiment(experiment_name)
        
        # Get experiment info
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            print(f"✅ Connected to experiment: {experiment_name}")
            print(f"   Experiment ID: {experiment.experiment_id}")
            print(f"   Lifecycle Stage: {experiment.lifecycle_stage}")
        else:
            print(f"⚠️  Experiment {experiment_name} not found, creating...")
        
        # Log a test run
        with mlflow.start_run(run_name=f"connection_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log some test metrics and parameters
            mlflow.log_param("test_parameter", "connection_test")
            mlflow.log_metric("test_metric", 1.0)
            mlflow.set_tag("test_type", "connection_verification")
            mlflow.set_tag("source", "production_chat_ui")
            
            print(f"✅ Successfully logged test run to {mlflow_uri}")
            
        return True
        
    except Exception as e:
        print(f"❌ MLflow connection failed: {e}")
        return False

def test_with_environment_variable():
    """Test using environment variable (simulates production deployment)"""
    print(f"\n🧪 Testing with MLFLOW_TRACKING_URI environment variable...")
    
    # Set environment variable
    os.environ["MLFLOW_TRACKING_URI"] = "https://mlflow-ui.project.andrewilliams.ai"
    
    # Import and test the same setup as your production app
    import sys
    sys.path.append('/Users/andrewilliams/Documents/development/projects/LLM-Adversarial-Prompt-Detector')
    
    # This should now use the environment variable
    from utils.fast_detection import AdversarialPromptDetector
    
    print("Creating detector (this will setup MLflow)...")
    detector = AdversarialPromptDetector()
    
    print("MLflow setup completed via environment variable!")
    return True

if __name__ == "__main__":
    print("🚀 MLflow Production Connection Test\n")
    
    # Test 1: Direct connection
    print("=" * 50)
    print("TEST 1: Direct MLflow Connection")
    print("=" * 50)
    success1 = test_mlflow_connection()
    
    # Test 2: Environment variable method (production simulation)
    print("\n" + "=" * 50)
    print("TEST 2: Environment Variable Method")
    print("=" * 50)
    try:
        success2 = test_with_environment_variable()
    except Exception as e:
        print(f"❌ Environment variable test failed: {e}")
        success2 = False
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Direct connection: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Environment variable: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! Your production app should now log to MLflow.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
