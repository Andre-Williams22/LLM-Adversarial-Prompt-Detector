#!/usr/bin/env python3
"""
Quick startup diagnostic for the adversarial prompt detector
Tests key components to identify potential deployment issues
"""

import sys
import time
import traceback

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing imports...")
    try:
        import os
        import time
        import mlflow
        import gradio as gr
        from openai import OpenAI
        from dotenv import load_dotenv
        from fastapi import FastAPI
        from fastapi.responses import Response
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram, Gauge
        print("✅ Core dependencies imported successfully")
        
        # Test custom modules
        from utils.model_processing import load_models, detect_adversarial_prompt
        from utils.fast_detection import detect_adversarial_prompt_fast
        from utils.mongodb_manager import mongodb_manager
        print("✅ Custom modules imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_environment():
    """Test environment variables"""
    print("\n🌍 Testing environment...")
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            print(f"✅ OpenAI API key found (length: {len(openai_key)})")
        else:
            print("⚠️ OpenAI API key not found")
            
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
        print(f"✅ MLflow URI: {mlflow_uri}")
        
        return True
    except Exception as e:
        print(f"❌ Environment error: {e}")
        return False

def test_fast_detection():
    """Test fast detection system"""
    print("\n⚡ Testing fast detection...")
    try:
        import asyncio
        from utils.fast_detection import detect_adversarial_prompt_fast
        
        # Simple test message
        test_message = "Hello, how are you today?"
        
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        start_time = time.time()
        result = loop.run_until_complete(detect_adversarial_prompt_fast(test_message))
        duration = time.time() - start_time
        
        loop.close()
        
        is_adv, reasoning = result
        print(f"✅ Fast detection working - Duration: {duration:.3f}s")
        print(f"   Result: {'Adversarial' if is_adv else 'Safe'}")
        print(f"   Reason: {reasoning.get('reason', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Fast detection error: {e}")
        traceback.print_exc()
        return False

def test_app_creation():
    """Test FastAPI app creation"""
    print("\n🚀 Testing FastAPI app creation...")
    try:
        from main import app
        print("✅ FastAPI app created successfully")
        
        # Test if app has the expected routes
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/health", "/metrics", "/chat"]
        
        for route in expected_routes:
            if any(route in r for r in routes):
                print(f"✅ Route {route} found")
            else:
                print(f"⚠️ Route {route} not found")
        
        return True
    except Exception as e:
        print(f"❌ App creation error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    print("🔍 Adversarial Prompt Detector - Startup Diagnostics")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Environment Test", test_environment),
        ("Fast Detection Test", test_fast_detection),
        ("App Creation Test", test_app_creation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The application should start successfully.")
    else:
        print("⚠️ Some tests failed. Check the errors above for deployment issues.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
