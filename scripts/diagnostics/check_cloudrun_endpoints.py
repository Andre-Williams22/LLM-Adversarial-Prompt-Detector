#!/usr/bin/env python3
"""
Test script to verify the application works in Cloud Run environment
"""

import sys
import os
import traceback

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
    except Exception as e:
        print(f"PyTorch import failed: {e}")
        return False

    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except Exception as e:
        print(f"Transformers import failed: {e}")
        return False

    try:
        import fastapi
        print(f"FastAPI: {fastapi.__version__}")
    except Exception as e:
        print(f"FastAPI import failed: {e}")
        return False

    try:
        import gradio as gr
        print(f"Gradio: {gr.__version__}")
    except Exception as e:
        print(f"Gradio import failed: {e}")
        return False

    return True

def test_model_loading():
    """Test model loading"""
    print("\nTesting model loading...")

    try:
        from utils.fast_detection import fast_detector
        print("Fast detector imported")

        # Test detection
        result = fast_detector.detect_adversarial_sync("test prompt")
        print(f"Detection test passed: {result}")

        return True
    except Exception as e:
        print(f"Model loading failed: {e}")
        traceback.print_exc()
        return False

def test_app_creation():
    """Test FastAPI app creation"""
    print("\nTesting app creation...")

    try:
        from fastapi import FastAPI
        app = FastAPI()

        @app.get("/test")
        def test():
            return {"status": "ok"}

        print("FastAPI app creation successful")
        return True
    except Exception as e:
        print(f"App creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running Cloud Run compatibility tests...\n")

    tests = [
        ("Imports", test_imports),
        ("Model Loading", test_model_loading),
        ("App Creation", test_app_creation)
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {name} test...")
        print('='*50)

        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"{name} test crashed: {e}")
            traceback.print_exc()
            results.append((name, False))

    print(f"\n{'='*50}")
    print("TEST RESULTS")
    print('='*50)

    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{name}: {status}")

    all_passed = all(success for _, success in results)

    if all_passed:
        print("\nAll tests passed! App should work in Cloud Run.")
        sys.exit(0)
    else:
        print("\nSome tests failed. Check the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
