#!/usr/bin/env python3
"""
Post-deployment warm-up script
This script can be run after deployment to pre-load models
"""

import requests
import time
import sys

def warm_up_deployment(base_url="http://localhost"):
    """Warm up the deployment by triggering model loading"""
    
    print("Starting deployment warm-up...")
    
    # 1. Check if the app is running
    try:
        health_response = requests.get(f"{base_url}/health", timeout=10)
        if health_response.status_code != 200:
            print(f"Health check failed: {health_response.status_code}")
            return False
        
        health_data = health_response.json()
        print(f"App is running - Status: {health_data.get('status', 'unknown')}")
        
    except Exception as e:
        print(f"Failed to connect to app: {e}")
        return False
    
    # 2. Check loading status
    try:
        status_response = requests.get(f"{base_url}/loading-status", timeout=10)
        status_data = status_response.json()
        print(f"Loading status: {status_data.get('message', 'unknown')}")
        
        if status_data.get('models_loaded', False):
            print("Models are already loaded!")
            return True
            
    except Exception as e:
        print(f"Could not check loading status: {e}")
    
    # 3. Trigger warm-up
    try:
        print("Triggering model warm-up...")
        warmup_response = requests.post(f"{base_url}/warm-up", timeout=10)
        warmup_data = warmup_response.json()
        print(f"Warm-up response: {warmup_data.get('message', 'unknown')}")
        
    except Exception as e:
        print(f"Could not trigger warm-up: {e}")
    
    # 4. Wait for models to load
    print("Waiting for models to load (up to 2 minutes)...")
    
    for i in range(24):  # Check every 5 seconds for 2 minutes
        try:
            status_response = requests.get(f"{base_url}/loading-status", timeout=10)
            status_data = status_response.json()
            
            if status_data.get('models_loaded', False):
                print("Models loaded successfully!")
                return True
            elif status_data.get('models_loading', False):
                print(f"Still loading... ({i*5}s elapsed)")
            else:
                print(f"Status: {status_data.get('message', 'unknown')}")
                
        except Exception as e:
            print(f"Status check failed: {e}")
        
        time.sleep(5)
    
    print("Timeout reached - models may still be loading")
    return False

def main():
    """Main warm-up function"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost"
    
    print(f"Warming up deployment at: {base_url}")
    
    success = warm_up_deployment(base_url)
    
    if success:
        print("Warm-up completed successfully!")
        sys.exit(0)
    else:
        print("Warm-up completed with issues")
        sys.exit(1)

if __name__ == "__main__":
    main()
