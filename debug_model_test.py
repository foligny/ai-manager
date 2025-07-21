#!/usr/bin/env python3
"""
Debug script for testing the model testing endpoint with real-time logging.
"""

import requests
import json
import time
import subprocess
import sys
import os

def test_model_endpoint():
    """Test the model testing endpoint with detailed logging."""
    
    # Base URL
    base_url = "http://localhost:8000"
    
    # Login to get token
    login_data = {
        "username": "admin",
        "password": "admin123"
    }
    
    try:
        print("🔐 Logging in...")
        response = requests.post(f"{base_url}/auth/login", data=login_data)
        response.raise_for_status()
        token_data = response.json()
        token = token_data["access_token"]
        print("✅ Login successful")
        
        # Test files
        test_files = [
            ("test_data/test_data.csv", "CSV file"),
            ("test_data/test_data.json", "JSON file")
        ]
        
        headers = {
            "Authorization": f"Bearer {token}"
        }
        
        for file_path, description in test_files:
            print(f"\n🧪 Testing with {description}: {file_path}")
            
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue
            
            try:
                with open(file_path, 'rb') as f:
                    files = {'test_data': f}
                    
                    print(f"📤 Sending request to {base_url}/models/test")
                    response = requests.post(
                        f"{base_url}/models/test",
                        headers=headers,
                        files=files
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Success! Status: {response.status_code}")
                    print(f"📊 Results:")
                    print(f"   Data shape: {result.get('data_shape')}")
                    print(f"   Predictions: {result.get('predictions')}")
                    print(f"   Accuracy: {result.get('accuracy')}")
                    print(f"   Loss: {result.get('loss')}")
                    print(f"   Test samples: {result.get('test_samples')}")
                    print(f"   Performance: {result.get('model_performance')}")
                else:
                    print(f"❌ Failed! Status: {response.status_code}")
                    print(f"   Error: {response.text}")
                    
            except Exception as e:
                print(f"❌ Error testing {file_path}: {e}")
        
        print("\n📋 To view real-time logs, run:")
        print("   tail -f ai_manager.log")
        print("\n📋 To view server logs, check the terminal where uvicorn is running")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

def monitor_logs():
    """Monitor the log file in real-time."""
    print("📊 Monitoring logs in real-time...")
    print("Press Ctrl+C to stop monitoring")
    print("-" * 50)
    
    try:
        # Use tail -f to monitor the log file
        process = subprocess.Popen(
            ["tail", "-f", "ai_manager.log"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        for line in process.stdout:
            print(line.rstrip())
            
    except KeyboardInterrupt:
        print("\n🛑 Log monitoring stopped")
        process.terminate()
    except Exception as e:
        print(f"❌ Error monitoring logs: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        monitor_logs()
    else:
        test_model_endpoint() 