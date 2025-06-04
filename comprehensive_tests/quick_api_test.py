#!/usr/bin/env python3
"""
Quick API Test Script
"""

import requests
import json
import time

def test_api():
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Nova AI API...")
    
    # Test 1: Health Check
    try:
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        print()
    except Exception as e:
        print(f"   Error: {e}")
        print()
    
    # Test 2: CORS Check
    try:
        print("2. Testing CORS preflight...")
        headers = {
            'Origin': 'http://localhost:5173',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'content-type'
        }
        response = requests.options(f"{base_url}/query", headers=headers, timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   CORS Headers: {dict(response.headers)}")
        print()
    except Exception as e:
        print(f"   Error: {e}")
        print()
    
    # Test 3: Query Endpoint
    try:
        print("3. Testing query endpoint...")
        payload = {"query": "Hello Nova AI", "tts_enabled": False}
        headers = {'Content-Type': 'application/json', 'Origin': 'http://localhost:5173'}
        response = requests.post(f"{base_url}/query", json=payload, headers=headers, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Agent: {data.get('agent_name', 'Unknown')}")
            print(f"   Answer: {data.get('answer', 'No answer')[:100]}...")
        else:
            print(f"   Response: {response.text[:200]}")
        print()
    except Exception as e:
        print(f"   Error: {e}")
        print()

if __name__ == "__main__":
    test_api()
