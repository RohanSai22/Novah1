#!/usr/bin/env python3
"""
Simple API Test - No External Dependencies
Tests the Nova AI API using only built-in Python modules
"""

import urllib.request
import urllib.parse
import urllib.error
import json
import socket
import time
from datetime import datetime

def test_endpoint(url, method='GET', data=None, headers=None):
    """Test an endpoint using urllib"""
    try:
        if headers is None:
            headers = {'Content-Type': 'application/json'}
        
        if data:
            data = json.dumps(data).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        
        with urllib.request.urlopen(req, timeout=10) as response:
            status_code = response.getcode()
            response_data = response.read().decode('utf-8')
            return True, status_code, response_data
            
    except urllib.error.HTTPError as e:
        return False, e.code, e.read().decode('utf-8')
    except Exception as e:
        return False, 0, str(e)

def check_port_open(host, port):
    """Check if a port is open"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            return result == 0
    except:
        return False

def run_comprehensive_tests():
    """Run comprehensive API tests"""
    print("=" * 60)
    print("NOVA AI COMPREHENSIVE API TESTS")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print()
    
    base_url = "http://localhost:8000"
    
    # Test 1: Check if backend port is open
    print("1. CHECKING BACKEND PORT ACCESSIBILITY")
    port_open = check_port_open("localhost", 8000)
    print(f"   Port 8000 status: {'OPEN' if port_open else 'CLOSED'}")
    
    if not port_open:
        print("   ❌ Backend is not running or not accessible on port 8000")
        return
    else:
        print("   ✅ Backend port is accessible")
    print()
    
    # Test 2: Health check
    print("2. HEALTH CHECK ENDPOINT")
    success, status, response = test_endpoint(f"{base_url}/health")
    if success and status == 200:
        print(f"   ✅ Health check passed (Status: {status})")
        print(f"   Response: {response}")
    else:
        print(f"   ❌ Health check failed (Status: {status})")
        print(f"   Error: {response}")
    print()
    
    # Test 3: Root endpoint
    print("3. ROOT ENDPOINT")
    success, status, response = test_endpoint(base_url)
    if success:
        print(f"   ✅ Root endpoint accessible (Status: {status})")
        print(f"   Response length: {len(response)} chars")
    else:
        print(f"   ❌ Root endpoint failed (Status: {status})")
        print(f"   Error: {response}")
    print()
    
    # Test 4: CORS preflight
    print("4. CORS PREFLIGHT CHECK")
    headers = {
        'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'Content-Type',
        'Origin': 'http://localhost:5173'
    }
    success, status, response = test_endpoint(f"{base_url}/api/search", method='OPTIONS', headers=headers)
    if success:
        print(f"   ✅ CORS preflight successful (Status: {status})")
    else:
        print(f"   ❌ CORS preflight failed (Status: {status})")
        print(f"   Error: {response}")
    print()
    
    # Test 5: Search endpoint
    print("5. SEARCH ENDPOINT TEST")
    search_data = {
        "query": "What is artificial intelligence?",
        "history": []
    }
    headers = {
        'Content-Type': 'application/json',
        'Origin': 'http://localhost:5173'
    }
    success, status, response = test_endpoint(f"{base_url}/api/search", method='POST', data=search_data, headers=headers)
    if success:
        print(f"   ✅ Search endpoint accessible (Status: {status})")
        print(f"   Response length: {len(response)} chars")
        # Try to parse response
        try:
            resp_json = json.loads(response)
            print(f"   Response structure: {list(resp_json.keys()) if isinstance(resp_json, dict) else 'Non-dict response'}")
        except:
            print(f"   Response preview: {response[:200]}...")
    else:
        print(f"   ❌ Search endpoint failed (Status: {status})")
        print(f"   Error: {response}")
    print()
    
    # Test 6: Frontend accessibility
    print("6. FRONTEND ACCESSIBILITY CHECK")
    frontend_open = check_port_open("localhost", 5173)
    print(f"   Frontend port 5173 status: {'OPEN' if frontend_open else 'CLOSED'}")
    
    if frontend_open:
        success, status, response = test_endpoint("http://localhost:5173")
        if success:
            print(f"   ✅ Frontend accessible (Status: {status})")
        else:
            print(f"   ❌ Frontend not responding (Status: {status})")
    print()
    
    print("=" * 60)
    print("COMPREHENSIVE TEST COMPLETED")
    print(f"Test completed at: {datetime.now()}")
    print("=" * 60)

if __name__ == "__main__":
    run_comprehensive_tests()
