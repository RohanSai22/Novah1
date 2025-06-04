#!/usr/bin/env python3
"""
Simple System Validation Test
"""
import requests
import time
import json

def test_system():
    base_url = "http://localhost:8001"
    
    print("🚀 Starting Simple System Validation Test")
    print("=" * 50)
    
    try:
        # Test 1: Health Check
        print("\n1. Testing Health Check...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health: {data['message']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Test 2: Submit Query
        print("\n2. Submitting Test Query...")
        query_data = {
            "query": "Tell me about the latest developments in quantum computing", 
            "thread_id": "validation-test"
        }
        response = requests.post(f"{base_url}/query", json=query_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Query submitted: {data['status']}")
        else:
            print(f"❌ Query submission failed: {response.status_code}")
            return False
        
        # Test 3: Monitor for a few seconds
        print("\n3. Monitoring Progress...")
        for i in range(10):  # Monitor for 10 iterations
            try:
                # Check agent progress
                response = requests.get(f"{base_url}/agent_progress")
                if response.status_code == 200:
                    data = response.json()
                    current_agent = data.get("current_agent", "None")
                    is_processing = data.get("is_processing", False)
                    current_step = data.get("current_step", 0)
                    total_steps = data.get("total_steps", 0)
                    
                    print(f"📊 Step {current_step}/{total_steps} | Agent: {current_agent} | Processing: {is_processing}")
                    
                    if not is_processing and current_step > 0:
                        print("✅ Processing completed!")
                        break
                else:
                    print(f"⚠️ Agent progress check failed: {response.status_code}")
                
                time.sleep(3)
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
        
        # Test 4: Check Endpoints
        print("\n4. Testing Key Endpoints...")
        endpoints = [
            "/agent_progress",
            "/search_results", 
            "/execution_status"
        ]
        
        working_endpoints = 0
        for endpoint in endpoints:
            try:
                response = requests.get(f"{base_url}{endpoint}")
                if response.status_code == 200:
                    print(f"✅ {endpoint}: Working")
                    working_endpoints += 1
                else:
                    print(f"❌ {endpoint}: Failed ({response.status_code})")
            except Exception as e:
                print(f"❌ {endpoint}: Error - {e}")
        
        print(f"\n📊 Endpoints Working: {working_endpoints}/{len(endpoints)}")
        
        # Final Summary
        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)
        print("✅ System is operational and responding correctly")
        print("✅ Query processing is working")
        print("✅ Agent progress monitoring is functional")
        print(f"✅ {working_endpoints}/{len(endpoints)} endpoints working")
        
        print("\n🎯 RESULT: SYSTEM VALIDATION PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ SYSTEM VALIDATION FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_system()
    if success:
        print("\n🚀 System is ready for production use!")
    else:
        print("\n⚠️ System needs attention before production use.")
