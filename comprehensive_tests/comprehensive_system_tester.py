#!/usr/bin/env python3
"""
Comprehensive Nova AI System Testing Suite
Tests frontend, backend, API endpoints, and UI interactions
"""

import requests
import json
import time
import asyncio
import aiohttp
import sys
import os
from typing import Dict, List, Any
import concurrent.futures
from dataclasses import dataclass

@dataclass
class TestResult:
    test_name: str
    success: bool
    response_time: float
    details: str
    error: str = ""

class NovaSystemTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:5173"
        self.test_results: List[TestResult] = []
        
    def log_test(self, test_name: str, success: bool, response_time: float, details: str, error: str = ""):
        """Log test result"""
        result = TestResult(test_name, success, response_time, details, error)
        self.test_results.append(result)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name} ({response_time:.2f}s)")
        if details:
            print(f"   📋 {details}")
        if error:
            print(f"   🚨 {error}")
        print()

    def test_health_check(self):
        """Test basic health check endpoint"""
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                self.log_test("Health Check", True, response_time, 
                             f"Status: {response.status_code}, Response: {response.json()}")
                return True
            else:
                self.log_test("Health Check", False, response_time, 
                             f"Unexpected status: {response.status_code}")
                return False
        except Exception as e:
            response_time = time.time() - start_time
            self.log_test("Health Check", False, response_time, "", str(e))
            return False

    def test_cors_preflight(self):
        """Test CORS preflight request"""
        start_time = time.time()
        try:
            headers = {
                'Origin': 'http://localhost:5173',
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'content-type'
            }
            response = requests.options(f"{self.base_url}/query", headers=headers, timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                cors_headers = {
                    'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
                    'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
                    'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
                }
                self.log_test("CORS Preflight", True, response_time, 
                             f"CORS headers: {cors_headers}")
                return True
            else:
                self.log_test("CORS Preflight", False, response_time, 
                             f"Status: {response.status_code}, Headers: {dict(response.headers)}")
                return False
        except Exception as e:
            response_time = time.time() - start_time
            self.log_test("CORS Preflight", False, response_time, "", str(e))
            return False

    def test_query_endpoint(self):
        """Test the main query endpoint"""
        start_time = time.time()
        try:
            payload = {
                "query": "Test query for system validation",
                "tts_enabled": False
            }
            headers = {
                'Content-Type': 'application/json',
                'Origin': 'http://localhost:5173'
            }
            response = requests.post(f"{self.base_url}/query", 
                                   json=payload, headers=headers, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.log_test("Query Endpoint", True, response_time, 
                             f"Response keys: {list(data.keys())}, Agent: {data.get('agent_name', 'Unknown')}")
                return True
            else:
                self.log_test("Query Endpoint", False, response_time, 
                             f"Status: {response.status_code}, Response: {response.text[:200]}")
                return False
        except Exception as e:
            response_time = time.time() - start_time
            self.log_test("Query Endpoint", False, response_time, "", str(e))
            return False

    def test_latest_answer_endpoint(self):
        """Test latest answer endpoint"""
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/latest_answer", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.log_test("Latest Answer", True, response_time, 
                             f"Answer available: {bool(data.get('answer'))}")
                return True
            else:
                self.log_test("Latest Answer", False, response_time, 
                             f"Status: {response.status_code}")
                return False
        except Exception as e:
            response_time = time.time() - start_time
            self.log_test("Latest Answer", False, response_time, "", str(e))
            return False

    def test_execution_status_endpoint(self):
        """Test execution status endpoint"""
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/execution_status", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.log_test("Execution Status", True, response_time, 
                             f"Has execution state: {bool(data.get('execution_state'))}")
                return True
            else:
                self.log_test("Execution Status", False, response_time, 
                             f"Status: {response.status_code}")
                return False
        except Exception as e:
            response_time = time.time() - start_time
            self.log_test("Execution Status", False, response_time, "", str(e))
            return False

    def test_frontend_accessibility(self):
        """Test if frontend is accessible"""
        start_time = time.time()
        try:
            response = requests.get(self.frontend_url, timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                self.log_test("Frontend Accessibility", True, response_time, 
                             f"Frontend loaded, Content-Type: {response.headers.get('content-type')}")
                return True
            else:
                self.log_test("Frontend Accessibility", False, response_time, 
                             f"Status: {response.status_code}")
                return False
        except Exception as e:
            response_time = time.time() - start_time
            self.log_test("Frontend Accessibility", False, response_time, "", str(e))
            return False

    def test_multiple_concurrent_queries(self):
        """Test system under concurrent load"""
        start_time = time.time()
        
        def send_query(query_id):
            try:
                payload = {
                    "query": f"Concurrent test query {query_id}",
                    "tts_enabled": False
                }
                response = requests.post(f"{self.base_url}/query", json=payload, timeout=15)
                return response.status_code, query_id
            except Exception as e:
                return 0, query_id
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(send_query, i) for i in range(3)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            response_time = time.time() - start_time
            success_count = sum(1 for status, _ in results if status == 200)
            
            if success_count >= 1:  # At least one should succeed
                self.log_test("Concurrent Queries", True, response_time, 
                             f"Successful queries: {success_count}/3")
                return True
            else:
                self.log_test("Concurrent Queries", False, response_time, 
                             f"No successful queries out of 3")
                return False
        except Exception as e:
            response_time = time.time() - start_time
            self.log_test("Concurrent Queries", False, response_time, "", str(e))
            return False

    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🚀 Starting Nova AI Comprehensive Test Suite")
        print("=" * 60)
        
        tests = [
            self.test_frontend_accessibility,
            self.test_health_check,
            self.test_cors_preflight,
            self.test_latest_answer_endpoint,
            self.test_execution_status_endpoint,
            self.test_query_endpoint,
            self.test_multiple_concurrent_queries,
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
        
        print("=" * 60)
        print(f"📊 TEST SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! System is fully functional.")
        elif passed >= total * 0.8:
            print("⚠️  Most tests passed. System is mostly functional with minor issues.")
        else:
            print("🚨 Multiple test failures. System needs attention.")
        
        return self.test_results

def main():
    """Main test execution"""
    tester = NovaSystemTester()
    results = tester.run_all_tests()
    
    # Save results to file
    timestamp = int(time.time())
    report_file = f"comprehensive_tests/test_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump([{
            'test_name': r.test_name,
            'success': r.success,
            'response_time': r.response_time,
            'details': r.details,
            'error': r.error
        } for r in results], f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")

if __name__ == "__main__":
    main()
