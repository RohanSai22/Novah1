"""
Comprehensive Frontend-Backend Integration Test
Tests the complete flow from frontend UI to backend API
"""

import json
import time
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime

class NovaIntegrationTester:
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:5173"
        self.test_results = []
    
    def log_test(self, test_name, success, details, duration=None):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        duration_str = f" ({duration:.2f}s)" if duration else ""
        print(f"{status} {test_name}{duration_str}")
        if not success or details:
            print(f"    {details}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })
    
    def test_backend_health(self):
        """Test backend health endpoint"""
        start_time = time.time()
        try:
            req = urllib.request.Request(f"{self.backend_url}/health")
            with urllib.request.urlopen(req, timeout=5) as response:
                status_code = response.getcode()
                response_data = response.read().decode('utf-8')
                duration = time.time() - start_time
                
                if status_code == 200:
                    self.log_test("Backend Health Check", True, f"Response: {response_data}", duration)
                    return True
                else:
                    self.log_test("Backend Health Check", False, f"Status: {status_code}", duration)
                    return False
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Backend Health Check", False, str(e), duration)
            return False
    
    def test_cors_headers(self):
        """Test CORS configuration"""
        start_time = time.time()
        try:
            headers = {
                'Origin': 'http://localhost:5173',
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'Content-Type'
            }
            req = urllib.request.Request(f"{self.backend_url}/api/search", headers=headers, method='OPTIONS')
            
            with urllib.request.urlopen(req, timeout=5) as response:
                status_code = response.getcode()
                response_headers = dict(response.headers)
                duration = time.time() - start_time
                
                cors_headers = {
                    'access-control-allow-origin': response_headers.get('access-control-allow-origin'),
                    'access-control-allow-methods': response_headers.get('access-control-allow-methods'),
                    'access-control-allow-headers': response_headers.get('access-control-allow-headers')
                }
                
                if status_code == 200 and cors_headers['access-control-allow-origin']:
                    self.log_test("CORS Configuration", True, f"Headers: {cors_headers}", duration)
                    return True
                else:
                    self.log_test("CORS Configuration", False, f"Status: {status_code}, Headers: {cors_headers}", duration)
                    return False
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("CORS Configuration", False, str(e), duration)
            return False
    
    def test_search_endpoint(self):
        """Test the main search endpoint with a simple query"""
        start_time = time.time()
        try:
            query_data = {
                "query": "What is machine learning?",
                "tts_enabled": False
            }
            
            headers = {
                'Content-Type': 'application/json',
                'Origin': 'http://localhost:5173'
            }
            
            data = json.dumps(query_data).encode('utf-8')
            req = urllib.request.Request(
                f"{self.backend_url}/query", 
                data=data, 
                headers=headers, 
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                status_code = response.getcode()
                response_data = response.read().decode('utf-8')
                duration = time.time() - start_time
                
                if status_code == 200:
                    try:
                        json_response = json.loads(response_data)
                        self.log_test("Search Endpoint", True, f"Response keys: {list(json_response.keys())}", duration)
                        return True
                    except json.JSONDecodeError:
                        self.log_test("Search Endpoint", True, "Valid response (non-JSON)", duration)
                        return True
                else:
                    self.log_test("Search Endpoint", False, f"Status: {status_code}", duration)
                    return False
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Search Endpoint", False, str(e), duration)
            return False
    
    def test_latest_answer_endpoint(self):
        """Test the latest answer polling endpoint"""
        start_time = time.time()
        try:
            req = urllib.request.Request(f"{self.backend_url}/latest_answer")
            with urllib.request.urlopen(req, timeout=5) as response:
                status_code = response.getcode()
                response_data = response.read().decode('utf-8')
                duration = time.time() - start_time
                
                if status_code == 200:
                    try:
                        json_response = json.loads(response_data)
                        self.log_test("Latest Answer Endpoint", True, f"Response structure valid", duration)
                        return True
                    except json.JSONDecodeError:
                        self.log_test("Latest Answer Endpoint", False, "Invalid JSON response", duration)
                        return False
                else:
                    self.log_test("Latest Answer Endpoint", False, f"Status: {status_code}", duration)
                    return False
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Latest Answer Endpoint", False, str(e), duration)
            return False
    
    def test_frontend_accessibility(self):
        """Test if frontend is accessible"""
        start_time = time.time()
        try:
            req = urllib.request.Request(self.frontend_url)
            with urllib.request.urlopen(req, timeout=5) as response:
                status_code = response.getcode()
                response_data = response.read().decode('utf-8')
                duration = time.time() - start_time
                
                if status_code == 200 and len(response_data) > 100:
                    self.log_test("Frontend Accessibility", True, f"HTML size: {len(response_data)} chars", duration)
                    return True
                else:
                    self.log_test("Frontend Accessibility", False, f"Status: {status_code}, Size: {len(response_data)}", duration)
                    return False
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Frontend Accessibility", False, str(e), duration)
            return False
    
    def run_full_integration_test(self):
        """Run complete integration test suite"""
        print("🚀 NOVA AI FRONTEND-BACKEND INTEGRATION TEST")
        print("=" * 60)
        print(f"Started at: {datetime.now()}")
        print()
        
        # Run all tests
        tests = [
            self.test_backend_health,
            self.test_cors_headers,
            self.test_frontend_accessibility,
            self.test_latest_answer_endpoint,
            self.test_search_endpoint
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
            print()
        
        # Summary
        print("=" * 60)
        print(f"INTEGRATION TEST SUMMARY: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Frontend-Backend integration is working!")
        elif passed >= total * 0.8:
            print("⚠️  Most tests passed. Minor issues detected.")
        else:
            print("❌ Multiple integration issues detected. Review required.")
        
        print(f"Completed at: {datetime.now()}")
        print("=" * 60)
        
        return passed == total

def main():
    tester = NovaIntegrationTester()
    tester.run_full_integration_test()

if __name__ == "__main__":
    main()
