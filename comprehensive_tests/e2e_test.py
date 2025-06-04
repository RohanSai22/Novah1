"""
End-to-End Testing Suite for Nova AI
Tests complete user workflows from frontend to backend
"""

import time
import json
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
import sys

class NovaE2ETester:
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:5173"
        self.test_queries = [
            "What is machine learning?",
            "Create a simple Python function",
            "Research the latest trends in web development",
            "Plan a mobile app development project"
        ]
        self.results = []

    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

    def test_service_availability(self):
        """Test if both frontend and backend are available"""
        self.log("Testing service availability...")
        
        # Test backend
        try:
            req = urllib.request.Request(f"{self.backend_url}/health")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.getcode() == 200:
                    self.log("✅ Backend is available", "SUCCESS")
                    backend_ok = True
                else:
                    self.log(f"❌ Backend returned status {response.getcode()}", "ERROR")
                    backend_ok = False
        except Exception as e:
            self.log(f"❌ Backend not accessible: {e}", "ERROR")
            backend_ok = False

        # Test frontend
        try:
            req = urllib.request.Request(self.frontend_url)
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.getcode() == 200:
                    self.log("✅ Frontend is available", "SUCCESS")
                    frontend_ok = True
                else:
                    self.log(f"❌ Frontend returned status {response.getcode()}", "ERROR")
                    frontend_ok = False
        except Exception as e:
            self.log(f"❌ Frontend not accessible: {e}", "ERROR")
            frontend_ok = False

        return backend_ok and frontend_ok

    def test_api_endpoints(self):
        """Test key API endpoints"""
        self.log("Testing API endpoints...")
        
        endpoints = [
            ("/health", "GET"),
            ("/latest_answer", "GET"),
        ]
        
        success_count = 0
        
        for endpoint, method in endpoints:
            try:
                req = urllib.request.Request(f"{self.backend_url}{endpoint}", method=method)
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.getcode() == 200:
                        self.log(f"✅ {endpoint} ({method}) - OK", "SUCCESS")
                        success_count += 1
                    else:
                        self.log(f"❌ {endpoint} ({method}) - Status {response.getcode()}", "ERROR")
            except Exception as e:
                self.log(f"❌ {endpoint} ({method}) - {e}", "ERROR")

        return success_count == len(endpoints)

    def test_search_functionality(self, query):
        """Test the main search functionality"""
        self.log(f"Testing search with query: '{query[:50]}...'")
        
        try:
            # Prepare the request
            data = json.dumps({
                "query": query,
                "tts_enabled": False
            }).encode('utf-8')
            
            headers = {
                'Content-Type': 'application/json',
                'Origin': 'http://localhost:5173'
            }
            
            req = urllib.request.Request(
                f"{self.backend_url}/query",
                data=data,
                headers=headers,
                method='POST'
            )
            
            # Send the request
            start_time = time.time()
            with urllib.request.urlopen(req, timeout=15) as response:
                status_code = response.getcode()
                response_data = response.read().decode('utf-8')
                duration = time.time() - start_time
                
                if status_code == 200:
                    self.log(f"✅ Search query accepted (took {duration:.2f}s)", "SUCCESS")
                    
                    # Try to parse response
                    try:
                        json_response = json.loads(response_data)
                        self.log(f"   Response structure: {list(json_response.keys())}")
                    except json.JSONDecodeError:
                        self.log(f"   Response is not JSON (length: {len(response_data)})")
                    
                    return True
                else:
                    self.log(f"❌ Search failed with status {status_code}", "ERROR")
                    return False
                    
        except urllib.error.HTTPError as e:
            self.log(f"❌ HTTP Error: {e.code} - {e.reason}", "ERROR")
            return False
        except Exception as e:
            self.log(f"❌ Search error: {e}", "ERROR")
            return False

    def test_cors_functionality(self):
        """Test CORS configuration"""
        self.log("Testing CORS configuration...")
        
        try:
            headers = {
                'Origin': 'http://localhost:5173',
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'Content-Type'
            }
            
            req = urllib.request.Request(
                f"{self.backend_url}/query",
                headers=headers,
                method='OPTIONS'
            )
            
            with urllib.request.urlopen(req, timeout=5) as response:
                cors_headers = dict(response.headers)
                
                required_headers = [
                    'access-control-allow-origin',
                    'access-control-allow-methods',
                    'access-control-allow-headers'
                ]
                
                missing_headers = [h for h in required_headers if h not in cors_headers]
                
                if not missing_headers:
                    self.log("✅ CORS headers present", "SUCCESS")
                    return True
                else:
                    self.log(f"❌ Missing CORS headers: {missing_headers}", "ERROR")
                    return False
                    
        except Exception as e:
            self.log(f"❌ CORS test failed: {e}", "ERROR")
            return False

    def run_comprehensive_test_suite(self):
        """Run the complete test suite"""
        self.log("🚀 Starting Nova AI End-to-End Test Suite")
        self.log("=" * 60)
        
        start_time = time.time()
        tests_passed = 0
        total_tests = 0
        
        # Test 1: Service Availability
        total_tests += 1
        if self.test_service_availability():
            tests_passed += 1
        
        self.log("")
        
        # Test 2: API Endpoints
        total_tests += 1
        if self.test_api_endpoints():
            tests_passed += 1
        
        self.log("")
        
        # Test 3: CORS Functionality
        total_tests += 1
        if self.test_cors_functionality():
            tests_passed += 1
        
        self.log("")
        
        # Test 4: Search Functionality (multiple queries)
        for i, query in enumerate(self.test_queries[:2]):  # Test first 2 queries
            total_tests += 1
            self.log(f"Search Test {i+1}/{len(self.test_queries[:2])}")
            if self.test_search_functionality(query):
                tests_passed += 1
            self.log("")
            time.sleep(2)  # Brief pause between tests
        
        # Summary
        duration = time.time() - start_time
        self.log("=" * 60)
        self.log(f"TEST SUITE COMPLETED in {duration:.2f} seconds")
        self.log(f"Results: {tests_passed}/{total_tests} tests passed")
        
        if tests_passed == total_tests:
            self.log("🎉 ALL TESTS PASSED! Nova AI is fully operational!", "SUCCESS")
            return True
        elif tests_passed >= total_tests * 0.8:
            self.log("⚠️  Most tests passed. Minor issues detected.", "WARNING")
            return False
        else:
            self.log("❌ Multiple critical issues detected. System needs attention.", "ERROR")
            return False

def main():
    tester = NovaE2ETester()
    success = tester.run_comprehensive_test_suite()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
