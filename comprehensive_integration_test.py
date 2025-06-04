#!/usr/bin/env python3
"""
Comprehensive End-to-End Test for Novah AI Enhanced Frontend-Backend Integration
Tests all enhanced features including deep search, agent views, timeline, and reporting
"""

import requests
import json
import time
import sys
import os
from typing import Dict, Any, List

class NovahIntegrationTester:
    def __init__(self):
        self.backend_url = "http://localhost:8002"
        self.frontend_url = "http://localhost:5174"
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "tests": []
        }
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if details:
            print(f"    {details}")
            
        self.test_results["tests"].append({
            "name": test_name,
            "passed": passed,
            "details": details
        })
        
        if passed:
            self.test_results["passed"] += 1
        else:
            self.test_results["failed"] += 1
    
    def test_backend_health(self) -> bool:
        """Test backend health endpoint"""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log_test("Backend Health", True, f"Version: {data.get('version')}")
                    return True
            self.log_test("Backend Health", False, f"Status: {response.status_code}")
            return False
        except Exception as e:
            self.log_test("Backend Health", False, f"Error: {str(e)}")
            return False
    
    def test_frontend_accessibility(self) -> bool:
        """Test frontend accessibility"""
        try:
            response = requests.get(self.frontend_url, timeout=5)
            if response.status_code == 200 and "Novah UI" in response.text:
                self.log_test("Frontend Accessibility", True, "Frontend responding correctly")
                return True
            self.log_test("Frontend Accessibility", False, f"Status: {response.status_code}")
            return False
        except Exception as e:
            self.log_test("Frontend Accessibility", False, f"Error: {str(e)}")
            return False
    
    def test_agent_capabilities(self) -> bool:
        """Test enhanced agent capabilities endpoint"""
        try:
            response = requests.get(f"{self.backend_url}/agent_capabilities", timeout=5)
            if response.status_code == 200:
                data = response.json()
                required_agents = ["enhanced_search_agent", "enhanced_web_agent", "enhanced_coding_agent"]
                available_agents = [agent for agent in required_agents if data.get(agent, {}).get("available")]
                
                if len(available_agents) >= 3:
                    self.log_test("Agent Capabilities", True, f"All enhanced agents available: {len(available_agents)}")
                    return True
                else:
                    self.log_test("Agent Capabilities", False, f"Missing agents: {set(required_agents) - set(available_agents)}")
                    return False
            self.log_test("Agent Capabilities", False, f"Status: {response.status_code}")
            return False
        except Exception as e:
            self.log_test("Agent Capabilities", False, f"Error: {str(e)}")
            return False
    
    def test_deep_search_query(self) -> str:
        """Test deep search query submission"""
        try:
            query_data = {
                "query": "Test comprehensive deep search functionality for enhanced UI integration",
                "deep_search": True
            }
            response = requests.post(f"{self.backend_url}/query", json=query_data, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "accepted":
                    self.log_test("Deep Search Query", True, f"Query accepted: {data.get('query')[:50]}...")
                    return data.get("query", "")
                else:
                    self.log_test("Deep Search Query", False, f"Query rejected: {data}")
                    return ""
            self.log_test("Deep Search Query", False, f"Status: {response.status_code}")
            return ""
        except Exception as e:
            self.log_test("Deep Search Query", False, f"Error: {str(e)}")
            return ""
    
    def test_execution_monitoring(self) -> bool:
        """Test real-time execution monitoring"""
        try:
            response = requests.get(f"{self.backend_url}/execution_status", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("is_active"):
                    execution_state = data.get("execution_state", {})
                    intent = execution_state.get("intent", "")
                    if intent:
                        self.log_test("Execution Monitoring", True, f"Active execution: {intent[:50]}...")
                        return True
                
                self.log_test("Execution Monitoring", True, "No active execution (expected)")
                return True
            
            self.log_test("Execution Monitoring", False, f"Status: {response.status_code}")
            return False
        except Exception as e:
            self.log_test("Execution Monitoring", False, f"Error: {str(e)}")
            return False
    
    def test_agent_view_data(self) -> bool:
        """Test comprehensive agent view data"""
        try:
            response = requests.get(f"{self.backend_url}/agent_view_data", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                required_sections = ["plan", "browser", "search", "coding", "report", "execution"]
                available_sections = [section for section in required_sections if section in data]
                
                if len(available_sections) == len(required_sections):
                    # Check if execution data is meaningful
                    execution_data = data.get("execution", {})
                    current_agent = execution_data.get("current_agent")
                    status = execution_data.get("status")
                    
                    self.log_test("Agent View Data", True, 
                                f"All sections available. Current: {current_agent} ({status})")
                    return True
                else:
                    missing = set(required_sections) - set(available_sections)
                    self.log_test("Agent View Data", False, f"Missing sections: {missing}")
                    return False
            
            self.log_test("Agent View Data", False, f"Status: {response.status_code}")
            return False
        except Exception as e:
            self.log_test("Agent View Data", False, f"Error: {str(e)}")
            return False
    
    def test_quality_metrics(self) -> bool:
        """Test quality metrics endpoint"""
        try:
            response = requests.get(f"{self.backend_url}/quality_metrics", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and len(data) > 0:
                    metrics_count = len(data)
                    self.log_test("Quality Metrics", True, f"Quality metrics available: {metrics_count} metrics")
                    return True
                else:
                    self.log_test("Quality Metrics", True, "Quality metrics endpoint responding (no active metrics)")
                    return True
            
            self.log_test("Quality Metrics", False, f"Status: {response.status_code}")
            return False
        except Exception as e:
            self.log_test("Quality Metrics", False, f"Error: {str(e)}")
            return False
    
    def test_enhanced_features_integration(self) -> bool:
        """Test integration of all enhanced features"""
        print("\n🔄 Testing Enhanced Features Integration...")
        
        # Submit a query and monitor the enhanced agent view
        query = self.test_deep_search_query()
        if not query:
            return False
        
        # Wait for processing to start
        time.sleep(2)
        
        # Check agent view data multiple times to see real-time updates
        updates_detected = 0
        previous_data = None
        
        for i in range(3):
            try:
                response = requests.get(f"{self.backend_url}/agent_view_data", timeout=5)
                if response.status_code == 200:
                    current_data = response.json()
                    
                    if previous_data and current_data != previous_data:
                        updates_detected += 1
                    
                    previous_data = current_data
                    
                    # Check if we have meaningful data
                    plan_steps = len(current_data.get("plan", {}).get("steps", []))
                    current_agent = current_data.get("execution", {}).get("current_agent")
                    
                    if plan_steps > 0 and current_agent:
                        self.log_test("Enhanced Integration", True, 
                                    f"Real-time updates: {updates_detected}, Plan steps: {plan_steps}, Agent: {current_agent}")
                        return True
                
                time.sleep(3)  # Wait between checks
            except Exception as e:
                continue
        
        self.log_test("Enhanced Integration", False, "No real-time updates detected")
        return False
    
    def run_comprehensive_test(self):
        """Run all comprehensive tests"""
        print("🚀 Starting Comprehensive Novah AI Integration Test\n")
        
        # Core infrastructure tests
        print("📋 Testing Core Infrastructure...")
        self.test_backend_health()
        self.test_frontend_accessibility()
        
        # Enhanced API tests
        print("\n📋 Testing Enhanced API Endpoints...")
        self.test_agent_capabilities()
        self.test_execution_monitoring()
        self.test_agent_view_data()
        self.test_quality_metrics()
        
        # Integration tests
        print("\n📋 Testing Deep Search & Real-time Integration...")
        self.test_enhanced_features_integration()
        
        # Print final results
        print(f"\n📊 Test Results Summary:")
        print(f"✅ Passed: {self.test_results['passed']}")
        print(f"❌ Failed: {self.test_results['failed']}")
        print(f"📈 Success Rate: {(self.test_results['passed'] / (self.test_results['passed'] + self.test_results['failed']) * 100):.1f}%")
        
        if self.test_results['failed'] == 0:
            print("\n🎉 ALL TESTS PASSED! Novah AI Enhanced Frontend-Backend Integration is fully operational!")
        else:
            print(f"\n⚠️  {self.test_results['failed']} tests failed. Check the details above.")
        
        return self.test_results['failed'] == 0

if __name__ == "__main__":
    tester = NovahIntegrationTester()
    success = tester.run_comprehensive_test()
    sys.exit(0 if success else 1)
