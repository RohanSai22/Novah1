#!/usr/bin/env python3
"""
Comprehensive System Integration Test
Tests the complete AI agent workflow from query to report generation
"""
import asyncio
import aiohttp
import time
import json
from typing import Dict, Any

class SystemIntegrationTester:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_health_check(self) -> bool:
        """Test if the API is healthy"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                data = await response.json()
                print(f"✅ Health Check: {data['message']}")
                return response.status == 200
        except Exception as e:
            print(f"❌ Health Check Failed: {e}")
            return False

    async def submit_query(self, query: str, thread_id: str = "integration-test") -> bool:
        """Submit a query to the system"""
        try:
            payload = {"query": query, "thread_id": thread_id}
            async with self.session.post(
                f"{self.base_url}/query", 
                json=payload
            ) as response:
                data = await response.json()
                print(f"✅ Query Submitted: {data['status']}")
                return response.status == 200
        except Exception as e:
            print(f"❌ Query Submission Failed: {e}")
            return False

    async def monitor_progress(self, max_wait_time: int = 120) -> Dict[str, Any]:
        """Monitor the progress until completion or timeout"""
        start_time = time.time()
        final_state = None
        
        print("🔄 Monitoring Progress...")
        
        while time.time() - start_time < max_wait_time:
            try:
                # Check execution status
                async with self.session.get(f"{self.base_url}/execution_status") as response:
                    if response.status == 200:
                        data = await response.json()
                        is_active = data.get("is_active", False)
                        execution_state = data.get("execution_state", {})
                        
                        current_step = execution_state.get("current_step", 0)
                        total_steps = execution_state.get("total_steps", 0)
                        current_agent = execution_state.get("current_agent", "Unknown")
                        status = execution_state.get("status", "unknown")
                        
                        print(f"📊 Step {current_step}/{total_steps} | Agent: {current_agent} | Status: {status}")
                        
                        if not is_active and status in ["completed", "error"]:
                            final_state = execution_state
                            break
                
                # Check agent progress
                async with self.session.get(f"{self.base_url}/agent_progress") as response:
                    if response.status == 200:
                        agent_data = await response.json()
                        agent_progress = agent_data.get("agent_progress", {})
                        
                        for agent_name, progress in agent_progress.items():
                            agent_status = progress.get("status", "unknown")
                            current_task = progress.get("current_task", "N/A")
                            print(f"🤖 {agent_name}: {agent_status} - {current_task[:50]}...")
                
                await asyncio.sleep(2)  # Poll every 2 seconds
                
            except Exception as e:
                print(f"⚠️ Monitoring Error: {e}")
                await asyncio.sleep(2)
        
        return final_state or {}

    async def check_search_results(self) -> int:
        """Check search results collected"""
        try:
            async with self.session.get(f"{self.base_url}/search_results") as response:
                if response.status == 200:
                    data = await response.json()
                    results_count = data.get("total_results", 0)
                    print(f"🔍 Search Results: {results_count} found")
                    return results_count
        except Exception as e:
            print(f"❌ Search Results Check Failed: {e}")
        return 0

    async def check_report_generation(self) -> str:
        """Check if report was generated"""
        try:
            async with self.session.get(f"{self.base_url}/execution_status") as response:
                if response.status == 200:
                    data = await response.json()
                    execution_state = data.get("execution_state", {})
                    report_url = execution_state.get("final_report_url")
                    
                    if report_url:
                        print(f"📄 Report Generated: {report_url}")
                        return report_url
                    else:
                        print("⚠️ No report URL found")
        except Exception as e:
            print(f"❌ Report Check Failed: {e}")
        return ""

    async def test_api_endpoints(self) -> Dict[str, bool]:
        """Test all available API endpoints"""
        endpoints = [
            "/health",
            "/agent_progress", 
            "/search_results",
            "/links_processed",
            "/execution_summary",
            "/execution_status"
        ]
        
        results = {}
        
        for endpoint in endpoints:
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    results[endpoint] = response.status == 200
                    status_symbol = "✅" if results[endpoint] else "❌"
                    print(f"{status_symbol} {endpoint}: Status {response.status}")
            except Exception as e:
                results[endpoint] = False
                print(f"❌ {endpoint}: Failed - {e}")
        
        return results

    async def run_comprehensive_test(self):
        """Run the complete system test"""
        print("🚀 Starting Comprehensive System Integration Test")
        print("=" * 60)
        
        # Test 1: Health Check
        print("\n1. Testing API Health...")
        health_ok = await self.test_health_check()
        
        if not health_ok:
            print("❌ System not healthy, aborting tests")
            return False
        
        # Test 2: API Endpoints
        print("\n2. Testing API Endpoints...")
        endpoint_results = await self.test_api_endpoints()
        working_endpoints = sum(endpoint_results.values())
        total_endpoints = len(endpoint_results)
        print(f"📊 Endpoints Working: {working_endpoints}/{total_endpoints}")
        
        # Test 3: Query Processing
        test_query = "What are the latest trends in renewable energy technology for 2025?"
        print(f"\n3. Testing Query Processing...")
        print(f"📝 Test Query: {test_query}")
        
        query_ok = await self.submit_query(test_query)
        if not query_ok:
            print("❌ Query submission failed")
            return False
        
        # Test 4: Progress Monitoring
        print("\n4. Monitoring Execution Progress...")
        final_state = await self.monitor_progress()
        
        if not final_state:
            print("⚠️ Execution monitoring timed out")
            return False
        
        # Test 5: Results Validation
        print("\n5. Validating Results...")
        search_count = await self.check_search_results()
        report_url = await self.check_report_generation()
        
        # Final Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ API Health: {'PASS' if health_ok else 'FAIL'}")
        print(f"✅ Endpoints: {working_endpoints}/{total_endpoints} working")
        print(f"✅ Query Processing: {'PASS' if query_ok else 'FAIL'}")
        print(f"✅ Execution: {'PASS' if final_state.get('status') == 'completed' else 'FAIL'}")
        print(f"✅ Search Results: {search_count} found")
        print(f"✅ Report Generation: {'PASS' if report_url else 'FAIL'}")
        
        overall_success = (
            health_ok and 
            query_ok and 
            final_state.get('status') == 'completed' and
            search_count > 0
        )
        
        print(f"\n🎯 OVERALL RESULT: {'SUCCESS' if overall_success else 'PARTIAL SUCCESS'}")
        return overall_success

async def main():
    """Main test runner"""
    async with SystemIntegrationTester() as tester:
        success = await tester.run_comprehensive_test()
        exit_code = 0 if success else 1
        return exit_code

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
