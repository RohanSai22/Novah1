#!/usr/bin/env python3
"""
Quick API Test - Basic connectivity check
"""

import socket
import time

def check_service(host, port, name):
    """Check if a service is running"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            status = "✅ RUNNING" if result == 0 else "❌ NOT ACCESSIBLE"
            print(f"{name:20} (:{port}): {status}")
            return result == 0
    except Exception as e:
        print(f"{name:20} (:{port}): ❌ ERROR - {e}")
        return False

def main():
    print("🔍 NOVA AI SYSTEM STATUS CHECK")
    print("=" * 50)
    
    # Check core services
    backend_ok = check_service("localhost", 8000, "Backend API")
    frontend_ok = check_service("localhost", 5173, "Frontend UI")
    redis_ok = check_service("localhost", 6379, "Redis")
    searxng_ok = check_service("localhost", 8080, "SearxNG")
    
    print()
    print("📊 SUMMARY:")
    total_services = 4
    running_services = sum([backend_ok, frontend_ok, redis_ok, searxng_ok])
    
    print(f"Services running: {running_services}/{total_services}")
    
    if backend_ok and frontend_ok:
        print("✅ Core services (Backend + Frontend) are operational!")
        print("🌐 You can access the application at: http://localhost:5173")
        print("🔧 API is available at: http://localhost:8000")
    else:
        print("❌ Core services are not fully operational!")
        if not backend_ok:
            print("   - Backend API needs to be started")
        if not frontend_ok:
            print("   - Frontend UI needs to be started")
    
    print()
    print("🧪 Next steps:")
    print("1. Open browser test interface for detailed testing")
    print("2. Test API endpoints manually")
    print("3. Verify frontend-backend integration")

if __name__ == "__main__":
    main()
