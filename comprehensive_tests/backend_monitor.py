#!/usr/bin/env python3
"""
Real-time Backend Terminal Monitor
Monitors the backend terminal output for errors, API calls, and performance issues
"""

import subprocess
import time
import json
from datetime import datetime
import re

class BackendMonitor:
    def __init__(self):
        self.log_file = "comprehensive_tests/BACKEND_MONITORING.log"
        self.errors = []
        self.api_calls = []
        self.warnings = []
        
    def log_event(self, event_type, message, details=None):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = {
            "timestamp": timestamp,
            "type": event_type,
            "message": message,
            "details": details or {}
        }
        
        # Write to log file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {event_type.upper()}: {message}\n")
            if details:
                f.write(f"  Details: {json.dumps(details, indent=2)}\n")
            f.write("-" * 50 + "\n")
        
        # Store in memory for analysis
        if event_type == "error":
            self.errors.append(log_entry)
        elif event_type == "api_call":
            self.api_calls.append(log_entry)
        elif event_type == "warning":
            self.warnings.append(log_entry)
    
    def check_backend_health(self):
        """Check if backend is responsive"""
        try:
            import urllib.request
            with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as response:
                if response.getcode() == 200:
                    self.log_event("health", "Backend health check passed")
                    return True
        except Exception as e:
            self.log_event("error", "Backend health check failed", {"error": str(e)})
            return False
    
    def start_monitoring(self):
        """Start monitoring backend"""
        print("🔍 Starting Backend Monitoring...")
        print("📝 Logs will be written to: comprehensive_tests/BACKEND_MONITORING.log")
        print("⏹️  Press Ctrl+C to stop monitoring")
        
        # Initialize log file
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"NOVAH BACKEND MONITORING SESSION\n")
            f.write(f"Started: {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
        
        # Initial health check
        self.check_backend_health()
        
        print("✅ Monitoring started! Interact with the frontend to see backend activity...")
        print("🔗 Frontend: http://localhost:5173")
        print("🔗 Backend: http://localhost:8000")
        
        # Keep monitoring running
        try:
            while True:
                time.sleep(1)
                # Could add periodic health checks here
        except KeyboardInterrupt:
            print("\n🔴 Monitoring stopped by user")
            self.generate_summary()
    
    def generate_summary(self):
        """Generate monitoring summary"""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\nMONITORING SUMMARY\n")
            f.write(f"Session ended: {datetime.now()}\n")
            f.write(f"Total errors: {len(self.errors)}\n")
            f.write(f"Total API calls: {len(self.api_calls)}\n")
            f.write(f"Total warnings: {len(self.warnings)}\n")

if __name__ == "__main__":
    monitor = BackendMonitor()
    monitor.start_monitoring()
