"""
Cleanup Script for Nova AI Project
Removes temporary files, old test files, and unnecessary artifacts
"""

import os
import shutil
from pathlib import Path

def cleanup_nova_project():
    """Clean up the Nova AI project directory"""
    project_root = Path(".")
    
    # Files and directories to remove
    cleanup_targets = [
        # Temporary Python files
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        
        # Old test files (keep only the good ones)
        "cors_test_server.py",
        "cors_test.html", 
        "debug_execution_endpoint.py",
        "debug_routes.py",
        "minimal_api.py",
        "redis_free_api.py",
        "run_api.py",
        "test_api.py",
        "test_endpoint.py",
        "verify_and_start.py",
        
        # Duplicate config files
        "config copy.ini",
        
        # Old screenshot directories
        "screenshots",
        ".screenshots",
        
        # Temporary logs
        ".logs",
        
        # Old conversation data
        "conversations",
        
        # Reports (keep structure but clean old files)
        "reports/*.pdf",
    ]
    
    preserved_files = [
        "comprehensive_tests",  # Keep our new test suite
        "api.py",  # Main API file
        "config.ini",  # Main config
        "requirements.txt",
        "docker-compose.yml",
        "README.md",
        "frontend",  # Keep entire frontend
        "sources",  # Keep source code
        "prompts",  # Keep prompts
    ]
    
    print("🧹 Nova AI Project Cleanup")
    print("=" * 40)
    
    files_removed = 0
    dirs_removed = 0
    
    # Walk through project directory
    for root, dirs, files in os.walk(project_root):
        root_path = Path(root)
        
        # Skip preserved directories
        if any(preserve in str(root_path) for preserve in preserved_files):
            continue
            
        # Remove __pycache__ directories
        if "__pycache__" in dirs:
            pycache_path = root_path / "__pycache__"
            try:
                shutil.rmtree(pycache_path)
                print(f"🗑️  Removed: {pycache_path}")
                dirs_removed += 1
            except Exception as e:
                print(f"❌ Could not remove {pycache_path}: {e}")
        
        # Remove specific files
        for file in files:
            file_path = root_path / file
            
            # Check if file should be removed
            if any(target in str(file_path) for target in cleanup_targets):
                try:
                    file_path.unlink()
                    print(f"🗑️  Removed: {file_path}")
                    files_removed += 1
                except Exception as e:
                    print(f"❌ Could not remove {file_path}: {e}")
    
    # Remove specific directories
    dirs_to_remove = [
        "conversations",
        ".logs", 
        "screenshots",
        ".screenshots",
        "a_v",  # Seems to be an old virtual environment
    ]
    
    for dir_name in dirs_to_remove:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            try:
                shutil.rmtree(dir_path)
                print(f"🗑️  Removed directory: {dir_path}")
                dirs_removed += 1
            except Exception as e:
                print(f"❌ Could not remove {dir_path}: {e}")
    
    print()
    print("✅ Cleanup Summary:")
    print(f"   Files removed: {files_removed}")
    print(f"   Directories removed: {dirs_removed}")
    print()
    print("📁 Preserved important files:")
    for preserve in preserved_files:
        if Path(preserve).exists():
            print(f"   ✓ {preserve}")
    
    print()
    print("🎯 Project is now cleaner and more organized!")

if __name__ == "__main__":
    cleanup_nova_project()
