#!/usr/bin/env python3
"""
Simple test to verify setup
"""
import sys
from pathlib import Path

# Add the current directory to Python path so imports work
sys.path.insert(0, str(Path.cwd()))

print("Testing Data Pipeline Setup...")
print("=" * 50)

# Test 1: Check directories
print("\n1. Checking directories:")
dirs_to_check = [
    "data-pipeline/configs",
    "data-pipeline/scripts/utils",
    "data-pipeline/data",
    "data-pipeline/logs"
]

for dir_path in dirs_to_check:
    if Path(dir_path).exists():
        print(f"   ✅ {dir_path} exists")
    else:
        print(f"   ❌ {dir_path} missing")

# Test 2: Check config files
print("\n2. Checking configuration files:")
config_files = [
    "data-pipeline/configs/pipeline_config.yaml",
    ".env"
]

for file_path in config_files:
    if Path(file_path).exists():
        print(f"   ✅ {file_path} exists")
    else:
        print(f"   ❌ {file_path} missing")

# Test 3: Check Python files exist
print("\n3. Checking Python modules exist:")
python_files = [
    "data-pipeline/scripts/utils/config_loader.py",
    "data-pipeline/scripts/utils/logging_config.py",
    "data-pipeline/scripts/utils/__init__.py"
]

for file_path in python_files:
    if Path(file_path).exists():
        print(f"   ✅ {file_path} exists")
    else:
        print(f"   ❌ {file_path} missing")

# Test 4: Try to import modules with fixed path
print("\n4. Testing Python imports:")

# Fix the import path
sys.path.insert(0, str(Path.cwd() / "data-pipeline"))

try:
    from scripts.utils.config_loader import config
    print("   ✅ Config loader imported successfully")
    
    # Test getting a config value
    project_name = config.get("project.name")
    print(f"   ✅ Project name: {project_name}")
except Exception as e:
    print(f"   ❌ Config import failed: {e}")
    print(f"      Tip: Check if config_loader.py has correct content")

try:
    from scripts.utils.logging_config import pipeline_logger
    print("   ✅ Logger imported successfully")
    
    # Test logging
    pipeline_logger.info("Test log message")
    print("   ✅ Logging successful")
except Exception as e:
    print(f"   ❌ Logging import failed: {e}")
    print(f"      Tip: Check if logging_config.py has correct content")

print("\n" + "=" * 50)
print("Setup test complete!")
print("\n📝 Note: If imports failed, make sure:")
print("   1. All Python files have been created with the correct content")
print("   2. The __init__.py files exist in scripts/utils/")
print("   3. You have installed: pip install python-dotenv pyyaml")