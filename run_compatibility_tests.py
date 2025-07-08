#!/usr/bin/env python
"""
MLflow Compatibility Test Runner

This script runs comprehensive compatibility tests to verify that Genesis-Flow
with MongoDB backend provides 100% API compatibility with standard MLflow.

Usage:
    python run_compatibility_tests.py
    
Requirements:
    - MongoDB running on localhost:27017
    - pytest installed
    - All dependencies from requirements.txt
"""

import subprocess
import sys
import os
import time

def check_mongodb_connection():
    """Check if MongoDB is accessible."""
    try:
        import pymongo
        client = pymongo.MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster')
        print("✅ MongoDB connection successful")
        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("   Please ensure MongoDB is running on localhost:27017")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'pytest',
        'scikit-learn',
        'pandas',
        'numpy',
        'pymongo'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("   Please install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required dependencies are installed")
    return True

def run_compatibility_tests():
    """Run the full compatibility test suite."""
    print("🧪 Running MLflow Compatibility Tests with MongoDB Backend")
    print("=" * 65)
    
    # Check prerequisites
    if not check_mongodb_connection():
        return False
    
    if not check_dependencies():
        return False
    
    print("\n🚀 Starting compatibility test suite...")
    print("   This may take a few minutes to complete all tests.")
    
    # Run the test suite
    test_file = "tests/integration/test_mlflow_compatibility.py"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            test_file,
            "-v",
            "--tb=short",
            "--color=yes"
        ], capture_output=True, text=True)
        
        print("\n" + "="*65)
        print("TEST RESULTS")
        print("="*65)
        
        # Print stdout
        if result.stdout:
            print(result.stdout)
        
        # Print stderr if there are errors
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n🎉 ALL COMPATIBILITY TESTS PASSED!")
            print("✅ Genesis-Flow provides 100% MLflow API compatibility")
            print("✅ MongoDB backend is fully functional")
            return True
        else:
            print(f"\n❌ Some tests failed (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    """Main function."""
    print("MLflow-MongoDB Compatibility Test Runner")
    print("Genesis-Flow Compatibility Verification")
    print("-" * 40)
    
    start_time = time.time()
    
    success = run_compatibility_tests()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n⏱️  Total test duration: {duration:.2f} seconds")
    
    if success:
        print("\n✅ COMPATIBILITY VERIFICATION COMPLETE")
        print("   Genesis-Flow is ready for production use!")
        sys.exit(0)
    else:
        print("\n❌ COMPATIBILITY VERIFICATION FAILED")
        print("   Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()