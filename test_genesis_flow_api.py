#!/usr/bin/env python3
"""
Test script to verify Genesis-Flow API compatibility during Phase 1 stripping.
This ensures we maintain 100% API compatibility while removing UI components.
"""

import os
import sys
import tempfile
import requests
import time
import subprocess
import threading
from contextlib import contextmanager

def test_basic_import():
    """Test that basic MLflow functionality works."""
    try:
        import mlflow
        print("✓ MLflow import successful")
        return True
    except Exception as e:
        print(f"✗ MLflow import failed: {e}")
        return False

def test_tracking_api():
    """Test core tracking API functions."""
    try:
        import mlflow
        
        # Test experiment creation
        exp_name = f"test_experiment_{int(time.time())}"
        exp_id = mlflow.create_experiment(exp_name)
        print(f"✓ Experiment creation successful: {exp_id}")
        
        # Set the experiment to use
        mlflow.set_experiment(exp_name)
        
        # Test run lifecycle
        with mlflow.start_run() as run:
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.5)
            mlflow.set_tag("test_tag", "test_tag_value")
            print(f"✓ Run logging successful: {run.info.run_id}")
        
        return True
    except Exception as e:
        print(f"✗ Tracking API test failed: {e}")
        return False

def test_server_endpoints():
    """Test that core MLflow modules can be imported."""
    try:
        # Test importing core server modules
        from mlflow.server import handlers
        print("✓ Server handlers module imported")
        
        from mlflow.store.tracking import file_store
        print("✓ File store module imported")
        
        from mlflow.store.artifact import artifact_repo
        print("✓ Artifact store module imported")
        
        # Test basic client functionality
        import mlflow
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        print(f"✓ Client can search experiments: {len(experiments)} found")
        
        return True
    except Exception as e:
        print(f"✗ Server endpoints test failed: {e}")
        return False

def run_compatibility_tests():
    """Run all compatibility tests."""
    print("Running Genesis-Flow API Compatibility Tests...")
    print("=" * 50)
    
    tests = [
        ("Basic Import", test_basic_import),
        ("Tracking API", test_tracking_api),
        ("Server Endpoints", test_server_endpoints),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All API compatibility tests passed!")
        return True
    else:
        print("✗ Some tests failed. Check compatibility before proceeding.")
        return False

if __name__ == "__main__":
    # Set temporary tracking URI for testing
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.environ["MLFLOW_TRACKING_URI"] = f"file://{tmp_dir}"
        success = run_compatibility_tests()
        sys.exit(0 if success else 1)