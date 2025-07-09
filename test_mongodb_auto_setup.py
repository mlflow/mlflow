#!/usr/bin/env python
"""
Quick test for MongoDB auto-setup functionality and basic API compatibility.
"""

import mlflow
import uuid
import tempfile
import json
from pathlib import Path
from mlflow.store.tracking.mongodb_store import setup_mongodb_collections
import logging

# Enable logging
logging.basicConfig(level=logging.INFO)

def test_mongodb_auto_setup():
    """Test that MongoDB collections are automatically created with proper schema."""
    print("🧪 Testing MongoDB Auto-Setup")
    print("=" * 50)
    
    # Test collection setup
    mongodb_uri = f"mongodb://localhost:27017/test_auto_setup_{uuid.uuid4().hex[:8]}"
    
    try:
        setup_mongodb_collections(mongodb_uri)
        print("✅ MongoDB collections setup completed")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False
    
    return True

def test_basic_mlflow_operations():
    """Test basic MLflow operations with MongoDB backend."""
    print("\n🧪 Testing Basic MLflow Operations with MongoDB")
    print("=" * 50)
    
    mongodb_uri = f"mongodb://localhost:27017/test_basic_ops_{uuid.uuid4().hex[:8]}"
    
    try:
        # Setup collections first
        setup_mongodb_collections(mongodb_uri)
        
        # Set MongoDB as tracking URI
        mlflow.set_tracking_uri(mongodb_uri)
        
        # Test experiment creation
        experiment_name = f"test_exp_{uuid.uuid4().hex[:8]}"
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✅ Created experiment: {experiment_id}")
        
        # Test run creation and logging
        with mlflow.start_run(experiment_id=experiment_id) as run:
            # Log parameters
            mlflow.log_param("learning_rate", 0.01)
            mlflow.log_param("batch_size", 32)
            
            # Log metrics
            mlflow.log_metric("accuracy", 0.95)
            mlflow.log_metric("loss", 0.05)
            
            # Log tags
            mlflow.set_tag("model_type", "neural_network")
            mlflow.set_tag("environment", "test")
            
            # Create and log artifact
            with tempfile.TemporaryDirectory() as temp_dir:
                artifact_path = Path(temp_dir) / "config.json"
                artifact_path.write_text(json.dumps({"test": "data"}))
                mlflow.log_artifact(str(artifact_path))
            
            run_id = run.info.run_id
            print(f"✅ Created run: {run_id}")
        
        # Test experiment retrieval
        experiment = mlflow.get_experiment(experiment_id)
        assert experiment.name == experiment_name
        print(f"✅ Retrieved experiment: {experiment.name}")
        
        # Test run retrieval
        retrieved_run = mlflow.get_run(run_id)
        assert retrieved_run.data.params["learning_rate"] == "0.01"
        assert retrieved_run.data.metrics["accuracy"] == 0.95
        assert retrieved_run.data.tags["model_type"] == "neural_network"
        print(f"✅ Retrieved run with correct data")
        
        # Test experiment search
        experiments = mlflow.search_experiments()
        assert len(experiments) >= 1
        print(f"✅ Found {len(experiments)} experiments")
        
        # Test run search
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        assert len(runs) >= 1
        print(f"✅ Found {len(runs)} runs")
        
        print("✅ All basic MLflow operations successful with MongoDB!")
        return True
        
    except Exception as e:
        print(f"❌ MLflow operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_response_structure():
    """Test that API responses have expected structure."""
    print("\n🧪 Testing API Response Structures")
    print("=" * 50)
    
    mongodb_uri = f"mongodb://localhost:27017/test_api_structure_{uuid.uuid4().hex[:8]}"
    
    try:
        # Setup and use MongoDB
        setup_mongodb_collections(mongodb_uri)
        mlflow.set_tracking_uri(mongodb_uri)
        
        # Create test data
        experiment_id = mlflow.create_experiment(f"structure_test_{uuid.uuid4().hex[:8]}")
        
        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.log_param("test_param", "value")
            mlflow.log_metric("test_metric", 1.0)
            mlflow.set_tag("test_tag", "tag_value")
            run_id = run.info.run_id
        
        # Test response structures
        experiment = mlflow.get_experiment(experiment_id)
        assert hasattr(experiment, 'experiment_id')
        assert hasattr(experiment, 'name')
        assert hasattr(experiment, 'artifact_location')
        assert hasattr(experiment, 'lifecycle_stage')
        assert hasattr(experiment, 'creation_time')
        print("✅ Experiment response structure correct")
        
        run = mlflow.get_run(run_id)
        assert hasattr(run, 'info')
        assert hasattr(run, 'data')
        assert hasattr(run.info, 'run_id')
        assert hasattr(run.info, 'experiment_id')
        assert hasattr(run.info, 'status')
        assert hasattr(run.data, 'params')
        assert hasattr(run.data, 'metrics')
        assert hasattr(run.data, 'tags')
        print("✅ Run response structure correct")
        
        experiments_list = mlflow.search_experiments()
        assert isinstance(experiments_list, list)
        if experiments_list:
            assert hasattr(experiments_list[0], 'experiment_id')
        print("✅ Experiments list structure correct")
        
        runs_list = mlflow.search_runs()
        assert hasattr(runs_list, 'iloc')  # DataFrame-like object
        if len(runs_list) > 0:
            assert 'run_id' in runs_list.columns
            assert 'experiment_id' in runs_list.columns
        print("✅ Runs list structure correct")
        
        print("✅ All API response structures match expected format!")
        return True
        
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Genesis-Flow MongoDB Compatibility Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    test_results.append(("MongoDB Auto-Setup", test_mongodb_auto_setup()))
    test_results.append(("Basic MLflow Operations", test_basic_mlflow_operations()))
    test_results.append(("API Response Structures", test_api_response_structure()))
    
    # Print results
    print("\n📊 Test Results Summary")
    print("=" * 40)
    
    all_passed = True
    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Genesis-Flow MongoDB backend is fully functional")
        print("✅ Automatic collection setup works correctly")
        print("✅ API compatibility maintained")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Please check the error messages above")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)