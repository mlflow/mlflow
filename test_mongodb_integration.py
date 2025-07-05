#!/usr/bin/env python3
"""
Test MongoDB Integration for Genesis-Flow

This test verifies that the MongoDB tracking store can be initialized
and basic experiment operations work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from mlflow.store.tracking.mongodb_store import MongoDBStore
from mlflow.store.tracking.mongodb_config import MongoDBConfig, get_mongodb_store_config
from mlflow.entities import ExperimentTag
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mongodb_config():
    """Test MongoDB configuration management."""
    print("Testing MongoDB Configuration...")
    print("=" * 50)
    
    # Test configuration retrieval
    try:
        config = MongoDBConfig.get_connection_config()
        print(f"✓ Configuration retrieved: {config['database_name']}")
        print(f"  URI scheme: {config['uri'].split('://')[0]}")
        print(f"  Is Cosmos DB: {config['is_cosmos_db']}")
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False
    
    # Test validation
    try:
        is_valid = MongoDBConfig.validate_configuration()
        if is_valid:
            print("✓ Configuration validation passed")
        else:
            print("⚠ Configuration validation had warnings")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False
    
    # Test connection string templates
    try:
        templates = MongoDBConfig.get_connection_string_template()
        print(f"✓ Connection templates available: {len(templates)}")
        for env, template in templates.items():
            print(f"  {env}: {template[:50]}...")
    except Exception as e:
        print(f"✗ Template generation failed: {e}")
        return False
    
    return True

def test_mongodb_store_initialization():
    """Test MongoDB store initialization without actual connection."""
    print("\nTesting MongoDB Store Initialization...")
    print("=" * 50)
    
    try:
        # Use test configuration to avoid requiring actual MongoDB
        test_config = MongoDBConfig.create_test_config(use_memory=False)
        
        # Test store initialization (without connecting)
        store = MongoDBStore(
            db_uri=test_config["uri"],
            default_artifact_root="azure://test-artifacts"
        )
        
        print("✓ MongoDB store instance created")
        print(f"  Database name: {store.database_name}")
        print(f"  Default artifact root: {store.default_artifact_root}")
        print(f"  Collections configured: {len([
            store.EXPERIMENTS_COLLECTION,
            store.RUNS_COLLECTION,
            store.PARAMS_COLLECTION,
            store.METRICS_COLLECTION,
            store.TAGS_COLLECTION
        ])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Store initialization failed: {e}")
        return False

def test_experiment_entity_conversion():
    """Test conversion between MongoDB documents and MLflow entities."""
    print("\nTesting Entity Conversion...")
    print("=" * 50)
    
    try:
        # Create test store instance
        test_config = MongoDBConfig.create_test_config()
        store = MongoDBStore(
            db_uri=test_config["uri"],
            default_artifact_root="azure://test-artifacts"
        )
        
        # Test experiment document to entity conversion
        test_experiment_doc = {
            "experiment_id": "123456789",
            "name": "test_experiment",
            "artifact_location": "azure://test-artifacts/123456789",
            "lifecycle_stage": "active",
            "creation_time": 1609459200000,
            "last_update_time": 1609459200000,
            "tags": [
                {"key": "model_type", "value": "classification"},
                {"key": "framework", "value": "scikit-learn"}
            ]
        }
        
        experiment = store._experiment_doc_to_entity(test_experiment_doc)
        
        print("✓ Experiment document converted to entity")
        print(f"  ID: {experiment.experiment_id}")
        print(f"  Name: {experiment.name}")
        print(f"  Tags: {len(experiment.tags)}")
        print(f"  Lifecycle: {experiment.lifecycle_stage}")
        
        # Test ID generation
        exp_id1 = store._generate_experiment_id()
        exp_id2 = store._generate_experiment_id()
        
        if exp_id1 != exp_id2:
            print("✓ Unique experiment ID generation")
        else:
            print("✗ Experiment ID generation not unique")
            return False
        
        # Test run UUID generation
        run_uuid1 = store._generate_run_uuid()
        run_uuid2 = store._generate_run_uuid()
        
        if run_uuid1 != run_uuid2 and len(run_uuid1) == 32:
            print("✓ Unique run UUID generation")
        else:
            print("✗ Run UUID generation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Entity conversion failed: {e}")
        return False

def test_collection_setup():
    """Test MongoDB collection and index setup."""
    print("\nTesting Collection Setup...")
    print("=" * 50)
    
    try:
        # Create test store instance
        test_config = MongoDBConfig.create_test_config()
        store = MongoDBStore(
            db_uri=test_config["uri"],
            default_artifact_root="azure://test-artifacts"
        )
        
        # Verify collection names are defined
        collections = [
            store.EXPERIMENTS_COLLECTION,
            store.RUNS_COLLECTION,
            store.PARAMS_COLLECTION,
            store.METRICS_COLLECTION,
            store.TAGS_COLLECTION
        ]
        
        print(f"✓ Collection names defined: {collections}")
        
        # Test that _initialize_collections doesn't throw errors
        store._initialize_collections()
        print("✓ Collection initialization completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Collection setup failed: {e}")
        return False

def test_azure_integration():
    """Test Azure integration helpers."""
    print("\nTesting Azure Integration...")
    print("=" * 50)
    
    try:
        from mlflow.store.tracking.mongodb_config import AzureIntegration
        
        # Test connection string retrieval (will be None in test environment)
        cosmos_uri = AzureIntegration.get_cosmos_db_connection_string()
        print(f"✓ Cosmos DB connection string check: {'Found' if cosmos_uri else 'Not configured (expected)'}")
        
        # Test blob storage artifact root
        blob_root = AzureIntegration.get_azure_blob_artifact_root()
        print(f"✓ Azure Blob artifact root: {blob_root}")
        
        # Test credential validation
        has_creds = AzureIntegration.validate_azure_credentials()
        print(f"✓ Azure credentials validation: {'Available' if has_creds else 'Not configured (expected)'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Azure integration test failed: {e}")
        return False

def main():
    """Run all MongoDB integration tests."""
    print("Genesis-Flow MongoDB Integration Tests")
    print("=" * 50)
    
    tests = [
        ("MongoDB Configuration", test_mongodb_config),
        ("Store Initialization", test_mongodb_store_initialization),
        ("Entity Conversion", test_experiment_entity_conversion),
        ("Collection Setup", test_collection_setup),
        ("Azure Integration", test_azure_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            print(f"\n❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"MongoDB Integration Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All MongoDB integration tests passed!")
        print("MongoDB store is ready for experiment operations.")
        return True
    else:
        print("❌ Some tests failed. Check configuration and dependencies.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)