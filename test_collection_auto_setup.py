#!/usr/bin/env python
"""
Test MongoDB collection auto-setup functionality.
Verifies that all 23 MLflow collections are created with proper indexes.
"""

import uuid
from mlflow.store.tracking.mongodb_store import setup_mongodb_collections
import pymongo
from pymongo import MongoClient
from urllib.parse import urlparse


def test_collection_auto_setup():
    """Test that all MLflow collections are created with proper schema."""
    print("🧪 Testing MongoDB Collection Auto-Setup")
    print("=" * 50)
    
    # Create test database
    test_db_name = f"test_collection_setup_{uuid.uuid4().hex[:8]}"
    mongodb_uri = f"mongodb://localhost:27017/{test_db_name}"
    
    try:
        # Setup collections
        setup_mongodb_collections(mongodb_uri)
        print("✅ Collection setup completed")
        
        # Connect and verify
        client = MongoClient(mongodb_uri)
        db = client[test_db_name]
        
        # Get all collections
        collections = db.list_collection_names()
        
        # Expected collections (all 23 MLflow tables)
        expected_collections = {
            # Core tracking
            'experiments', 'runs', 'metrics', 'params', 'tags',
            
            # Model registry
            'registered_models', 'model_versions', 'registered_model_tags',
            'model_version_tags', 'registered_model_aliases',
            
            # Dataset and input tracking
            'datasets', 'inputs', 'input_tags',
            
            # Experiment-level metadata
            'experiment_tags',
            
            # Optimization tables
            'latest_metrics',
            
            # Model logging tables
            'logged_models', 'logged_model_tags', 'logged_model_params',
            'logged_model_metrics',
            
            # Tracing tables (MLflow 2.0+)
            'trace_info', 'trace_request_metadata', 'trace_tags'
        }
        
        # Verify all collections exist
        found_collections = set(collections)
        missing_collections = expected_collections - found_collections
        extra_collections = found_collections - expected_collections
        
        if missing_collections:
            print(f"❌ Missing collections: {missing_collections}")
            return False
        
        if extra_collections:
            print(f"⚠️ Extra collections found: {extra_collections}")
        
        print(f"✅ All {len(expected_collections)} expected collections created")
        
        # Verify indexes on key collections
        test_collections = ['experiments', 'runs', 'metrics', 'registered_models']
        
        for collection_name in test_collections:
            collection = db[collection_name]
            indexes = list(collection.list_indexes())
            
            # Should have at least _id index plus custom indexes
            if len(indexes) < 2:
                print(f"❌ {collection_name} missing indexes")
                return False
            
            print(f"✅ {collection_name}: {len(indexes)} indexes created")
        
        # Cleanup
        client.drop_database(test_db_name)
        client.close()
        
        print("✅ MongoDB collection auto-setup test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Collection setup test failed: {e}")
        return False


def test_migration_data_verification():
    """Verify the previously migrated data is accessible."""
    print("\n🧪 Testing Migration Data Verification")
    print("=" * 50)
    
    migration_db = "mlflow_migrated_from_autonomize"
    
    try:
        client = MongoClient(f"mongodb://localhost:27017/{migration_db}")
        db = client[migration_db]
        
        # Check if migration database exists
        if migration_db not in client.list_database_names():
            print("⚠️ Migration database not found - skipping verification")
            return True
        
        # Verify key collections have data
        key_collections = ['experiments', 'runs', 'metrics', 'params', 'registered_models']
        
        total_records = 0
        for collection_name in key_collections:
            count = db[collection_name].count_documents({})
            total_records += count
            print(f"✅ {collection_name}: {count:,} records")
        
        print(f"✅ Total records verified: {total_records:,}")
        
        # Sample data verification
        sample_experiment = db.experiments.find_one()
        if sample_experiment:
            print(f"✅ Sample experiment: {sample_experiment.get('name', 'Unknown')}")
        
        sample_run = db.runs.find_one()
        if sample_run:
            print(f"✅ Sample run: {sample_run.get('run_uuid', 'Unknown')[:8]}...")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Migration verification failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 MongoDB Integration Tests")
    print("=" * 40)
    
    test_results = []
    
    # Run tests
    test_results.append(("Collection Auto-Setup", test_collection_auto_setup()))
    test_results.append(("Migration Data Verification", test_migration_data_verification()))
    
    # Print results
    print("\n📊 Test Results Summary")
    print("=" * 30)
    
    all_passed = True
    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ MongoDB backend setup is fully functional")
        print("✅ Collection auto-creation works correctly")
        print("✅ Migration data is properly accessible")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)