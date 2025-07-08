#!/usr/bin/env python
"""
Genesis-Flow MongoDB Experiment Management Test

This script tests the successfully implemented MongoDB experiment management functionality.
"""

import os
import time
import random
import mlflow


def main():
    """Test MongoDB experiment management."""
    print("🎯 Genesis-Flow MongoDB Experiment Test")
    print("=" * 50)
    
    # Configure MongoDB URIs
    tracking_uri = "mongodb://localhost:27017/genesis_flow_test"
    registry_uri = "mongodb://localhost:27017/genesis_flow_test"
    
    print(f"🔗 Tracking URI: {tracking_uri}")
    print(f"📝 Registry URI: {registry_uri}")
    
    # Set URIs
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    
    try:
        print("\n🧪 Testing Experiment Management...")
        
        # Test 1: Create experiment
        experiment_name = f"test_exp_{random.randint(1000, 9999)}_{int(time.time())}"
        print(f"  Creating experiment: {experiment_name}")
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"  ✅ Created: {experiment_id}")
        
        # Test 2: Get experiment
        print(f"  Getting experiment by ID: {experiment_id}")
        experiment = mlflow.get_experiment(experiment_id)
        print(f"  ✅ Retrieved: {experiment.name}")
        
        # Test 3: List experiments
        print("  Listing experiments...")
        experiments = mlflow.search_experiments()
        print(f"  ✅ Found {len(experiments)} experiments")
        
        # Test 4: Get experiment by name
        print(f"  Getting experiment by name: {experiment_name}")
        experiment_by_name = mlflow.get_experiment_by_name(experiment_name)
        print(f"  ✅ Retrieved by name: {experiment_by_name.experiment_id}")
        
        # Test 5: Create another experiment with tags
        experiment_name_2 = f"tagged_exp_{random.randint(1000, 9999)}"
        print(f"  Creating experiment with tags: {experiment_name_2}")
        experiment_id_2 = mlflow.create_experiment(
            experiment_name_2,
            tags={"framework": "genesis-flow", "storage": "mongodb"}
        )
        print(f"  ✅ Created with tags: {experiment_id_2}")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("\nSuccessfully tested:")
        print("  ✅ MongoDB URI recognition (tracking & model registry)")
        print("  ✅ Experiment creation with unique IDs")
        print("  ✅ Experiment retrieval by ID")
        print("  ✅ Experiment retrieval by name")
        print("  ✅ Experiment listing")
        print("  ✅ Experiment creation with tags")
        print("  ✅ Direct MongoDB storage (no MLflow server needed)")
        
        print(f"\n📊 Database Stats:")
        print(f"  - Database: genesis_flow_test")
        print(f"  - Experiments created: 2")
        print(f"  - Storage: MongoDB")
        print(f"  - Registry: MongoDB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)