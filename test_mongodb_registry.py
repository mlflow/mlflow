#!/usr/bin/env python
"""Test script to verify MongoDB model registry is properly registered."""

import mlflow
from mlflow.tracking._model_registry.utils import _get_store_registry

def test_mongodb_registry():
    """Test if MongoDB is properly registered in the model registry."""
    
    print("🔍 Testing MongoDB Model Registry Registration...")
    
    # Get the store registry
    registry = _get_store_registry()
    
    # Check if mongodb is in the registry
    print(f"📋 Available schemes: {list(registry._registry.keys())}")
    
    if "mongodb" in registry._registry:
        print("✅ MongoDB scheme is registered!")
        
        # Try to get the MongoDB store builder
        try:
            store_builder = registry.get_store_builder("mongodb://localhost:27017/test")
            print("✅ MongoDB store builder retrieved successfully!")
            
            # Try to create a store instance
            store = store_builder("mongodb://localhost:27017/test")
            print(f"✅ MongoDB store created: {type(store)}")
            
        except Exception as e:
            print(f"❌ Error creating MongoDB store: {e}")
    else:
        print("❌ MongoDB scheme is NOT registered!")
    
    # Test setting registry URI
    print("\n🔗 Testing registry URI setting...")
    try:
        mlflow.set_registry_uri("mongodb://localhost:27017/test")
        registry_uri = mlflow.get_registry_uri()
        print(f"✅ Registry URI set to: {registry_uri}")
    except Exception as e:
        print(f"❌ Error setting registry URI: {e}")

if __name__ == "__main__":
    test_mongodb_registry()