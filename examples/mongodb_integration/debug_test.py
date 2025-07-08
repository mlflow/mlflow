#!/usr/bin/env python
"""
Debug Genesis-Flow MongoDB Registration
"""

import mlflow

def main():
    """Debug the registry issue."""
    print("🔍 Debug Genesis-Flow MongoDB Registration")
    print("=" * 50)
    
    # Check registries
    print("\n📋 Checking registries...")
    from mlflow.tracking._tracking_service.utils import _tracking_store_registry
    from mlflow.tracking._model_registry.utils import _get_store_registry
    
    tracking_schemes = list(_tracking_store_registry._registry.keys())
    print(f"Tracking schemes: {tracking_schemes}")
    
    model_registry = _get_store_registry()
    registry_schemes = list(model_registry._registry.keys())
    print(f"Registry schemes: {registry_schemes}")
    
    print(f"MongoDB in tracking: {'mongodb' in tracking_schemes}")
    print(f"MongoDB in registry: {'mongodb' in registry_schemes}")
    
    # Test URI setting
    print("\n🔗 Testing URI setting...")
    tracking_uri = "mongodb://localhost:27017/debug_test"
    
    print(f"Setting tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    
    # Don't set registry URI, let it default to tracking URI
    print(f"Registry URI (should default): {mlflow.get_registry_uri()}")
    
    # Test store creation directly
    print("\n🏪 Testing store creation...")
    try:
        from mlflow.tracking._tracking_service.utils import _get_store
        store = _get_store(tracking_uri)
        print(f"✅ Tracking store created: {type(store).__name__}")
    except Exception as e:
        print(f"❌ Tracking store creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        from mlflow.tracking._model_registry.utils import _get_store
        registry_store = _get_store(tracking_uri)
        print(f"✅ Registry store created: {type(registry_store).__name__}")
    except Exception as e:
        print(f"❌ Registry store creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test client creation step by step
    print("\n👤 Testing client creation...")
    try:
        from mlflow.tracking.client import MlflowClient
        print("Creating MlflowClient...")
        client = MlflowClient()
        print(f"✅ Client created successfully")
        print(f"Client tracking URI: {client._tracking_client.tracking_uri}")
        print(f"Client registry URI: {client._registry_uri}")
    except Exception as e:
        print(f"❌ Client creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()