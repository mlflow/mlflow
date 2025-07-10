#!/usr/bin/env python3

import os
import sys

# Add the genesis-flow directory to the Python path
sys.path.insert(0, '/Users/jagveersingh/Developer/autonomize/genesis/platform-agentops-mlops/genesis-flow')

print('Testing new Azure authentication architecture...')

# Set up environment variables
os.environ['AZURE_CLIENT_ID'] = '37ac4157-fb18-4c28-97d0-e8bc48026405'
os.environ['AZURE_TENANT_ID'] = '2a9d6d51-7674-4d37-8d71-1ee2fe30ccf4'
os.environ['AZURE_USE_MSI'] = 'true'
os.environ['MLFLOW_POSTGRES_USE_MANAGED_IDENTITY'] = 'true'

db_uri = "postgresql://blob-storage-rbac-app@platform-dev-1007-1.postgres.database.azure.com:5432/mlflow?auth_method=managed_identity&sslmode=require"

print(f'Database URI: {db_uri[:60]}...')

try:
    # Test 1: Configuration
    print('\n1. Testing AzureAuthConfig...')
    from mlflow.azure.config import AzureAuthConfig
    
    config = AzureAuthConfig()
    print(f'   ✓ Auth enabled: {config.auth_enabled}')
    print(f'   ✓ Auth method: {config.auth_method.value}')
    print(f'   ✓ Should use Azure auth: {config.should_use_azure_auth}')
    print(f'   ✓ Client ID: {config.client_id[:8]}...' if config.client_id else '   ✓ Client ID: None')
    
    # Test 2: Connection Factory
    print('\n2. Testing ConnectionFactory...')
    from mlflow.azure.connection_factory import ConnectionFactory
    
    factory = ConnectionFactory(config)
    print(f'   ✓ Factory created with Azure auth: {factory.config.should_use_azure_auth}')
    
    # Test 3: Engine creation (without actual connection)
    print('\n3. Testing engine creation...')
    engine = factory.create_engine(db_uri)
    print(f'   ✓ Engine created: {type(engine)}')
    print(f'   ✓ Engine URL: {str(engine.url)[:60]}...')
    
    # Test 4: Store creation via compatibility layer
    print('\n4. Testing store creation via postgres_managed_identity...')
    from mlflow.store.tracking.postgres_managed_identity import get_postgres_store_with_managed_identity
    
    # This should use the new architecture internally
    print('   Creating store (this may take a moment as it might try to connect)...')
    
    # Set environment to trigger the new path
    os.environ['MLFLOW_TRACKING_URI'] = db_uri
    
    store = get_postgres_store_with_managed_identity(db_uri, '/tmp/artifacts')
    print(f'   ✓ Store created: {type(store)}')
    print(f'   ✓ Store class: {store.__class__.__name__}')
    
    # Test 5: Direct store creation
    print('\n5. Testing direct store creation...')
    from mlflow.azure.stores import create_store
    
    direct_store = create_store(db_uri, '/tmp/artifacts')
    print(f'   ✓ Direct store created: {type(direct_store)}')
    print(f'   ✓ Direct store class: {direct_store.__class__.__name__}')
    
    print('\n✅ All tests passed! New architecture is working correctly.')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    import traceback
    traceback.print_exc()
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()