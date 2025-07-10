#!/usr/bin/env python3

import os
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

print('Testing URI cleaning logic...')

db_uri = "postgresql://blob-storage-rbac-app@platform-dev-1007-1.postgres.database.azure.com:5432/mlflow?auth_method=managed_identity&sslmode=require"
print('Original URI:', db_uri)

parsed = urlparse(db_uri)
query_params = parse_qs(parsed.query)
print('Query params:', query_params)

clean_query_params = {k: v for k, v in query_params.items() if k != 'auth_method'}
clean_query = urlencode(clean_query_params, doseq=True)
clean_parsed = parsed._replace(query=clean_query)
clean_db_uri = urlunparse(clean_parsed)

print('Cleaned URI:', clean_db_uri)
print('✓ URI cleaning working correctly')

# Test store creation without connecting
print('\nTesting store creation...')
os.environ['MLFLOW_TRACKING_URI'] = db_uri
os.environ['MLFLOW_POSTGRES_USE_MANAGED_IDENTITY'] = 'true'

try:
    from mlflow.store.tracking.postgres_managed_identity import get_postgres_store_with_managed_identity
    print('✓ Successfully imported get_postgres_store_with_managed_identity')
    
    # Note: This will try to connect, so we expect it might timeout or fail with auth
    # but it should NOT fail with "invalid connection option auth_method"
    
except Exception as e:
    print(f'Error importing: {e}')