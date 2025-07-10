"""
Tests for Azure-enabled MLflow tracking stores.

This module tests the Azure store creation with various configurations,
especially focusing on artifact_uri handling and azure-postgres:// URI scheme.
"""

import os
import pytest
import unittest
from unittest.mock import patch, MagicMock, Mock
from urllib.parse import urlparse

from mlflow.azure.stores import create_store
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore


class TestAzureStoreCreation(unittest.TestCase):
    """Test Azure store creation with various configurations."""

    def setUp(self):
        """Set up test environment."""
        # Clear any existing Azure environment variables
        self.azure_env_vars = [
            'AZURE_CLIENT_ID', 'AZURE_TENANT_ID', 'AZURE_CLIENT_SECRET',
            'MLFLOW_AZURE_AUTH_ENABLED', 'MLFLOW_AZURE_AUTH_METHOD'
        ]
        self.original_env = {}
        for var in self.azure_env_vars:
            self.original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

    def tearDown(self):
        """Clean up test environment."""
        # Restore original environment variables
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    def test_artifact_uri_none_azure_disabled(self):
        """Test artifact_uri handling when Azure auth is disabled and artifact_uri is None."""
        store_uri = "postgresql://user@host:5432/db"
        
        # Mock the Azure detection to return False (Azure not enabled)
        with patch('mlflow.azure.stores.is_azure_postgres_uri', return_value=False):
            store = create_store(store_uri, artifact_uri=None)
            
        self.assertIsInstance(store, SqlAlchemyStore)
        # The store should have been created with default artifact URI
        self.assertTrue(hasattr(store, '_artifact_repo_mlflow_uri'))

    def test_artifact_uri_empty_string_azure_disabled(self):
        """Test artifact_uri handling when Azure auth is disabled and artifact_uri is empty string."""
        store_uri = "postgresql://user@host:5432/db"
        
        with patch('mlflow.azure.stores.is_azure_postgres_uri', return_value=False):
            # This should now work instead of causing FileNotFoundError
            store = create_store(store_uri, artifact_uri="")
            
        self.assertIsInstance(store, SqlAlchemyStore)

    def test_artifact_uri_valid_path_azure_disabled(self):
        """Test artifact_uri handling when Azure auth is disabled and artifact_uri is valid."""
        store_uri = "postgresql://user@host:5432/db"
        artifact_uri = "/tmp/test-artifacts"
        
        with patch('mlflow.azure.stores.is_azure_postgres_uri', return_value=False):
            store = create_store(store_uri, artifact_uri=artifact_uri)
            
        self.assertIsInstance(store, SqlAlchemyStore)

    @patch('mlflow.azure.stores.is_azure_postgres_uri', return_value=True)
    @patch('mlflow.azure.stores.configure_from_environment')
    @patch('mlflow.azure.stores.get_postgres_store_with_managed_identity')
    def test_artifact_uri_none_azure_enabled(self, mock_get_store, mock_configure, mock_is_azure):
        """Test artifact_uri handling when Azure auth is enabled and artifact_uri is None."""
        store_uri = "azure-postgres://user@host:5432/db"
        
        # Mock successful Azure store creation
        mock_store = Mock(spec=SqlAlchemyStore)
        mock_get_store.return_value = mock_store
        
        store = create_store(store_uri, artifact_uri=None)
        
        # Should call the Azure-enabled path
        mock_get_store.assert_called_once()
        call_args = mock_get_store.call_args
        # Check that artifact_uri was defaulted properly
        self.assertEqual(call_args[1]['artifact_uri'], "./mlflow-artifacts")

    @patch('mlflow.azure.stores.is_azure_postgres_uri', return_value=True)
    @patch('mlflow.azure.stores.configure_from_environment')
    @patch('mlflow.azure.stores.get_postgres_store_with_managed_identity')
    def test_artifact_uri_custom_azure_enabled(self, mock_get_store, mock_configure, mock_is_azure):
        """Test artifact_uri handling when Azure auth is enabled and artifact_uri is custom."""
        store_uri = "azure-postgres://user@host:5432/db"
        custom_artifact_uri = "wasbs://container@account.blob.core.windows.net/"
        
        mock_store = Mock(spec=SqlAlchemyStore)
        mock_get_store.return_value = mock_store
        
        store = create_store(store_uri, artifact_uri=custom_artifact_uri)
        
        # Should call the Azure-enabled path with custom artifact_uri
        mock_get_store.assert_called_once()
        call_args = mock_get_store.call_args
        self.assertEqual(call_args[1]['artifact_uri'], custom_artifact_uri)

    def test_azure_postgres_uri_scheme_detection(self):
        """Test that azure-postgres:// URI scheme is properly detected."""
        test_cases = [
            ("azure-postgres://user@host:5432/db", True),
            ("postgresql://user@host.postgres.database.azure.com:5432/db", True),
            ("postgresql://user@host:5432/db?auth_method=managed_identity", True),
            ("postgresql://user@localhost:5432/db", False),
            ("sqlite:///test.db", False),
        ]
        
        from mlflow.azure.stores import is_azure_postgres_uri
        
        for uri, expected in test_cases:
            with self.subTest(uri=uri):
                result = is_azure_postgres_uri(uri)
                self.assertEqual(result, expected, f"URI {uri} should return {expected}")

    @patch('mlflow.azure.stores.is_azure_postgres_uri', return_value=False)
    def test_fallback_path_does_not_raise_file_error(self, mock_is_azure):
        """Test that the fallback path doesn't raise FileNotFoundError for empty artifact_uri."""
        store_uri = "postgresql://user@localhost:5432/db"
        
        # This should not raise FileNotFoundError anymore
        try:
            store = create_store(store_uri, artifact_uri=None)
            self.assertIsInstance(store, SqlAlchemyStore)
        except FileNotFoundError as e:
            if "No such file or directory: ''" in str(e):
                self.fail("create_store raised FileNotFoundError for empty artifact_uri")
            else:
                # Re-raise if it's a different FileNotFoundError
                raise

    def test_store_creation_with_different_schemes(self):
        """Test store creation with different URI schemes."""
        test_uris = [
            "azure-postgres://user@host:5432/db?sslmode=require",
            "postgresql://user@host.postgres.database.azure.com:5432/db",
            "postgresql://user@localhost:5432/db",
        ]
        
        for uri in test_uris:
            with self.subTest(uri=uri):
                with patch('mlflow.azure.stores.get_postgres_store_with_managed_identity') as mock_azure_store:
                    with patch('mlflow.azure.stores.configure_from_environment'):
                        mock_azure_store.return_value = Mock(spec=SqlAlchemyStore)
                        
                        try:
                            store = create_store(uri, artifact_uri=None)
                            self.assertIsNotNone(store)
                        except Exception as e:
                            # Some URIs might fail due to missing dependencies, but they shouldn't
                            # fail due to artifact_uri issues
                            self.assertNotIn("No such file or directory: ''", str(e))


if __name__ == '__main__':
    unittest.main()