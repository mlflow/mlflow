"""
Tests for PostgreSQL with Managed Identity Support

This module tests the PostgreSQL tracking store with Azure Managed Identity authentication,
ensuring passwordless authentication works correctly in various environments.
"""

import os
import pytest
import unittest
from unittest.mock import patch, MagicMock, Mock
from urllib.parse import urlparse

from mlflow.store.tracking.postgres_managed_identity import (
    PostgresManagedIdentityAuth,
    create_postgres_engine_with_managed_identity,
    get_postgres_store_with_managed_identity,
    PostgresConfig,
    configure_from_environment,
)


class TestPostgresManagedIdentityAuth(unittest.TestCase):
    """Test PostgresManagedIdentityAuth class."""
    
    def test_get_azure_ad_token_no_environment(self):
        """Test token retrieval when not in Azure environment."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.run") as mock_run:
                # Simulate Azure CLI not available
                mock_run.side_effect = Exception("Azure CLI not found")
                token = PostgresManagedIdentityAuth.get_azure_ad_token()
                self.assertIsNone(token)
    
    @patch.dict(os.environ, {"IDENTITY_ENDPOINT": "http://localhost", "IDENTITY_HEADER": "secret"})
    @patch("requests.get")
    def test_get_app_service_token_success(self, mock_get):
        """Test successful token retrieval in App Service environment."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "test_token_123"}
        mock_get.return_value = mock_response
        
        token = PostgresManagedIdentityAuth._get_app_service_token()
        self.assertEqual(token, "test_token_123")
        
        # Verify correct parameters were used
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        self.assertEqual(call_args[0][0], "http://localhost")
        self.assertIn("api-version", call_args[1]["params"])
        self.assertIn("X-IDENTITY-HEADER", call_args[1]["headers"])
    
    @patch.dict(os.environ, {"IDENTITY_ENDPOINT": "http://localhost", "IDENTITY_HEADER": "secret"})
    @patch("requests.get")
    def test_get_app_service_token_failure(self, mock_get):
        """Test token retrieval failure in App Service environment."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        token = PostgresManagedIdentityAuth._get_app_service_token()
        self.assertIsNone(token)
    
    @patch("requests.get")
    def test_get_vm_token_success(self, mock_get):
        """Test successful token retrieval from VM IMDS."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "vm_token_456"}
        mock_get.return_value = mock_response
        
        token = PostgresManagedIdentityAuth._get_vm_token()
        self.assertEqual(token, "vm_token_456")
        
        # Verify IMDS endpoint was called
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        self.assertIn("169.254.169.254", call_args[0][0])
        self.assertEqual(call_args[1]["headers"]["Metadata"], "true")
    
    @patch("subprocess.run")
    def test_get_cli_token_success(self, mock_run):
        """Test successful token retrieval using Azure CLI."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = '{"accessToken": "cli_token_789"}'
        mock_run.return_value = mock_result
        
        token = PostgresManagedIdentityAuth._get_cli_token()
        self.assertEqual(token, "cli_token_789")
        
        # Verify Azure CLI command
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        self.assertEqual(call_args[0], "az")
        self.assertEqual(call_args[1], "account")
        self.assertEqual(call_args[2], "get-access-token")


class TestPostgresEngineCreation(unittest.TestCase):
    """Test PostgreSQL engine creation with Managed Identity."""
    
    @patch("mlflow.store.tracking.postgres_managed_identity.create_engine")
    @patch("mlflow.store.tracking.postgres_managed_identity.PostgresManagedIdentityAuth.get_azure_ad_token")
    def test_create_engine_with_managed_identity(self, mock_get_token, mock_create_engine):
        """Test engine creation with Managed Identity enabled."""
        mock_get_token.return_value = "test_token"
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        db_uri = "postgresql://user@server.postgres.database.azure.com:5432/db?auth_method=managed_identity"
        engine = create_postgres_engine_with_managed_identity(db_uri)
        
        # Verify engine was created
        self.assertIsNotNone(engine)
        mock_create_engine.assert_called_once()
        
        # Verify URI was modified to remove password if present
        call_args = mock_create_engine.call_args[0][0]
        parsed = urlparse(call_args)
        self.assertIsNone(parsed.password)
    
    @patch("mlflow.store.tracking.postgres_managed_identity.create_engine")
    def test_create_engine_without_managed_identity(self, mock_create_engine):
        """Test standard engine creation when Managed Identity is not requested."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        db_uri = "postgresql://user:password@server:5432/db"
        engine = create_postgres_engine_with_managed_identity(db_uri)
        
        # Verify standard engine creation
        mock_create_engine.assert_called_once_with(db_uri)
    
    @patch.dict(os.environ, {"MLFLOW_POSTGRES_USE_MANAGED_IDENTITY": "true"})
    @patch("mlflow.store.tracking.postgres_managed_identity.create_engine")
    @patch("mlflow.store.tracking.postgres_managed_identity.PostgresManagedIdentityAuth.get_azure_ad_token")
    def test_create_engine_with_env_var(self, mock_get_token, mock_create_engine):
        """Test engine creation with environment variable configuration."""
        mock_get_token.return_value = "env_token"
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        db_uri = "postgresql://user@server:5432/db"
        engine = create_postgres_engine_with_managed_identity(db_uri)
        
        # Verify Managed Identity was used based on env var
        self.assertIsNotNone(engine)


class TestPostgresConfig(unittest.TestCase):
    """Test PostgresConfig helper class."""
    
    def test_get_connection_string_with_managed_identity(self):
        """Test connection string generation with Managed Identity."""
        conn_str = PostgresConfig.get_connection_string(
            host="server.postgres.database.azure.com",
            database="mydb",
            username="user@tenant",
            use_managed_identity=True
        )
        
        self.assertIn("postgresql://", conn_str)
        self.assertIn("auth_method=managed_identity", conn_str)
        self.assertIn("sslmode=require", conn_str)
    
    def test_get_connection_string_without_managed_identity(self):
        """Test connection string generation without Managed Identity."""
        conn_str = PostgresConfig.get_connection_string(
            host="localhost",
            database="testdb",
            username="testuser",
            use_managed_identity=False,
            ssl_mode="disable"
        )
        
        self.assertIn("postgresql://", conn_str)
        self.assertNotIn("auth_method=managed_identity", conn_str)
        self.assertIn("sslmode=disable", conn_str)
    
    @patch("mlflow.store.tracking.postgres_managed_identity.PostgresManagedIdentityAuth.get_azure_ad_token")
    def test_validate_azure_environment(self, mock_get_token):
        """Test Azure environment validation."""
        with patch.dict(os.environ, {"IDENTITY_ENDPOINT": "http://localhost"}):
            mock_get_token.return_value = "test_token"
            
            results = PostgresConfig.validate_azure_environment()
            
            self.assertTrue(results["is_azure_environment"])
            self.assertTrue(results["managed_identity_available"])
            self.assertEqual(results["environment_type"], "app_service")
            self.assertEqual(len(results["issues"]), 0)


class TestEnvironmentConfiguration(unittest.TestCase):
    """Test environment variable based configuration."""
    
    @patch.dict(os.environ, {
        "MLFLOW_POSTGRES_HOST": "myserver.postgres.database.azure.com",
        "MLFLOW_POSTGRES_DATABASE": "mydb",
        "MLFLOW_POSTGRES_USERNAME": "myuser",
        "MLFLOW_POSTGRES_PORT": "5432",
        "MLFLOW_POSTGRES_USE_MANAGED_IDENTITY": "true",
        "MLFLOW_POSTGRES_SSL_MODE": "require"
    })
    def test_configure_from_environment_complete(self):
        """Test configuration from complete environment variables."""
        conn_str = configure_from_environment()
        
        self.assertIsNotNone(conn_str)
        self.assertIn("myserver.postgres.database.azure.com", conn_str)
        self.assertIn("mydb", conn_str)
        self.assertIn("myuser", conn_str)
        self.assertIn("auth_method=managed_identity", conn_str)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_configure_from_environment_missing(self):
        """Test configuration with missing environment variables."""
        conn_str = configure_from_environment()
        self.assertIsNone(conn_str)


class TestPostgresStoreIntegration(unittest.TestCase):
    """Test PostgreSQL store integration with Managed Identity."""
    
    @patch("mlflow.store.tracking.postgres_managed_identity.create_postgres_engine_with_managed_identity")
    def test_get_postgres_store_with_managed_identity(self, mock_create_engine):
        """Test store creation with Managed Identity."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        store_uri = "postgresql://user@server:5432/db?auth_method=managed_identity"
        artifact_uri = "gs://my-bucket/artifacts"
        
        store = get_postgres_store_with_managed_identity(store_uri, artifact_uri)
        
        self.assertIsNotNone(store)
        self.assertEqual(store.db_uri, store_uri)
        self.assertEqual(store.default_artifact_root, artifact_uri)
        mock_create_engine.assert_called_once_with(store_uri)


if __name__ == "__main__":
    unittest.main()