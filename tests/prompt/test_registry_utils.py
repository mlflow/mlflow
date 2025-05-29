import pytest
from unittest import mock

from mlflow.environment_variables import MLFLOW_ENABLE_UC_PROMPT_SUPPORT
from mlflow.prompt.registry_utils import is_prompt_supported_registry


class TestIsPromptSupportedRegistry:
    """Tests for the is_prompt_supported_registry function."""

    def test_oss_mlflow_registry_supported(self):
        """OSS MLflow registry should always support prompts."""
        # Various OSS registry URIs
        oss_uris = [
            "sqlite:///mlflow.db", 
            "postgresql://user:pass@localhost/mlflow",
            "mysql://user:pass@localhost/mlflow",
            "file:///tmp/mlflow",
            ""  # default/empty URI
        ]
        
        for uri in oss_uris:
            assert is_prompt_supported_registry(uri) is True, f"Failed for URI: {uri}"

    def test_legacy_databricks_workspace_not_supported(self):
        """Legacy Databricks workspace registry should not support prompts."""
        databricks_uris = ["databricks", "databricks://profile"]
        
        for uri in databricks_uris:
            assert is_prompt_supported_registry(uri) is False, f"Failed for URI: {uri}"

    def test_uc_registries_with_feature_flag_disabled(self):
        """UC registries should not support prompts when feature flag is disabled."""
        with mock.patch.object(MLFLOW_ENABLE_UC_PROMPT_SUPPORT, 'get', return_value=False):
            uc_uris = [
                "databricks-uc",
                "databricks-uc://profile",
                "uc:http://localhost:8080",
                "uc://localhost:8080"
            ]
            
            for uri in uc_uris:
                assert is_prompt_supported_registry(uri) is False, f"Failed for URI: {uri}"

    def test_uc_registries_with_feature_flag_enabled(self):
        """UC registries should support prompts when feature flag is enabled."""
        with mock.patch.object(MLFLOW_ENABLE_UC_PROMPT_SUPPORT, 'get', return_value=True):
            uc_uris = [
                "databricks-uc",
                "databricks-uc://profile",
                "uc:http://localhost:8080",
                "uc://localhost:8080"
            ]
            
            for uri in uc_uris:
                assert is_prompt_supported_registry(uri) is True, f"Failed for URI: {uri}"

    @mock.patch('mlflow.get_registry_uri')
    def test_none_registry_uri_uses_global_registry_uri(self, mock_get_registry_uri):
        """When registry_uri is None, should use mlflow.get_registry_uri()."""
        # Test with OSS URI
        mock_get_registry_uri.return_value = "sqlite:///mlflow.db"
        assert is_prompt_supported_registry(None) is True
        
        # Test with legacy Databricks URI
        mock_get_registry_uri.return_value = "databricks"
        assert is_prompt_supported_registry(None) is False
        
        # Test with UC URI and feature flag enabled
        mock_get_registry_uri.return_value = "databricks-uc"
        with mock.patch.object(MLFLOW_ENABLE_UC_PROMPT_SUPPORT, 'get', return_value=True):
            assert is_prompt_supported_registry(None) is True
        
        # Test with UC URI and feature flag disabled
        with mock.patch.object(MLFLOW_ENABLE_UC_PROMPT_SUPPORT, 'get', return_value=False):
            assert is_prompt_supported_registry(None) is False


class TestPromptRegistryFeatureFlagIntegration:
    """Integration tests for the prompt registry feature flag."""
    
    def test_feature_flag_enables_uc_support(self):
        """When the UC prompt support feature flag is True, UC registries should be supported."""
        from mlflow.prompt.registry_utils import require_prompt_registry, MlflowException
        
        @require_prompt_registry  
        def dummy_prompt_function():
            return "success"
        
        # Without feature flag (default False), should raise exception
        try:
            # Simulate calling from a UC registry context
            with mock.patch('mlflow.get_registry_uri', return_value='databricks-uc'):
                with pytest.raises(MlflowException, match="not supported"):
                    dummy_prompt_function()
        except Exception:
            pass  # Expected
        
        # With feature flag enabled, should work
        with mock.patch.object(MLFLOW_ENABLE_UC_PROMPT_SUPPORT, 'get', return_value=True):
            with mock.patch('mlflow.get_registry_uri', return_value='databricks-uc'):
                result = dummy_prompt_function()
                assert result == "success" 