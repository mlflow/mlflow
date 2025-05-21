import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.prompt.registry_utils import is_prompt_supported_registry, require_prompt_registry


def test_is_prompt_supported_registry():
    # Test default local registry (should be supported)
    assert is_prompt_supported_registry()
    
    # Test specific registry URIs
    assert is_prompt_supported_registry("sqlite:///some/path.db")
    assert not is_prompt_supported_registry("databricks://some-workspace")
    assert not is_prompt_supported_registry("uc:some-catalog")
    assert not is_prompt_supported_registry("azureml://workspace.api.azureml.ms/mlflow/v1.0/subscriptions/xxx")


def test_require_prompt_registry_decorator():
    # Define a test function with the decorator
    @require_prompt_registry
    def test_func():
        return "success"
    
    # Test with a supported registry
    with mlflow.start_run():
        assert test_func() == "success"
    
    # Test with an unsupported registry (Azure ML)
    with pytest.raises(MlflowException, match="only available with the OSS MLflow Tracking Server"):
        with mlflow.start_run():
            # Mock the registry URI
            original_get_registry_uri = mlflow.get_registry_uri
            try:
                mlflow.get_registry_uri = lambda: "azureml://workspace.api.azureml.ms/mlflow/v1.0"
                test_func()
            finally:
                mlflow.get_registry_uri = original_get_registry_uri