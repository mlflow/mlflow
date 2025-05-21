#!/usr/bin/env python3
"""
Simple test script to verify that Azure ML URIs are properly rejected for prompt registration.
"""
import sys

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.prompt.registry_utils import is_prompt_supported_registry


def test_is_prompt_supported_registry():
    """Test that Azure ML URIs are not supported for prompts."""
    print("Testing is_prompt_supported_registry function...")
    
    # Test default local registry (should be supported)
    assert is_prompt_supported_registry(), "Default registry should be supported"
    
    # Test specific registry URIs
    assert is_prompt_supported_registry("sqlite:///some/path.db"), "SQLite registry should be supported"
    assert not is_prompt_supported_registry("databricks://some-workspace"), "Databricks registry should not be supported"
    assert not is_prompt_supported_registry("uc:some-catalog"), "UC registry should not be supported"
    assert not is_prompt_supported_registry("azureml://workspace.api.azureml.ms/mlflow/v1.0/subscriptions/xxx"), "Azure ML registry should not be supported"
    
    print("is_prompt_supported_registry tests passed!")


def test_register_prompt_with_azure_ml():
    """Test that trying to register a prompt with Azure ML fails with the right error."""
    print("Testing register_prompt with Azure ML URI...")
    
    # Save the original registry URI
    original_uri = mlflow.get_registry_uri()
    
    try:
        # Set a mock Azure ML registry URI
        registry_uri = "azureml://workspace.api.azureml.ms/mlflow/v1.0/subscriptions/xxx"
        mlflow.set_registry_uri(registry_uri)
        
        try:
            # This should fail with a specific error about Azure ML not being supported
            client = mlflow.MlflowClient(registry_uri=registry_uri)
            client.register_prompt(
                name="test_prompt",
                template="This is a {{test}} prompt.",
            )
            # If we get here, the test failed
            print("ERROR: register_prompt did not raise an exception for Azure ML URI")
            return False
        except MlflowException as e:
            # Check that the error message is the one we expect
            if "only available with the OSS MLflow Tracking Server" in str(e):
                print("register_prompt correctly rejected Azure ML URI")
                return True
            else:
                print(f"ERROR: Unexpected exception message: {e}")
                return False
    finally:
        # Restore the original registry URI
        mlflow.set_registry_uri(original_uri)


if __name__ == "__main__":
    test_is_prompt_supported_registry()
    success = test_register_prompt_with_azure_ml()
    
    # Exit with the appropriate status code
    sys.exit(0 if success else 1)