import pytest
from unittest import mock

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking._model_registry import fluent


def test_register_prompt_with_azure_ml():
    # Test that attempting to register a prompt with Azure ML fails with a clear error
    with mock.patch("mlflow.get_registry_uri", return_value="azureml://workspace.api.azureml.ms/mlflow/v1.0"):
        with pytest.raises(MlflowException, match="only available with the OSS MLflow Tracking Server"):
            fluent.register_prompt(
                name="test_prompt",
                template="This is a {{test}} prompt.",
            )