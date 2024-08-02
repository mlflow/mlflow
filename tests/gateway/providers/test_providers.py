import pytest

from mlflow.exceptions import MlflowException
from mlflow.gateway import providers
from mlflow.gateway.config import Provider


def test_check_all_providers_have_a_valid_mapping():
    for provider in Provider:
        if "databricks" in provider.lower():
            continue
        try:
            providers.get_provider(provider=provider)
        except MlflowException:
            pytest.fail(f"Provide not found {provider}")
