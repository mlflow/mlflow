from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.utils.rest_utils import MlflowHostCreds

ACTIVE_WORKSPACE = "team-a"


def test_model_registry_rest_store_workspace_guard():
    creds = MlflowHostCreds("https://hello")
    store = RestStore(lambda: creds)
    store._workspace_support = False

    with mock.patch(
        "mlflow.store.workspace_rest_store_mixin.get_request_workspace",
        return_value=ACTIVE_WORKSPACE,
    ):
        with pytest.raises(
            MlflowException,
            match="Active workspace 'team-a' cannot be used because the remote server does not",
        ):
            store.search_registered_models()
