# pylint: disable=unused-argument

import importlib
import copy
import inspect
import mock
import os
import pytest

import mlflow
import mlflow.utils.autologging_utils as autologging_utils
from mlflow.entities import RunStatus
from mlflow.tracking.client import MlflowClient
from mlflow.utils.autologging_utils import (
    safe_patch,
    autologging_integration,
    exception_safe_function,
    ExceptionSafeClass,
    PatchFunction,
    with_managed_run,
    _validate_args,
    _is_testing,
    get_autologging_config,
    autologging_is_disabled,
)


pytestmark = pytest.mark.large


AUTOLOGGING_INTEGRATIONS_TO_TEST = {
    mlflow.sklearn: "sklearn",
    mlflow.keras: "keras",
    mlflow.xgboost: "xgboost",
}

for library_module in AUTOLOGGING_INTEGRATIONS_TO_TEST.values():
    importlib.import_module(library_module)


@pytest.fixture
def test_mode_on():
    with mock.patch("mlflow.utils.autologging_utils._is_testing") as testing_mock:
        testing_mock.return_value = True
        assert autologging_utils._is_testing()
        yield


def test_autologging_integrations_expose_configs_and_support_disablement():
    mlflow.autolog()

    for integration in AUTOLOGGING_INTEGRATIONS_TO_TEST:
        assert not autologging_is_disabled(integration.FLAVOR_NAME)
        assert not get_autologging_config(integration.FLAVOR_NAME, "disable", True)

        integration.autolog(disable=True)

        assert autologging_is_disabled(integration.FLAVOR_NAME)
        assert get_autologging_config(integration.FLAVOR_NAME, "disable", False)




