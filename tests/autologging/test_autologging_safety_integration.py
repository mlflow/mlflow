# pylint: disable=unused-argument

import importlib
import pytest
from unittest import mock

import mlflow
from mlflow.utils import gorilla
from mlflow.utils.autologging_utils import (
    safe_patch,
    get_autologging_config,
    autologging_is_disabled,
)


pytestmark = pytest.mark.large


AUTOLOGGING_INTEGRATIONS_TO_TEST = {
    mlflow.sklearn: "sklearn",
    mlflow.keras: "keras",
    mlflow.xgboost: "xgboost",
}


@pytest.fixture(autouse=True, scope="module")
def import_integration_libraries():
    for library_module in AUTOLOGGING_INTEGRATIONS_TO_TEST.values():
        importlib.import_module(library_module)


@pytest.fixture(autouse=True)
def disable_autologging_at_test_end():
    for integration in AUTOLOGGING_INTEGRATIONS_TO_TEST:
        integration.autolog(disable=True)


def test_autologging_integrations_expose_configs_and_support_disablement():
    for integration in AUTOLOGGING_INTEGRATIONS_TO_TEST:
        integration.autolog(disable=False)

        assert not autologging_is_disabled(integration.FLAVOR_NAME)
        assert not get_autologging_config(integration.FLAVOR_NAME, "disable", True)

        integration.autolog(disable=True)

        assert autologging_is_disabled(integration.FLAVOR_NAME)
        assert get_autologging_config(integration.FLAVOR_NAME, "disable", False)


def test_autologging_integrations_use_safe_patch_for_monkey_patching():
    for integration in AUTOLOGGING_INTEGRATIONS_TO_TEST:
        with mock.patch(
            "mlflow.utils.gorilla.apply", wraps=gorilla.apply
        ) as gorilla_mock, mock.patch(
            integration.__name__ + ".safe_patch", wraps=safe_patch
        ) as safe_patch_mock:
            integration.autolog(disable=False)
            assert safe_patch_mock.call_count > 0
            # `safe_patch` leverages `gorilla.apply` in its implementation. Accordingly, we expect
            # that the total number of `gorilla.apply` calls to be equivalent to the number of
            # `safe_patch` calls. This verifies that autologging integrations are leveraging
            # `safe_patch`, rather than calling `gorilla.apply` directly (which does not provide
            # exception safety properties)
            assert safe_patch_mock.call_count == gorilla_mock.call_count
