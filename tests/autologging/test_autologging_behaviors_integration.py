# pylint: disable=unused-argument

import importlib
import logging
import pytest
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from itertools import permutations
from unittest import mock

import mlflow
from mlflow.utils import gorilla
from mlflow.utils.autologging_utils import (
    safe_patch,
    get_autologging_config,
    autologging_is_disabled,
)

from tests.autologging.fixtures import test_mode_off
from tests.autologging.fixtures import reset_stderr  # pylint: disable=unused-import


pytestmark = pytest.mark.large


AUTOLOGGING_INTEGRATIONS_TO_TEST = {
    mlflow.sklearn: "sklearn",
    mlflow.keras: "keras",
    mlflow.xgboost: "xgboost",
    mlflow.lightgbm: "lightgbm",
    mlflow.pytorch: "torch",
    mlflow.gluon: "mxnet.gluon",
    mlflow.fastai: "fastai",
    mlflow.statsmodels: "statsmodels",
    mlflow.spark: "pyspark",
}


@pytest.fixture(autouse=True, scope="module")
def import_integration_libraries():
    for library_module in AUTOLOGGING_INTEGRATIONS_TO_TEST.values():
        importlib.import_module(library_module)


@pytest.fixture(autouse=True)
def disable_autologging_at_test_end():
    # The yeild statement is to insure that code below is executed as teardown code.
    # This will avoid bleeding of an active autologging session from test suite.
    yield
    for integration in AUTOLOGGING_INTEGRATIONS_TO_TEST:
        integration.autolog(disable=True)


@pytest.fixture()
def setup_keras_model():
    from keras.models import Sequential
    from keras.layers import Dense

    x = [1, 2, 3]
    y = [0, 1, 0]
    model = Sequential()
    model.add(Dense(12, input_dim=1, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return x, y, model


@pytest.mark.parametrize("integration", AUTOLOGGING_INTEGRATIONS_TO_TEST.keys())
def test_autologging_integrations_expose_configs_and_support_disablement(integration):
    for integration in AUTOLOGGING_INTEGRATIONS_TO_TEST:
        integration.autolog(disable=False)

    assert not autologging_is_disabled(integration.FLAVOR_NAME)
    assert not get_autologging_config(integration.FLAVOR_NAME, "disable", True)

    integration.autolog(disable=True)

    assert autologging_is_disabled(integration.FLAVOR_NAME)
    assert get_autologging_config(integration.FLAVOR_NAME, "disable", False)


@pytest.mark.parametrize("integration", AUTOLOGGING_INTEGRATIONS_TO_TEST.keys())
def test_autologging_integrations_use_safe_patch_for_monkey_patching(integration):
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
