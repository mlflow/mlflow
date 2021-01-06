import pytest
from collections import namedtuple
from unittest import mock

import mlflow
from mlflow.utils.autologging_utils import (
    get_autologging_config,
    autologging_is_disabled,
    AutologgingEventLogger,
)

import tensorflow
import keras
import fastai
import sklearn
import xgboost
import lightgbm
import statsmodels
import mxnet.gluon
import pyspark
import pytorch_lightning

from tests.autologging.fixtures import test_mode_off, test_mode_on

library_to_mlflow_module_without_pyspark = {
    tensorflow: mlflow.tensorflow,
    keras: mlflow.keras,
    fastai: mlflow.fastai,
    sklearn: mlflow.sklearn,
    xgboost: mlflow.xgboost,
    lightgbm: mlflow.lightgbm,
    statsmodels: mlflow.statsmodels,
    mxnet.gluon: mlflow.gluon,
    pytorch_lightning: mlflow.pytorch,
}

library_to_mlflow_module = {**library_to_mlflow_module_without_pyspark, pyspark: mlflow.spark}


@pytest.fixture(autouse=True)
def reset_global_states():
    for integration_name in library_to_mlflow_module.keys():
        try:
            del mlflow.utils.import_hooks._post_import_hooks[integration_name.__name__]
        except Exception:
            pass

    assert mlflow.utils.import_hooks._post_import_hooks == {}

    yield

    for integration_name in library_to_mlflow_module.keys():
        try:
            del mlflow.utils.import_hooks._post_import_hooks[integration_name.__name__]
        except Exception:
            pass

    assert mlflow.utils.import_hooks._post_import_hooks == {}


# We are pretending the module is not already imported (in reality it is, at the top of this file),
#   and is only imported when we call wrapt.notify_module_loaded in the tests below. Normally,
#   notify_module_loaded would be called by register_post_import_hook if it sees that the module
#   is already loaded.
def only_register(callback_fn, module, overwrite):  # pylint: disable=unused-argument
    mlflow.utils.import_hooks._post_import_hooks[module] = [callback_fn]


@pytest.fixture(autouse=True)
def disable_new_import_hook_firing_if_module_already_exists():
    with mock.patch("mlflow.tracking.fluent.register_post_import_hook", wraps=only_register):
        yield


@pytest.mark.large
@pytest.mark.usefixtures(test_mode_off.__name__)
@pytest.mark.parametrize("library,mlflow_module", library_to_mlflow_module.items())
def test_universal_autolog_does_not_throw_if_specific_autolog_throws_in_standard_mode(
    library, mlflow_module
):
    with mock.patch("mlflow." + mlflow_module.__name__ + ".autolog") as autolog_mock:
        autolog_mock.side_effect = Exception("asdf")
        mlflow.autolog()
        if library != pyspark:
            autolog_mock.assert_not_called()
        mlflow.utils.import_hooks.notify_module_loaded(library)
        autolog_mock.assert_called_once()


@pytest.mark.large
@pytest.mark.usefixtures(test_mode_on.__name__)
@pytest.mark.parametrize("library,mlflow_module", library_to_mlflow_module.items())
def test_universal_autolog_throws_if_specific_autolog_throws_in_test_mode(library, mlflow_module):
    with mock.patch("mlflow." + mlflow_module.__name__ + ".autolog") as autolog_mock:
        autolog_mock.side_effect = Exception("asdf")

        if library == pyspark:
            with pytest.raises(Exception, match="asdf"):
                # mlflow.autolog() invokes mlflow.spark.autolog() immediately, rather
                # than relying on import hooks; accordingly, we expect an exception
                # to be propagated as soon as mlflow.autolog() is called
                mlflow.autolog()
        else:
            mlflow.autolog()
            with pytest.raises(Exception, match="asdf"):
                mlflow.utils.import_hooks.notify_module_loaded(library)

        autolog_mock.assert_called_once()


@pytest.mark.large
@pytest.mark.parametrize("library,mlflow_module", library_to_mlflow_module_without_pyspark.items())
def test_universal_autolog_calls_specific_autologs_correctly(library, mlflow_module):
    integrations_with_additional_config = [xgboost, lightgbm, sklearn]
    args_to_test = {
        "log_models": False,
        "disable": True,
        "exclusive": True,
    }
    if library in integrations_with_additional_config:
        args_to_test.update({"log_input_examples": True, "log_model_signatures": True})

    mlflow.autolog(**args_to_test)
    mlflow.utils.import_hooks.notify_module_loaded(library)

    for arg_key, arg_value in args_to_test.items():
        assert get_autologging_config(mlflow_module.FLAVOR_NAME, arg_key, None) == arg_value


@pytest.mark.large
def test_universal_autolog_calls_pyspark_immediately():
    mlflow.autolog()
    assert not autologging_is_disabled(mlflow.spark.FLAVOR_NAME)

    mlflow.autolog(disable=True)
    assert autologging_is_disabled(mlflow.spark.FLAVOR_NAME)

    mlflow.autolog(disable=False)
    assert not autologging_is_disabled(mlflow.spark.FLAVOR_NAME)

    with mock.patch("mlflow.spark.autolog", wraps=mlflow.spark.autolog) as autolog_mock:
        # there should be no import hook on pyspark since autologging was already
        # applied to an active spark session
        mlflow.utils.import_hooks.notify_module_loaded(pyspark)
        autolog_mock.assert_not_called()


@pytest.mark.large
@pytest.mark.parametrize("config", [{"disable": False}, {"disable": True}])
def test_universal_autolog_attaches_pyspark_import_hook_if_pyspark_isnt_installed(config):
    with mock.patch("mlflow.spark.autolog", wraps=mlflow.spark.autolog) as autolog_mock:
        # simulate pyspark not being installed
        autolog_mock.side_effect = ImportError("no module named pyspark blahblah")

        mlflow.autolog(**config)
        autolog_mock.assert_called_once()  # it was called once and failed

        # now the user installs pyspark
        autolog_mock.side_effect = None

        mlflow.utils.import_hooks.notify_module_loaded(pyspark)

        # assert autolog is called again once pyspark is imported
        assert autolog_mock.call_count == 2
        assert autolog_mock.call_args_list[1] == config


@pytest.mark.large
def test_universal_autolog_makes_expected_event_logging_calls():
    class TestLogger(AutologgingEventLogger):

        LoggerCall = namedtuple("LoggerCall", ["integration", "call_args", "call_kwargs"])

        def __init__(self):
            self.calls = []

        def reset(self):
            self.calls = []

        def log_autolog_called(self, integration, call_args, call_kwargs):
            self.calls.append(TestLogger.LoggerCall(integration, call_args, call_kwargs))

    logger = TestLogger()
    AutologgingEventLogger.set_logger(logger)

    mlflow.autolog(exclusive=True, disable=True)

    universal_autolog_event_logging_calls = [
        call for call in logger.calls if call.integration == "mlflow"
    ]
    assert len(universal_autolog_event_logging_calls) == 1
    call = universal_autolog_event_logging_calls[0]
    assert call.integration == "mlflow"
    assert {"disable": True, "exclusive": True}.items() <= call.call_kwargs.items()
