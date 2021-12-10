# pylint: disable=unused-argument

import inspect
import time
import pytest
from collections import namedtuple
from unittest.mock import Mock, call
from unittest import mock

import mlflow
from mlflow.utils import gorilla
from mlflow.tracking.client import MlflowClient
from mlflow.utils.autologging_utils import (
    AUTOLOGGING_INTEGRATIONS,
    log_fn_args_as_params,
    resolve_input_example_and_signature,
    batch_metrics_logger,
    AutologgingEventLogger,
    BatchMetricsLogger,
    autologging_integration,
    get_autologging_config,
    autologging_is_disabled,
    get_instance_method_first_arg_value,
    get_method_call_arg_value,
)
from mlflow.utils.autologging_utils.safety import _wrap_patch, AutologgingSession
from mlflow.utils.autologging_utils.versioning import (
    FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY,
    _check_version_in_range,
    _is_pre_or_dev_release,
    _strip_dev_version_suffix,
    _violates_pep_440,
    is_flavor_supported_for_associated_package_versions,
)

from tests.autologging.fixtures import test_mode_off


pytestmark = pytest.mark.large


# Example function signature we are testing on
# def fn(arg1, default1=1, default2=2):
#     pass


two_default_test_args = [
    (["arg1", "default1"], {"default2": 42}, ["arg1", "default1", "default2"], [1, 2], {}),
    (["arg1", "default1", "default2"], {}, ["arg1", "default1", "default2"], [1, 2], {}),
    (["arg1"], {"default1": 42, "default2": 42}, ["arg1", "default1", "default2"], [1, 2], {}),
    (
        [],
        {"arg1": 42, "default1": 42, "default2": 42},
        ["arg1", "default1", "default2"],
        [1, 2],
        {},
    ),
    (["user_arg"], {"default1": 42}, ["arg1", "default1", "default2"], [1, 2], {"default2": 2}),
    (["user_arg"], {"default2": 42}, ["arg1", "default1", "default2"], [1, 2], {"default1": 1}),
    ([], {"arg1": 42, "default1": 42}, ["arg1", "default1", "default2"], [1, 2], {"default2": 2}),
    (["arg1", "default1"], {}, ["arg1", "default1", "default2"], [1, 2], {"default2": 2}),
    (["arg1"], {}, ["arg1", "default1", "default2"], [1, 2], {"default1": 1, "default2": 2}),
    ([], {"arg1": 42}, ["arg1", "default1", "default2"], [1, 2], {"default1": 1, "default2": 2}),
]


# Test function signature for the following tests
# def fn_default_default(default1=1, default2=2, default3=3):
#     pass


three_default_test_args = [
    (
        [],
        {},
        ["default1", "default2", "default3"],
        [1, 2, 3],
        {"default1": 1, "default2": 2, "default3": 3},
    ),
    (
        [],
        {"default2": 42},
        ["default1", "default2", "default3"],
        [1, 2, 3],
        {"default1": 1, "default3": 3},
    ),
]


@pytest.fixture
def start_run():
    mlflow.start_run()
    yield
    mlflow.end_run()


def dummy_fn(arg1, arg2="value2", arg3="value3"):  # pylint: disable=W0613
    pass


log_test_args = [
    ([], {"arg1": "value_x", "arg2": "value_y"}, ["value_x", "value_y", "value3"]),
    (["value_x"], {"arg2": "value_y"}, ["value_x", "value_y", "value3"]),
    (["value_x"], {"arg3": "value_z"}, ["value_x", "value2", "value_z"]),
    (["value_x", "value_y"], {}, ["value_x", "value_y", "value3"]),
    (["value_x", "value_y", "value_z"], {}, ["value_x", "value_y", "value_z"]),
    (
        [],
        {"arg1": "value_x", "arg2": "value_y", "arg3": "value_z"},
        ["value_x", "value_y", "value_z"],
    ),
]


@pytest.mark.parametrize("args,kwargs,expected", log_test_args)
def test_log_fn_args_as_params(args, kwargs, expected, start_run):  # pylint: disable=W0613
    log_fn_args_as_params(dummy_fn, args, kwargs)
    client = mlflow.tracking.MlflowClient()
    params = client.get_run(mlflow.active_run().info.run_id).data.params
    for arg, value in zip(["arg1", "arg2", "arg3"], expected):
        assert arg in params
        assert params[arg] == value


def test_log_fn_args_as_params_ignores_unwanted_parameters(start_run):  # pylint: disable=W0613
    args, kwargs, unlogged = ("arg1", {"arg2": "value"}, ["arg1", "arg2", "arg3"])
    log_fn_args_as_params(dummy_fn, args, kwargs, unlogged)
    client = mlflow.tracking.MlflowClient()
    params = client.get_run(mlflow.active_run().info.run_id).data.params
    assert len(params.keys()) == 0


def get_func_attrs(f):
    assert callable(f)

    return (f.__name__, f.__doc__, f.__module__, inspect.signature(f))


def test_wrap_patch_with_class():
    class Math:
        def add(self, a, b):
            """add"""
            return a + b

    def new_add(self, *args, **kwargs):
        """new add"""
        orig = gorilla.get_original_attribute(self, "add")
        return 2 * orig(*args, **kwargs)

    _wrap_patch(Math, Math.add.__name__, new_add)

    assert Math().add(1, 2) == 6


def sample_function_to_patch(a, b):
    return a + b


def test_wrap_patch_with_module():
    import sys

    this_module = sys.modules[__name__]

    def new_sample_function(a, b):
        """new mlflow.log_param"""
        return a - b

    assert sample_function_to_patch(10, 5) == 15
    _wrap_patch(this_module, sample_function_to_patch.__name__, new_sample_function)
    assert sample_function_to_patch(10, 5) == 5


@pytest.fixture()
def logger():
    return Mock()


def get_input_example():
    return "data"


def infer_model_signature(_):
    return "signature"


def test_if_getting_input_example_fails(logger):
    error_msg = "NoneType has no whatever"

    def throws():
        raise Exception(error_msg)

    input_example, signature = resolve_input_example_and_signature(
        throws, infer_model_signature, True, True, logger
    )

    assert input_example is None
    assert signature is None
    calls = [
        call("Failed to gather input example: " + error_msg),
        call(
            "Failed to infer model signature: "
            + "could not sample data to infer model signature: "
            + error_msg
        ),
    ]
    assert logger.warning.has_calls(calls)


def test_if_model_signature_inference_fails(logger):
    error_msg = "stack overflow"

    def throws(_):
        raise Exception(error_msg)

    input_example, signature = resolve_input_example_and_signature(
        get_input_example, throws, True, True, logger
    )

    assert input_example == "data"
    assert signature is None
    logger.warning.assert_called_with("Failed to infer model signature: " + error_msg)


def test_happy_path_works(logger):
    input_example, signature = resolve_input_example_and_signature(
        get_input_example, infer_model_signature, True, True, logger
    )

    assert input_example == "data"
    assert signature == "signature"
    logger.warning.assert_not_called()


def test_avoids_collecting_input_example_if_not_needed(logger):
    # We create a get_input_example that modifies the value of x
    # If get_input_example was not invoked, x should not have been modified.

    x = {"data": 0}

    def modifies():
        x["data"] = 1

    resolve_input_example_and_signature(modifies, infer_model_signature, False, False, logger)

    assert x["data"] == 0
    logger.warning.assert_not_called()


def test_avoids_inferring_signature_if_not_needed(logger):
    # We create an infer_model_signature that modifies the value of x
    # If infer_model_signature was not invoked, x should not have been modified.

    x = {"data": 0}

    def modifies(_):
        x["data"] = 1

    resolve_input_example_and_signature(get_input_example, modifies, True, False, logger)

    assert x["data"] == 0
    logger.warning.assert_not_called()


def test_batch_metrics_logger_logs_all_metrics(start_run):
    run_id = mlflow.active_run().info.run_id
    with batch_metrics_logger(run_id) as metrics_logger:
        for i in range(100):
            metrics_logger.record_metrics({hex(i): i}, i)

    metrics_on_run = mlflow.tracking.MlflowClient().get_run(run_id).data.metrics

    for i in range(100):
        assert hex(i) in metrics_on_run
        assert metrics_on_run[hex(i)] == i


def test_batch_metrics_logger_flush_logs_to_mlflow(start_run):
    run_id = mlflow.active_run().info.run_id

    # Need to patch _should_flush() to return False, so that we can manually flush the logger
    with mock.patch(
        "mlflow.utils.autologging_utils.BatchMetricsLogger._should_flush", return_value=False
    ):
        metrics_logger = BatchMetricsLogger(run_id)
        metrics_logger.record_metrics({"my_metric": 10}, 5)

        # Recorded metrics should not be logged to mlflow run before flushing BatchMetricsLogger
        metrics_on_run = mlflow.tracking.MlflowClient().get_run(run_id).data.metrics
        assert "my_metric" not in metrics_on_run

        metrics_logger.flush()

        # Recorded metric should be logged to mlflow run after flushing BatchMetricsLogger
        metrics_on_run = mlflow.tracking.MlflowClient().get_run(run_id).data.metrics
        assert "my_metric" in metrics_on_run
        assert metrics_on_run["my_metric"] == 10


def test_batch_metrics_logger_runs_training_and_logging_in_correct_ratio(start_run):
    with mock.patch.object(MlflowClient, "log_batch") as log_batch_mock:
        run_id = mlflow.active_run().info.run_id
        with batch_metrics_logger(run_id) as metrics_logger:
            metrics_logger.record_metrics({"x": 1}, step=0)  # data doesn't matter

            # first metrics should be logged immediately to record a previous timestamp and
            #   batch log time
            log_batch_mock.assert_called_once()

            metrics_logger.total_log_batch_time = 1
            metrics_logger.total_training_time = 1

            log_batch_mock.reset_mock()  # resets the 'calls' of this mock

            # the above 'training' took 1 second. So with target training-to-logging time ratio of
            #   10:1, 9 more 'training' should happen without sending the batch and then after the
            #   10th training the batch should be sent.
            for i in range(2, 11):
                metrics_logger.record_metrics({"x": 1}, step=0)
                log_batch_mock.assert_not_called()
                metrics_logger.total_training_time = i

            # at this point, average log batch time is 1, and total training time is 9
            # thus the next record_metrics call should send the batch.
            metrics_logger.record_metrics({"x": 1}, step=0)
            log_batch_mock.assert_called_once()

            # update log_batch time to reflect the 'mocked' training time
            metrics_logger.total_log_batch_time = 2

            log_batch_mock.reset_mock()  # reset the recorded calls

            for i in range(12, 21):
                metrics_logger.record_metrics({"x": 1}, step=0)
                log_batch_mock.assert_not_called()
                metrics_logger.total_training_time = i

            metrics_logger.record_metrics({"x": 1}, step=0)
            log_batch_mock.assert_called_once()


def test_batch_metrics_logger_chunks_metrics_when_batch_logging(start_run):
    with mock.patch.object(MlflowClient, "log_batch") as log_batch_mock:
        run_id = mlflow.active_run().info.run_id
        with batch_metrics_logger(run_id) as metrics_logger:
            metrics_logger.record_metrics({hex(x): x for x in range(5000)}, step=0)
            run_id = mlflow.active_run().info.run_id

            for call_idx, call in enumerate(log_batch_mock.call_args_list):
                _, kwargs = call

                assert kwargs["run_id"] == run_id
                assert len(kwargs["metrics"]) == 1000
                for metric_idx, metric in enumerate(kwargs["metrics"]):
                    assert metric.key == hex(call_idx * 1000 + metric_idx)
                    assert metric.value == call_idx * 1000 + metric_idx
                    assert metric.step == 0


def test_batch_metrics_logger_records_time_correctly(start_run):
    with mock.patch.object(MlflowClient, "log_batch", wraps=lambda *args, **kwargs: time.sleep(1)):
        run_id = mlflow.active_run().info.run_id
        with batch_metrics_logger(run_id) as metrics_logger:
            metrics_logger.record_metrics({"x": 1}, step=0)

            assert metrics_logger.total_log_batch_time >= 1

            time.sleep(2)

            metrics_logger.record_metrics({"x": 1}, step=0)

            assert metrics_logger.total_training_time >= 2


def test_batch_metrics_logger_logs_timestamps_as_int_milliseconds(start_run):
    with mock.patch.object(MlflowClient, "log_batch") as log_batch_mock, mock.patch(
        "time.time", return_value=123.45678901234567890
    ):
        run_id = mlflow.active_run().info.run_id
        with batch_metrics_logger(run_id) as metrics_logger:
            metrics_logger.record_metrics({"x": 1}, step=0)

        _, kwargs = log_batch_mock.call_args

        logged_metric = kwargs["metrics"][0]

        assert logged_metric.timestamp == 123456


def test_autologging_integration_calls_underlying_function_correctly():
    @autologging_integration("test_integration")
    def autolog(foo=7, disable=False, silent=False):
        return foo

    assert autolog(foo=10) == 10


def test_autologging_integration_stores_and_updates_config():
    @autologging_integration("test_integration")
    def autolog(foo=7, bar=10, disable=False, silent=False):
        return foo

    autolog()
    assert AUTOLOGGING_INTEGRATIONS["test_integration"] == {
        "foo": 7,
        "bar": 10,
        "disable": False,
        "silent": False,
    }
    autolog(bar=11)
    assert AUTOLOGGING_INTEGRATIONS["test_integration"] == {
        "foo": 7,
        "bar": 11,
        "disable": False,
        "silent": False,
    }
    autolog(6, disable=True)
    assert AUTOLOGGING_INTEGRATIONS["test_integration"] == {
        "foo": 6,
        "bar": 10,
        "disable": True,
        "silent": False,
    }
    autolog(1, 2, False, silent=True)
    assert AUTOLOGGING_INTEGRATIONS["test_integration"] == {
        "foo": 1,
        "bar": 2,
        "disable": False,
        "silent": True,
    }


def test_autologging_integration_forwards_positional_and_keyword_arguments_as_expected():
    @autologging_integration("test_integration")
    def autolog(foo=7, bar=10, disable=False, silent=False):
        return foo, bar, disable

    assert autolog(1, bar=2, disable=False) == (1, 2, False)


def test_autologging_integration_validates_structure_of_autolog_function():
    def fn_missing_disable_conf():
        pass

    def fn_bad_disable_conf_1(disable=True):
        pass

    # Try to use a falsy value that isn't "false"
    def fn_bad_disable_conf_2(disable=0):
        pass

    for fn in [fn_missing_disable_conf, fn_bad_disable_conf_1, fn_bad_disable_conf_2]:
        with pytest.raises(Exception, match="must specify a 'disable' argument"):
            autologging_integration("test")(fn)

    # Failure to apply the @autologging_integration decorator should not create a
    # placeholder for configuration state
    assert "test" not in AUTOLOGGING_INTEGRATIONS


def test_autologging_integration_makes_expected_event_logging_calls():
    @autologging_integration("test_success")
    def autolog_success(foo, bar=7, disable=False, silent=False):
        pass

    @autologging_integration("test_failure")
    def autolog_failure(biz, baz="val", disable=False, silent=False):
        raise Exception("autolog failed")

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

    autolog_success("a", bar=9, disable=True)
    assert len(logger.calls) == 1
    call = logger.calls[0]
    assert call.integration == "test_success"
    # NB: In MLflow > 1.13.1, the `call_args` argument to `log_autolog_called` is deprecated.
    # Positional arguments passed to `autolog()` should be forwarded to `log_autolog_called`
    # in keyword format
    assert call.call_args == ()
    assert call.call_kwargs == {"foo": "a", "bar": 9, "disable": True, "silent": False}

    logger.reset()

    with pytest.raises(Exception, match="autolog failed"):
        autolog_failure(82, disable=False, silent=True)
    assert len(logger.calls) == 1
    call = logger.calls[0]
    assert call.integration == "test_failure"
    # NB: In MLflow > 1.13.1, the `call_args` argument to `log_autolog_called` is deprecated.
    # Positional arguments passed to `autolog()` should be forwarded to `log_autolog_called`
    # in keyword format
    assert call.call_args == ()
    assert call.call_kwargs == {"biz": 82, "baz": "val", "disable": False, "silent": True}


@pytest.mark.usefixtures(test_mode_off.__name__)
def test_autologging_integration_succeeds_when_event_logging_throws_in_standard_mode():
    @autologging_integration("test")
    def autolog(disable=False, silent=False):
        return "result"

    class ThrowingLogger(AutologgingEventLogger):
        def __init__(self):
            self.logged_event = False

        def log_autolog_called(self, integration, call_args, call_kwargs):
            self.logged_event = True
            raise Exception("autolog failed")

    logger = ThrowingLogger()
    AutologgingEventLogger.set_logger(logger)
    assert autolog() == "result"
    assert logger.logged_event


def test_get_autologging_config_returns_configured_values_or_defaults_as_expected():

    assert get_autologging_config("nonexistent_integration", "foo") is None

    @autologging_integration("test_integration_for_config")
    def autolog(foo="bar", t=7, disable=False, silent=False):
        pass

    # Before `autolog()` has been invoked, config values should not be available
    assert get_autologging_config("test_integration_for_config", "foo") is None
    assert get_autologging_config("test_integration_for_config", "disable") is None
    assert get_autologging_config("test_integration_for_config", "silent") is None
    assert get_autologging_config("test_integration_for_config", "t", 10) == 10

    autolog()

    assert get_autologging_config("test_integration_for_config", "foo") == "bar"
    assert get_autologging_config("test_integration_for_config", "disable") is False
    assert get_autologging_config("test_integration_for_config", "silent") is False
    assert get_autologging_config("test_integration_for_config", "t", 10) == 7
    assert get_autologging_config("test_integration_for_config", "nonexistent") is None

    autolog(foo="baz", silent=True)

    assert get_autologging_config("test_integration_for_config", "foo") == "baz"
    assert get_autologging_config("test_integration_for_config", "silent") is True


def test_autologging_is_disabled_returns_expected_values():

    assert autologging_is_disabled("nonexistent_integration") is True

    @autologging_integration("test_integration_for_disable_check")
    def autolog(disable=False, silent=False):
        pass

    # Before `autolog()` has been invoked, `autologging_is_disabled` should return False
    assert autologging_is_disabled("test_integration_for_disable_check") is True

    autolog(disable=True)

    assert autologging_is_disabled("test_integration_for_disable_check") is True

    autolog(disable=False)

    assert autologging_is_disabled("test_integration_for_disable_check") is False


def test_autologging_disable_restores_behavior():
    import pandas as pd
    from sklearn.datasets import load_boston
    from sklearn.linear_model import LinearRegression

    mlflow.sklearn.autolog()

    dataset = load_boston()
    X = pd.DataFrame(dataset.data[:50, :8], columns=dataset.feature_names[:8])
    y = dataset.target[:50]

    # train a model
    model = LinearRegression()

    run = mlflow.start_run()
    model.fit(X, y)
    mlflow.end_run()
    run = MlflowClient().get_run(run.info.run_id)
    assert run.data.metrics
    assert run.data.params

    run = mlflow.start_run()
    with mlflow.utils.autologging_utils.disable_autologging():
        model.fit(X, y)
    mlflow.end_run()
    run = MlflowClient().get_run(run.info.run_id)
    assert not run.data.metrics
    assert not run.data.params

    run = mlflow.start_run()
    model.fit(X, y)
    mlflow.end_run()
    run = MlflowClient().get_run(run.info.run_id)
    assert run.data.metrics
    assert run.data.params


def test_autologging_event_logger_default_implementation_does_not_throw_for_valid_inputs():
    AutologgingEventLogger.set_logger(AutologgingEventLogger())

    class PatchObj:
        def test_fn(self):
            pass

    # Test successful autologging workflow
    AutologgingEventLogger.get_logger().log_autolog_called(
        "test_integration", ("a"), {"b": 1, "c": "d"}
    )
    AutologgingEventLogger.get_logger().log_patch_function_start(
        AutologgingSession("test_integration", "123"), PatchObj(), "test_fn", (1000), {"a": 2}
    )
    AutologgingEventLogger.get_logger().log_original_function_start(
        AutologgingSession("test_integration", "123"), PatchObj(), "test_fn", (1000), {"a": 2}
    )
    AutologgingEventLogger.get_logger().log_original_function_success(
        AutologgingSession("test_integration", "123"), PatchObj(), "test_fn", (1000), {"a": 2}
    )
    AutologgingEventLogger.get_logger().log_patch_function_success(
        AutologgingSession("test_integration", "123"), PatchObj(), "test_fn", (1000), {"a": 2}
    )

    # Test patch function failure autologging workflow
    AutologgingEventLogger.get_logger().log_patch_function_start(
        AutologgingSession("test_integration", "123"), PatchObj(), "test_fn", (1000), {"a": 2}
    )
    AutologgingEventLogger.get_logger().log_patch_function_error(
        AutologgingSession("test_integration", "123"),
        PatchObj(),
        "test_fn",
        (1000),
        {"a": 2},
        Exception("patch error"),
    )

    # Test original function failure autologging workflow
    AutologgingEventLogger.get_logger().log_patch_function_start(
        AutologgingSession("test_integration", "123"), PatchObj(), "test_fn", (1000), {"a": 2}
    )
    AutologgingEventLogger.get_logger().log_original_function_start(
        AutologgingSession("test_integration", "123"), PatchObj(), "test_fn", (1000), {"a": 2}
    )
    AutologgingEventLogger.get_logger().log_patch_function_error(
        AutologgingSession("test_integration", "123"),
        PatchObj(),
        "test_fn",
        (1000),
        {"a": 2},
        Exception("patch error"),
    )


def test_autologging_event_logger_default_impl_warns_for_log_autolog_called_with_deprecated_args():
    AutologgingEventLogger.set_logger(AutologgingEventLogger())

    with pytest.warns(DeprecationWarning, match="Received 1 positional arguments"):
        AutologgingEventLogger.get_logger().log_autolog_called(
            "test_integration",
            # call_args is deprecated in MLflow > 1.13.1; specifying a non-empty
            # value for this parameter should emit a warning
            call_args=("a"),
            call_kwargs={"b": "c"},
        )


def test_check_version_in_range():
    assert _check_version_in_range("1.0.2", "1.0.1", "1.0.3")
    assert _check_version_in_range("1.0.1", "1.0.1", "1.0.3")
    assert _check_version_in_range("1.0.3", "1.0.1", "1.0.3")
    assert not _check_version_in_range("1.0.0", "1.0.1", "1.0.3")
    assert not _check_version_in_range("1.0.4", "1.0.1", "1.0.3")
    assert not _check_version_in_range("0.99.99", "1.0.1", "1.0.3")
    assert not _check_version_in_range("1.1.0", "1.0.1", "1.0.3")
    assert _check_version_in_range("1.0.3", "1.0.1", "1.0.3.post1")


def test_is_pre_or_dev_release():
    assert _is_pre_or_dev_release("0.24.0rc1")
    assert _is_pre_or_dev_release("0.24.0dev1")
    assert not _is_pre_or_dev_release("0.24.0")


def test_strip_dev_version_suffix():
    assert _strip_dev_version_suffix("1.0.dev0") == "1.0"
    assert _strip_dev_version_suffix("1.0dev0") == "1.0"
    assert _strip_dev_version_suffix("1.0.dev") == "1.0"
    assert _strip_dev_version_suffix("1.0") == "1.0"


def test_violates_pep_440():
    assert _violates_pep_440("0.24.0-SNAPSHOT")
    assert not _violates_pep_440("0.24.0rc1")
    assert not _violates_pep_440("0.24.0dev1")
    assert not _violates_pep_440("0.24.0")


_module_version_info_dict_patch = {
    "sklearn": {
        "package_info": {"pip_release": "scikit-learn"},
        "autologging": {"minimum": "0.20.3", "maximum": "0.23.2"},
    },
    "pytorch-lightning": {
        "package_info": {"pip_release": "pytorch-lightning"},
        "autologging": {"minimum": "1.0.5", "maximum": "1.1.2"},
    },
    "tensorflow": {
        "package_info": {"pip_release": "tensorflow"},
        "autologging": {"minimum": "1.15.4", "maximum": "2.3.1"},
    },
    "keras": {
        "package_info": {"pip_release": "keras"},
        "autologging": {"minimum": "2.2.4", "maximum": "2.4.3"},
    },
    "xgboost": {
        "package_info": {"pip_release": "xgboost"},
        "autologging": {"minimum": "0.90", "maximum": "1.2.1"},
    },
    "lightgbm": {
        "package_info": {"pip_release": "lightgbm"},
        "autologging": {"minimum": "2.3.1", "maximum": "3.1.0"},
    },
    "gluon": {
        "package_info": {"pip_release": "mxnet"},
        "autologging": {"minimum": "1.5.1", "maximum": "1.7.0.post1"},
    },
    "fastai": {
        "package_info": {"pip_release": "fastai"},
        "autologging": {"minimum": "2.4.1", "maximum": "2.4.1"},
    },
    "statsmodels": {
        "package_info": {"pip_release": "statsmodels"},
        "autologging": {"minimum": "0.11.1", "maximum": "0.12.2"},
    },
    "spark": {
        "package_info": {"pip_release": "pyspark"},
        "autologging": {"minimum": "3.0.1", "maximum": "3.1.1"},
    },
}


@pytest.mark.parametrize(
    "flavor,module_version,expected_result",
    [
        ("fastai", "2.4.1", True),
        ("fastai", "2.3.1", False),
        ("fastai", "1.0.60", False),
        ("gluon", "1.6.1", True),
        ("gluon", "1.5.0", False),
        ("keras", "2.2.4", True),
        ("keras", "2.2.3", False),
        ("lightgbm", "2.3.1", True),
        ("lightgbm", "2.3.0", False),
        ("statsmodels", "0.11.1", True),
        ("statsmodels", "0.11.0", False),
        ("tensorflow", "1.15.4", True),
        ("tensorflow", "1.15.3", False),
        ("xgboost", "0.90", True),
        ("xgboost", "0.89", False),
        ("sklearn", "0.20.3", True),
        ("sklearn", "0.20.2", False),
        ("sklearn", "0.23.0rc1", False),
        ("sklearn", "0.23.0dev0", False),
        ("sklearn", "0.23.0-SNAPSHOT", False),
        ("pytorch", "1.0.5", True),
        ("pytorch", "1.0.4", False),
        ("pyspark.ml", "3.1.0", True),
        ("pyspark.ml", "3.0.0", False),
    ],
)
@mock.patch(
    "mlflow.utils.autologging_utils.versioning._module_version_info_dict",
    _module_version_info_dict_patch,
)
def test_is_autologging_integration_supported(flavor, module_version, expected_result):
    module_name, _ = FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY[flavor]
    with mock.patch(module_name + ".__version__", module_version):
        assert expected_result == is_flavor_supported_for_associated_package_versions(flavor)


@pytest.mark.parametrize(
    "flavor,module_version,expected_result",
    [
        ("pyspark.ml", "3.1.2.dev0", False),
        ("pyspark.ml", "3.1.1.dev0", True),
        ("pyspark.ml", "3.0.1.dev0", True),
        ("pyspark.ml", "3.0.0.dev0", False),
    ],
)
@mock.patch(
    "mlflow.utils.autologging_utils.versioning._module_version_info_dict",
    _module_version_info_dict_patch,
)
def test_dev_version_pyspark_is_supported_in_databricks(flavor, module_version, expected_result):
    module_name, _ = FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY[flavor]
    with mock.patch(module_name + ".__version__", module_version):
        # In Databricks
        with mock.patch(
            "mlflow.utils.autologging_utils.versioning.is_in_databricks_runtime",
            return_value=True,
        ) as mock_runtime:
            assert is_flavor_supported_for_associated_package_versions(flavor) == expected_result
            mock_runtime.assert_called()

        # Not in Databricks
        assert is_flavor_supported_for_associated_package_versions(flavor) is False


@mock.patch(
    "mlflow.utils.autologging_utils.versioning._module_version_info_dict",
    _module_version_info_dict_patch,
)
def test_disable_for_unsupported_versions_warning_sklearn_integration():
    log_warn_fn_name = "mlflow.utils.autologging_utils._logger.warning"
    log_info_fn_name = "mlflow.tracking.fluent._logger.info"

    def is_sklearn_warning_fired(log_warn_fn_args):
        return (
            "You are using an unsupported version of" in log_warn_fn_args[0][0]
            and log_warn_fn_args[0][1] == "sklearn"
        )

    def is_sklearn_autolog_enabled_info_fired(log_info_fn_args):
        return (
            "Autologging successfully enabled for " in log_info_fn_args[0][0]
            and log_info_fn_args[0][1] == "sklearn"
        )

    with mock.patch("sklearn.__version__", "0.20.3"):
        AUTOLOGGING_INTEGRATIONS.clear()
        with mock.patch(log_warn_fn_name) as log_warn_fn, mock.patch(
            log_info_fn_name
        ) as log_info_fn:
            mlflow.autolog(disable_for_unsupported_versions=True)
            assert all(not is_sklearn_warning_fired(args) for args in log_warn_fn.call_args_list)
            assert any(
                is_sklearn_autolog_enabled_info_fired(args) for args in log_info_fn.call_args_list
            )
        with mock.patch(log_warn_fn_name) as log_warn_fn, mock.patch(
            log_info_fn_name
        ) as log_info_fn:
            mlflow.autolog(disable_for_unsupported_versions=False)
            assert all(not is_sklearn_warning_fired(args) for args in log_warn_fn.call_args_list)
            assert any(
                is_sklearn_autolog_enabled_info_fired(args) for args in log_info_fn.call_args_list
            )

        with mock.patch(log_warn_fn_name) as log_warn_fn:
            mlflow.sklearn.autolog(disable_for_unsupported_versions=True)
            log_warn_fn.assert_not_called()
        with mock.patch(log_warn_fn_name) as log_warn_fn:
            mlflow.sklearn.autolog(disable_for_unsupported_versions=False)
            log_warn_fn.assert_not_called()

    with mock.patch("sklearn.__version__", "0.20.2"):
        AUTOLOGGING_INTEGRATIONS.clear()
        with mock.patch(log_warn_fn_name) as log_warn_fn, mock.patch(
            log_info_fn_name
        ) as log_info_fn:
            mlflow.autolog(disable_for_unsupported_versions=True)
            assert all(not is_sklearn_warning_fired(args) for args in log_warn_fn.call_args_list)
            assert all(
                not is_sklearn_autolog_enabled_info_fired(args)
                for args in log_info_fn.call_args_list
            )
        with mock.patch(log_warn_fn_name) as log_warn_fn, mock.patch(
            log_info_fn_name
        ) as log_info_fn:
            mlflow.autolog(disable_for_unsupported_versions=False)
            assert any(is_sklearn_warning_fired(args) for args in log_warn_fn.call_args_list)
            assert any(
                is_sklearn_autolog_enabled_info_fired(args) for args in log_info_fn.call_args_list
            )
        with mock.patch(log_warn_fn_name) as log_warn_fn:
            mlflow.sklearn.autolog(disable_for_unsupported_versions=True)
            log_warn_fn.assert_not_called()
        with mock.patch(log_warn_fn_name) as log_warn_fn:
            mlflow.sklearn.autolog(disable_for_unsupported_versions=False)
            assert log_warn_fn.call_count == 1 and is_sklearn_warning_fired(log_warn_fn.call_args)


def test_get_instance_method_first_arg_value():
    class Test:
        def f1(self, ab1, cd2):
            pass

        def f2(self, *args):
            pass

        def f3(self, *kwargs):
            pass

        def f4(self, *args, **kwargs):
            pass

    assert 3 == get_instance_method_first_arg_value(Test.f1, [3, 4], {})
    assert 3 == get_instance_method_first_arg_value(Test.f1, [3], {"cd2": 4})
    assert 3 == get_instance_method_first_arg_value(Test.f1, [], {"ab1": 3, "cd2": 4})
    assert 3 == get_instance_method_first_arg_value(Test.f2, [3, 4], {})
    with pytest.raises(AssertionError, match=""):
        get_instance_method_first_arg_value(Test.f3, [], {"ab1": 3, "cd2": 4})
    with pytest.raises(AssertionError, match=""):
        get_instance_method_first_arg_value(Test.f4, [], {"ab1": 3, "cd2": 4})


def test_get_method_call_arg_value():
    # suppose we call on a method defined like: `def f1(a, b=3, *, c=4, e=5)`
    assert 2 == get_method_call_arg_value(1, "b", 3, [1, 2], {})
    assert 3 == get_method_call_arg_value(1, "b", 3, [1], {})
    assert 2 == get_method_call_arg_value(1, "b", 3, [1], {"b": 2})
