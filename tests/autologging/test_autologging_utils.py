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
    log_fn_args_as_params,
    wrap_patch,
    resolve_input_example_and_signature,
    batch_metrics_logger,
    AutologgingEventLogger,
    AutologgingSession,
    BatchMetricsLogger,
    autologging_integration,
    get_autologging_config,
    autologging_is_disabled,
)
from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS


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

    before = get_func_attrs(Math.add)
    wrap_patch(Math, Math.add.__name__, new_add)
    after = get_func_attrs(Math.add)

    assert after == before
    assert Math().add(1, 2) == 6


def sample_function_to_patch(a, b):
    return a + b


def test_wrap_patch_with_module():
    import sys

    this_module = sys.modules[__name__]

    def new_sample_function(a, b):
        """new mlflow.log_param"""
        return a - b

    before_attrs = get_func_attrs(mlflow.log_param)
    assert sample_function_to_patch(10, 5) == 15

    wrap_patch(this_module, sample_function_to_patch.__name__, new_sample_function)
    after_attrs = get_func_attrs(mlflow.log_param)
    assert after_attrs == before_attrs
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


def test_batch_metrics_logger_logs_all_metrics(start_run,):
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


def test_batch_metrics_logger_runs_training_and_logging_in_correct_ratio(start_run,):
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


def test_batch_metrics_logger_chunks_metrics_when_batch_logging(start_run,):
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


def test_batch_metrics_logger_records_time_correctly(start_run,):
    with mock.patch.object(MlflowClient, "log_batch", wraps=lambda *args, **kwargs: time.sleep(1)):
        run_id = mlflow.active_run().info.run_id
        with batch_metrics_logger(run_id) as metrics_logger:
            metrics_logger.record_metrics({"x": 1}, step=0)

            assert metrics_logger.total_log_batch_time >= 1

            time.sleep(2)

            metrics_logger.record_metrics({"x": 1}, step=0)

            assert metrics_logger.total_training_time >= 2


def test_batch_metrics_logger_logs_timestamps_as_int_milliseconds(start_run,):
    with mock.patch.object(MlflowClient, "log_batch") as log_batch_mock, mock.patch(
        "time.time", return_value=123.45678901234567890
    ):
        run_id = mlflow.active_run().info.run_id
        with batch_metrics_logger(run_id) as metrics_logger:
            metrics_logger.record_metrics({"x": 1}, step=0)

        _, kwargs = log_batch_mock.call_args

        logged_metric = kwargs["metrics"][0]

        assert logged_metric.timestamp == 123456


@pytest.mark.usefixtures(test_mode_off.__name__)
def test_batch_metrics_logger_continues_if_log_batch_fails(start_run,):
    with mock.patch.object(MlflowClient, "log_batch") as log_batch_mock:
        log_batch_mock.side_effect = [Exception("asdf"), None]

        run_id = mlflow.active_run().info.run_id
        with batch_metrics_logger(run_id) as metrics_logger:
            # this call should fail to record since log_batch raised exception
            metrics_logger.record_metrics({"x": 1}, step=0)

            metrics_logger.record_metrics({"y": 2}, step=1)

        # even though the first call to log_batch failed, the BatchMetricsLogger should continue
        #   logging subsequent batches
        last_call = log_batch_mock.call_args_list[-1]

        _, kwargs = last_call

        assert kwargs["run_id"] == run_id
        assert len(kwargs["metrics"]) == 1
        metric = kwargs["metrics"][0]
        assert metric.key == "y"
        assert metric.value == 2
        assert metric.step == 1


def test_autologging_integration_calls_underlying_function_correctly():
    @autologging_integration("test_integration")
    def autolog(foo=7, disable=False):
        return foo

    assert autolog(foo=10) == 10


def test_autologging_integration_stores_and_updates_config():
    @autologging_integration("test_integration")
    def autolog(foo=7, bar=10, disable=False):
        return foo

    autolog()
    assert AUTOLOGGING_INTEGRATIONS["test_integration"] == {"foo": 7, "bar": 10, "disable": False}
    autolog(bar=11)
    assert AUTOLOGGING_INTEGRATIONS["test_integration"] == {"foo": 7, "bar": 11, "disable": False}
    autolog(6, disable=True)
    assert AUTOLOGGING_INTEGRATIONS["test_integration"] == {"foo": 6, "bar": 10, "disable": True}
    autolog(1, 2, False)
    assert AUTOLOGGING_INTEGRATIONS["test_integration"] == {"foo": 1, "bar": 2, "disable": False}


def test_autologging_integration_forwards_positional_and_keyword_arguments_as_expected():
    @autologging_integration("test_integration")
    def autolog(foo=7, bar=10, disable=False):
        return foo, bar, disable

    assert autolog(1, bar=2, disable=True) == (1, 2, True)


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
    def autolog_success(foo, bar=7, disable=False):
        pass

    @autologging_integration("test_failure")
    def autolog_failure(biz, baz="val", disable=False):
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
    assert call.call_kwargs == {"foo": "a", "bar": 9, "disable": True}

    logger.reset()

    with pytest.raises(Exception, match="autolog failed"):
        autolog_failure(82, disable=False)
    assert len(logger.calls) == 1
    call = logger.calls[0]
    assert call.integration == "test_failure"
    # NB: In MLflow > 1.13.1, the `call_args` argument to `log_autolog_called` is deprecated.
    # Positional arguments passed to `autolog()` should be forwarded to `log_autolog_called`
    # in keyword format
    assert call.call_args == ()
    assert call.call_kwargs == {"biz": 82, "baz": "val", "disable": False}


@pytest.mark.usefixtures(test_mode_off.__name__)
def test_autologging_integration_succeeds_when_event_logging_throws_in_standard_mode():
    @autologging_integration("test")
    def autolog(disable=False):
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
    def autolog(foo="bar", t=7, disable=False):
        pass

    # Before `autolog()` has been invoked, config values should not be available
    assert get_autologging_config("test_integration_for_config", "foo") is None
    assert get_autologging_config("test_integration_for_config", "disable") is None
    assert get_autologging_config("test_integration_for_config", "t", 10) == 10

    autolog()

    assert get_autologging_config("test_integration_for_config", "foo") == "bar"
    assert get_autologging_config("test_integration_for_config", "disable") is False
    assert get_autologging_config("test_integration_for_config", "t", 10) == 7
    assert get_autologging_config("test_integration_for_config", "nonexistent") is None

    autolog(foo="baz")

    assert get_autologging_config("test_integration_for_config", "foo") == "baz"


def test_autologging_is_disabled_returns_expected_values():

    assert autologging_is_disabled("nonexistent_integration") is True

    @autologging_integration("test_integration_for_disable_check")
    def autolog(disable=False):
        pass

    # Before `autolog()` has been invoked, `autologging_is_disabled` should return False
    assert autologging_is_disabled("test_integration_for_disable_check") is True

    autolog(disable=True)

    assert autologging_is_disabled("test_integration_for_disable_check") is True

    autolog(disable=False)

    assert autologging_is_disabled("test_integration_for_disable_check") is False


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
