import inspect
import time
import pytest
from unittest.mock import Mock, call
from unittest import mock


import mlflow
from mlflow.utils import gorilla
from mlflow.tracking.client import MlflowClient
from mlflow.utils.autologging_utils import (
    get_unspecified_default_args,
    log_fn_args_as_params,
    wrap_patch,
    resolve_input_example_and_signature,
    batch_metrics_logger,
)

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


@pytest.mark.large
@pytest.mark.parametrize(
    "user_args,user_kwargs,all_param_names,all_default_values,expected", two_default_test_args
)
@pytest.mark.usefixtures("reset_active_experiment")
def test_get_two_unspecified_default_args(
    user_args, user_kwargs, all_param_names, all_default_values, expected
):

    default_dict = get_unspecified_default_args(
        user_args, user_kwargs, all_param_names, all_default_values
    )

    assert default_dict == expected


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


@pytest.mark.large
@pytest.mark.parametrize(
    "user_args,user_kwargs,all_param_names,all_default_values,expected", three_default_test_args
)
@pytest.mark.usefixtures("reset_active_experiment")
def test_get_three_unspecified_default_args(
    user_args, user_kwargs, all_param_names, all_default_values, expected
):

    default_dict = get_unspecified_default_args(
        user_args, user_kwargs, all_param_names, all_default_values
    )

    assert default_dict == expected


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


@pytest.mark.large
@pytest.mark.parametrize("args,kwargs,expected", log_test_args)
@pytest.mark.usefixtures("reset_active_experiment")
def test_log_fn_args_as_params(args, kwargs, expected, start_run):  # pylint: disable=W0613
    log_fn_args_as_params(dummy_fn, args, kwargs)
    client = mlflow.tracking.MlflowClient()
    params = client.get_run(mlflow.active_run().info.run_id).data.params
    for arg, value in zip(["arg1", "arg2", "arg3"], expected):
        assert arg in params
        assert params[arg] == value


@pytest.mark.large
@pytest.mark.usefixtures("reset_active_experiment")
def test_log_fn_args_as_params_ignores_unwanted_parameters(start_run):  # pylint: disable=W0613
    args, kwargs, unlogged = ("arg1", {"arg2": "value"}, ["arg1", "arg2", "arg3"])
    log_fn_args_as_params(dummy_fn, args, kwargs, unlogged)
    client = mlflow.tracking.MlflowClient()
    params = client.get_run(mlflow.active_run().info.run_id).data.params
    assert len(params.keys()) == 0


def get_func_attrs(f):
    assert callable(f)

    return (f.__name__, f.__doc__, f.__module__, inspect.signature(f))


@pytest.mark.large
@pytest.mark.usefixtures("reset_active_experiment")
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


@pytest.mark.large
@pytest.mark.usefixtures("reset_active_experiment")
def test_wrap_patch_with_module():
    def new_log_param(key, value):
        """new mlflow.log_param"""
        return (key, value)

    before = get_func_attrs(mlflow.log_param)
    wrap_patch(mlflow, mlflow.log_param.__name__, new_log_param)
    after = get_func_attrs(mlflow.log_param)

    assert after == before
    assert mlflow.log_param("foo", "bar") == ("foo", "bar")


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


def test_batch_metrics_logger_logs_all_metrics(start_run,):  # pylint: disable=unused-argument
    run_id = mlflow.active_run().info.run_id
    with batch_metrics_logger(run_id) as metrics_logger:
        for i in range(100):
            metrics_logger.record_metrics({hex(i): i}, i)

    metrics_on_run = mlflow.tracking.MlflowClient().get_run(run_id).data.metrics

    for i in range(100):
        assert hex(i) in metrics_on_run
        assert metrics_on_run[hex(i)] == i


def test_batch_metrics_logger_runs_training_and_logging_in_correct_ratio(
    start_run,
):  # pylint: disable=unused-argument
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


def test_batch_metrics_logger_chunks_metrics_when_batch_logging(
    start_run,
):  # pylint: disable=unused-argument
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


def test_batch_metrics_logger_records_time_correctly(start_run,):  # pylint: disable=unused-argument
    with mock.patch.object(MlflowClient, "log_batch", wraps=lambda *args, **kwargs: time.sleep(1)):
        run_id = mlflow.active_run().info.run_id
        with batch_metrics_logger(run_id) as metrics_logger:
            metrics_logger.record_metrics({"x": 1}, step=0)

            assert metrics_logger.total_log_batch_time >= 1

            time.sleep(2)

            metrics_logger.record_metrics({"x": 1}, step=0)

            assert metrics_logger.total_training_time >= 2


def test_batch_metrics_logger_logs_timestamps_as_int_milliseconds(
    start_run,
):  # pylint: disable=unused-argument
    with mock.patch.object(MlflowClient, "log_batch") as log_batch_mock, mock.patch(
        "time.time", return_value=123.45678901234567890
    ):
        run_id = mlflow.active_run().info.run_id
        with batch_metrics_logger(run_id) as metrics_logger:
            metrics_logger.record_metrics({"x": 1}, step=0)

        _, kwargs = log_batch_mock.call_args

        logged_metric = kwargs["metrics"][0]

        assert logged_metric.timestamp == 123456


def test_batch_metrics_logger_continues_if_log_batch_fails(
    start_run,
):  # pylint: disable=unused-argument
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
