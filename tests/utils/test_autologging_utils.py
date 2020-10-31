import inspect
import pytest
from unittest.mock import Mock, call
from unittest import mock
from mlflow.tracking.client import MlflowClient


import mlflow
from mlflow.utils import gorilla
from mlflow.utils.autologging_utils import (
    get_unspecified_default_args,
    log_fn_args_as_params,
    wrap_patch,
    resolve_input_example_and_signature,
    with_batch_metrics_handler,
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
def test_log_fn_args_as_params(args, kwargs, expected, start_run):  # pylint: disable=W0613
    log_fn_args_as_params(dummy_fn, args, kwargs)
    client = mlflow.tracking.MlflowClient()
    params = client.get_run(mlflow.active_run().info.run_id).data.params
    for arg, value in zip(["arg1", "arg2", "arg3"], expected):
        assert arg in params
        assert params[arg] == value


@pytest.mark.large
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


def test_batch_metrics_handler_logs_all_metrics():
    with mock.patch.object(MlflowClient, 'log_batch') as log_batch_mock:
        with with_batch_metrics_handler() as batch_metrics_handler:
            for i in range(100):
                batch_metrics_handler.record_metrics({x: i})

        # collect the args of all the logging calls
        recorded_metrics = []
        for call in log_batch_mock.call_args_list:
            recorded_metrics.append(call.kwargs['metrics'])

        desired_metrics = [{ 'x': i } for i in range(100)]
    
        assert recorded_metrics == desired_metrics

def test_batch_metrics_handler():
    with mock.patch.object(MlflowClient, 'log_batch') as log_batch_mock, mock.patch('mlflow.utils.autologging_utils.time_wrapper_for_log') as log_time_mock, mock.patch('mlflow.utils.autologging_utils.time_wrapper_for_current') as current_time_mock, mock.patch('mlflow.utils.autologging_utils.time_wrapper_for_timestamp') as timestamp_time_mock:
        current_time_mock.side_effect = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # training occurs every second
        log_time_mock.side_effect = [100, 101] # logging takes 1 second, numbers don't matter here
        timestamp_time_mock.side_effect = [9999] # this doesn't matter

        with with_batch_metrics_handler() as batch_metrics_handler:
            batch_metrics_handler.record_metrics({'x': 1}, 0) # data doesn't matter

            # first metrics should be logged immediately
            log_batch_mock.assert_called_with(metrics={'x': 1}, run_id=0)

            log_batch_mock.reset_mock() # resets the 'calls' of this mock

            # the above 'training' took 1 second. So with fudge factor of 10x, 10 more 'training' should happen before the metrics are sent.
            for _ in range(9):
                batch_metrics_handler.record_metrics({'x': 1}, 0)
                log_batch_mock.assert_not_called()

            batch_metrics_handler.record_metrics({'x': 1}, 0)
            log_batch_mock.assert_called_once()
        