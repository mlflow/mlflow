import gorilla
import inspect
import pytest

import mlflow
from mlflow.utils.autologging_utils import (
    get_unspecified_default_args,
    log_fn_args_as_params,
    wrap_patch,
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
