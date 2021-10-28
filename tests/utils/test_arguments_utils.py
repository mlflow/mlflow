import functools

import pytest

from mlflow.utils.arguments_utils import _get_arg_names


def no_args():
    pass


def positional(a, b):
    return a, b


def keyword(a=0, b=0):
    return a, b


def positional_and_keyword(a, b=0):
    return a, b


def keyword_only(*, a, b=0):
    return a, b


def var_positional(*args):
    return args


def var_keyword(**kwargs):
    return kwargs


def var_positional_and_keyword(*args, **kwargs):
    return args, kwargs


@functools.wraps(positional)
def wrapper(*args, **kwargs):
    return positional(*args, **kwargs)


@pytest.mark.parametrize(
    "func, expected_args",
    [
        (no_args, []),
        (positional, ["a", "b"]),
        (keyword, ["a", "b"]),
        (positional_and_keyword, ["a", "b"]),
        (keyword_only, ["a", "b"]),
        (var_positional, ["args"]),
        (var_keyword, ["kwargs"]),
        (var_positional_and_keyword, ["args", "kwargs"]),
        (wrapper, ["a", "b"]),
    ],
)
def test_get_arg_names(func, expected_args):
    assert _get_arg_names(func) == expected_args
