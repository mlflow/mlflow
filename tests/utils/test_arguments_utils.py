import functools

import pytest

from mlflow.utils.arguments_utils import _get_arg_names


def only_positional(a, b):
    return a + b


def only_keyword(a=0, b=0):
    return a + b


def positional_and_keyword(a, b=0):
    return a + b


def keyword_only(*, a, b=0):
    return a + b


@functools.wraps(only_positional)
def wrapper(*args, **kwargs):
    return only_positional(*args, **kwargs)


@pytest.mark.parametrize(
    "func", [only_positional, only_keyword, positional_and_keyword, keyword_only, wrapper]
)
def test_get_arg_names(func):
    assert _get_arg_names(func) == ["a", "b"]
