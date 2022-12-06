import pytest
from mlflow.entities import Param
from tests.helper_functions import random_str, random_int


def _check(param, key, value):
    assert isinstance(param, Param)
    assert param.key == key
    assert param.value == value


@pytest.fixture(scope="module")
def key():
    yield random_str(random_int(10, 25))  # random string on size in range [10, 25]


@pytest.fixture(scope="module")
def value():
    yield random_str(random_int(55, 75))  # random string on size in range [55, 75]


def test_creation_and_hydration(key, value):

    param = Param(key, value)
    _check(param, key, value)

    as_dict = {"key": key, "value": value}
    assert dict(param) == as_dict

    proto = param.to_proto()
    param2 = Param.from_proto(proto)
    _check(param2, key, value)

    param3 = Param.from_dictionary(as_dict)
    _check(param3, key, value)
