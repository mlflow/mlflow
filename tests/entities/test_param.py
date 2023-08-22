from mlflow.entities import Param

from tests.helper_functions import random_int, random_str


def _check(param, key, value):
    assert isinstance(param, Param)
    assert param.key == key
    assert param.value == value


def test_creation_and_hydration():
    key = random_str(random_int(10, 25))  # random string on size in range [10, 25]
    value = random_str(random_int(55, 75))  # random string on size in range [55, 75]
    param = Param(key, value)
    _check(param, key, value)

    as_dict = {"key": key, "value": value}
    assert dict(param) == as_dict

    proto = param.to_proto()
    param2 = Param.from_proto(proto)
    _check(param2, key, value)

    param3 = Param.from_dictionary(as_dict)
    _check(param3, key, value)
