import unittest

from mlflow.entities import Param, RunTag
from tests.helper_functions import random_str, random_int


class TestParam(unittest.TestCase):
    def _check(self, param, key, value):
        self.assertIsInstance(param, Param)
        self.assertEqual(param.key, key)
        self.assertEqual(param.value, value)

    def test_creation_and_hydration(self):
        key = random_str(random_int(10, 25))  # random string on size in range [10, 25]
        value = random_str(random_int(55, 75))  # random string on size in range [55, 75]

        param = Param(key, value)
        self._check(param, key, value)

        as_dict = {"key": key, "value": value}
        self.assertEqual(dict(param), as_dict)

        proto = param.to_proto()
        param2 = Param.from_proto(proto)
        self._check(param2, key, value)

        param3 = Param.from_dictionary(as_dict)
        self._check(param3, key, value)

    def test_param_equality(self):
        assert Param("abc", "def") == Param("abc", "def")
        assert Param("abc", "dif-val") != Param("abc", "def")
        assert Param("dif-key", "def") != Param("abc", "def")
        # We detect type differences when field values are otherwise identical
        assert Param(key="a", value="b") != RunTag(key="a", value="b")
