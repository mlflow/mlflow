import unittest

from mlflow.entities.model_registry.registered_model import RegisteredModel
from tests.helper_functions import random_str


class TestRegisteredModel(unittest.TestCase):
    def _check(self, registered_model, name):
        self.assertIsInstance(registered_model, RegisteredModel)
        self.assertEqual(registered_model.name, name)

    def test_creation_and_hydration(self):
        name = random_str()
        registered_model = RegisteredModel(name)
        self._check(registered_model, name)

        as_dict = {"name": name}
        self.assertEqual(dict(registered_model), as_dict)

        proto = registered_model.to_proto()
        self.assertEqual(proto.name, name)
        registered_model2 = RegisteredModel.from_proto(proto)
        self._check(registered_model2, name)

        registered_model3 = RegisteredModel.from_dictionary(as_dict)
        self._check(registered_model3, name)

    def test_string_repr(self):
        registered_model = RegisteredModel(name="myname")
        assert str(registered_model) == "<RegisteredModel: name='myname'>"
