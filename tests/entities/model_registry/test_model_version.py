import unittest

from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.registered_model import RegisteredModel
from tests.helper_functions import random_str


class TestModelVersion(unittest.TestCase):
    def _check(self, model_version, name, version):
        self.assertIsInstance(model_version, ModelVersion)
        self.assertEqual(model_version.registered_model.name, name)
        self.assertEqual(model_version.get_name(), name)
        self.assertEqual(model_version.version, version)

    def test_creation_and_hydration(self):
        name = random_str()
        rm = RegisteredModel(name)
        model_version = ModelVersion(rm, 100)
        self._check(model_version, name, 100)

        expected_dict = {"version": 100}
        model_version_as_dict = dict(model_version)
        self.assertEqual(model_version_as_dict["version"], 100)
        self.assertIsInstance(model_version_as_dict["registered_model"], RegisteredModel)
        self.assertEqual(model_version_as_dict["registered_model"].name, name)
        model_version_as_dict.pop("registered_model")
        self.assertEqual(model_version_as_dict, expected_dict)

        proto = model_version.to_proto()
        self.assertEqual(proto.registered_model.name, name)
        self.assertEqual(proto.version, 100)
        model_version_2 = ModelVersion.from_proto(proto)
        self._check(model_version_2, name, 100)

        expected_dict.update({"registered_model": RegisteredModel(name)})
        model_version_3 = ModelVersion.from_dictionary(expected_dict)
        self._check(model_version_3, name, 100)

    def test_string_repr(self):
        model_version = ModelVersion(RegisteredModel(name="myname"), version=4)
        assert str(model_version) == "<ModelVersion: registered_model=<RegisteredModel: " \
                                     "name='myname'>, version=4>"
