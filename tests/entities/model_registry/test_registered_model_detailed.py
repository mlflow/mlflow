import unittest

from mlflow.entities.model_registry.model_version_detailed import ModelVersionDetailed
from mlflow.entities.model_registry.registered_model import RegisteredModel
from mlflow.entities.model_registry.registered_model_detailed import RegisteredModelDetailed
from tests.helper_functions import random_str


class TestRegisteredModel(unittest.TestCase):
    def _check(self, registered_model_detailed, name, creation_timestamp, last_updated_timestamp,
               description, latest_versions):
        self.assertIsInstance(registered_model_detailed, RegisteredModelDetailed)
        self.assertEqual(registered_model_detailed.name, name)
        self.assertEqual(registered_model_detailed.creation_timestamp, creation_timestamp)
        self.assertEqual(registered_model_detailed.last_updated_timestamp, last_updated_timestamp)
        self.assertEqual(registered_model_detailed.description, description)
        self.assertEqual(registered_model_detailed.last_updated_timestamp, last_updated_timestamp)
        self.assertEqual(registered_model_detailed.latest_versions, latest_versions)

    def test_creation_and_hydration(self):
        name = random_str()
        description = random_str()
        rmd_1 = RegisteredModelDetailed(name, 1, 2, description, [])
        self._check(rmd_1, name, 1, 2, description, [])

        as_dict = {"name": name,
                   "creation_timestamp": 1,
                   "last_updated_timestamp": 2,
                   "description": description,
                   "latest_versions": []}
        self.assertEqual(dict(rmd_1), as_dict)

        proto = rmd_1.to_proto()
        self.assertEqual(proto.registered_model.name, name)
        self.assertEqual(proto.creation_timestamp, 1)
        self.assertEqual(proto.last_updated_timestamp, 2)
        self.assertEqual(proto.description, description)
        rmd_2 = RegisteredModelDetailed.from_proto(proto)
        self._check(rmd_2, name, 1, 2, description, [])

        rmd_3 = RegisteredModelDetailed.from_dictionary(as_dict)
        self._check(rmd_3, name, 1, 2, description, [])

    def test_with_latest_model_versions(self):
        name = random_str()
        mvd_1 = ModelVersionDetailed(RegisteredModel(name), 1, 1000, 2000,
                                     "version 1", "user 1", "Production", "source 1", "run ID 1",
                                     "FAILED_DELETION", "Model version is in production!")
        mvd_2 = ModelVersionDetailed(RegisteredModel(name), 4, 1300, 2002,
                                     "version 4", "user 2", "Staging", "source 4", "run ID 12",
                                     "READY", "Model copied over!")
        as_dict = {"name": name,
                   "creation_timestamp": 1,
                   "last_updated_timestamp": 4000,
                   "description": random_str(),
                   "latest_versions": [mvd_1, mvd_2]}
        rmd_1 = RegisteredModelDetailed.from_dictionary(as_dict)
        self.assertEqual(dict(rmd_1), as_dict)

        proto = rmd_1.to_proto()
        self.assertEqual(proto.creation_timestamp, 1)
        self.assertEqual(proto.last_updated_timestamp, 4000)
        self.assertEqual(set([mvd.model_version.version for mvd in proto.latest_versions]),
                         set([1, 4]))
        self.assertEqual(set([mvd.model_version.registered_model.name
                              for mvd in proto.latest_versions]),
                         set([name, ]))
        self.assertEqual(set([mvd.current_stage for mvd in proto.latest_versions]),
                         set(["Production", "Staging"]))
        self.assertEqual(set([mvd.last_updated_timestamp for mvd in proto.latest_versions]),
                         set([2000, 2002]))
        self.assertEqual(set([mvd.creation_timestamp for mvd in proto.latest_versions]),
                         set([1300, 1000]))

    def test_string_repr(self):
        rmd = RegisteredModelDetailed(name="myname",
                                      creation_timestamp=1000,
                                      last_updated_timestamp=2002,
                                      description="something about a model",
                                      latest_versions=[1, 2, "3"])
        assert str(rmd) == "<RegisteredModelDetailed: creation_timestamp=1000, " \
                           "description='something about a model', last_updated_timestamp=2002, " \
                           "latest_versions=[1, 2, '3'], name='myname'>"
