import unittest

from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.registered_model import RegisteredModel
from mlflow.entities.model_registry.registered_model_tag import RegisteredModelTag
from tests.helper_functions import random_str


class TestRegisteredModel(unittest.TestCase):
    def _check(
        self,
        registered_model,
        name,
        creation_timestamp,
        last_updated_timestamp,
        description,
        latest_versions,
        tags,
    ):
        assert isinstance(registered_model, RegisteredModel)
        assert registered_model.name == name
        assert registered_model.creation_timestamp == creation_timestamp
        assert registered_model.last_updated_timestamp == last_updated_timestamp
        assert registered_model.description == description
        assert registered_model.last_updated_timestamp == last_updated_timestamp
        assert registered_model.latest_versions == latest_versions
        assert registered_model.tags == tags

    def test_creation_and_hydration(self):
        name = random_str()
        description = random_str()
        rmd_1 = RegisteredModel(name, 1, 2, description, [], [])
        self._check(rmd_1, name, 1, 2, description, [], {})

        as_dict = {
            "name": name,
            "creation_timestamp": 1,
            "last_updated_timestamp": 2,
            "description": description,
            "latest_versions": [],
            "tags": {},
        }
        assert dict(rmd_1) == as_dict

        proto = rmd_1.to_proto()
        assert proto.name == name
        assert proto.creation_timestamp == 1
        assert proto.last_updated_timestamp == 2
        assert proto.description == description
        rmd_2 = RegisteredModel.from_proto(proto)
        self._check(rmd_2, name, 1, 2, description, [], {})
        as_dict["tags"] = []
        rmd_3 = RegisteredModel.from_dictionary(as_dict)
        self._check(rmd_3, name, 1, 2, description, [], {})

    def test_with_latest_model_versions(self):
        name = random_str()
        mvd_1 = ModelVersion(
            name,
            "1",
            1000,
            2000,
            "version 1",
            "user 1",
            "Production",
            "source 1",
            "run ID 1",
            "PENDING_REGISTRATION",
            "Model version is in production!",
        )
        mvd_2 = ModelVersion(
            name,
            "4",
            1300,
            2002,
            "version 4",
            "user 2",
            "Staging",
            "source 4",
            "run ID 12",
            "READY",
            "Model copied over!",
        )
        as_dict = {
            "name": name,
            "creation_timestamp": 1,
            "last_updated_timestamp": 4000,
            "description": random_str(),
            "latest_versions": [mvd_1, mvd_2],
            "tags": [],
        }
        rmd_1 = RegisteredModel.from_dictionary(as_dict)
        as_dict["tags"] = {}
        assert dict(rmd_1) == as_dict

        proto = rmd_1.to_proto()
        assert proto.creation_timestamp == 1
        assert proto.last_updated_timestamp == 4000
        assert {mvd.version for mvd in proto.latest_versions} == {"1", "4"}
        assert {mvd.name for mvd in proto.latest_versions} == {name}
        assert {mvd.current_stage for mvd in proto.latest_versions} == {"Production", "Staging"}
        assert {mvd.last_updated_timestamp for mvd in proto.latest_versions} == {2000, 2002}

        assert {mvd.creation_timestamp for mvd in proto.latest_versions} == {1300, 1000}

    def test_with_tags(self):
        name = random_str()
        tag1 = RegisteredModelTag("key", "value")
        tag2 = RegisteredModelTag("randomKey", "not a random value")
        tags = [tag1, tag2]
        as_dict = {
            "name": name,
            "creation_timestamp": 1,
            "last_updated_timestamp": 4000,
            "description": random_str(),
            "latest_versions": [],
            "tags": tags,
        }
        rmd_1 = RegisteredModel.from_dictionary(as_dict)
        as_dict["tags"] = {tag.key: tag.value for tag in (tags or [])}
        assert dict(rmd_1) == as_dict
        proto = rmd_1.to_proto()
        assert proto.creation_timestamp == 1
        assert proto.last_updated_timestamp == 4000
        assert {tag.key for tag in proto.tags} == {"key", "randomKey"}
        assert {tag.value for tag in proto.tags} == {"value", "not a random value"}

    def test_string_repr(self):
        rmd = RegisteredModel(
            name="myname",
            creation_timestamp=1000,
            last_updated_timestamp=2002,
            description="something about a model",
            latest_versions=["1", "2", "3"],
            tags=[],
        )
        assert (
            str(rmd) == "<RegisteredModel: creation_timestamp=1000, "
            "description='something about a model', last_updated_timestamp=2002, "
            "latest_versions=['1', '2', '3'], name='myname', tags={}>"
        )
