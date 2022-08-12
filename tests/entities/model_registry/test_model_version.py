import unittest

import uuid

from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.entities.model_registry.registered_model import RegisteredModel
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
from tests.helper_functions import random_str


class TestModelVersion(unittest.TestCase):
    def _check(
        self,
        model_version,
        name,
        version,
        creation_timestamp,
        last_updated_timestamp,
        description,
        user_id,
        current_stage,
        source,
        run_id,
        status,
        status_message,
        tags,
    ):
        self.assertIsInstance(model_version, ModelVersion)
        self.assertEqual(model_version.name, name)
        self.assertEqual(model_version.version, version)
        self.assertEqual(model_version.creation_timestamp, creation_timestamp)
        self.assertEqual(model_version.last_updated_timestamp, last_updated_timestamp)
        self.assertEqual(model_version.description, description)
        self.assertEqual(model_version.user_id, user_id)
        self.assertEqual(model_version.current_stage, current_stage)
        self.assertEqual(model_version.source, source)
        self.assertEqual(model_version.run_id, run_id)
        self.assertEqual(model_version.status, status)
        self.assertEqual(model_version.status_message, status_message)
        self.assertEqual(model_version.tags, tags)

    def test_creation_and_hydration(self):
        name = random_str()
        t1, t2 = 100, 150
        source = "path/to/source"
        run_id = uuid.uuid4().hex
        run_link = "http://localhost:5000/path/to/run"
        tags = [ModelVersionTag("key", "value"), ModelVersionTag("randomKey", "not a random value")]
        mvd = ModelVersion(
            name,
            "5",
            t1,
            t2,
            "version five",
            "user 1",
            "Production",
            source,
            run_id,
            "READY",
            "Model version #5 is ready to use.",
            tags,
            run_link,
        )
        self._check(
            mvd,
            name,
            "5",
            t1,
            t2,
            "version five",
            "user 1",
            "Production",
            source,
            run_id,
            "READY",
            "Model version #5 is ready to use.",
            {tag.key: tag.value for tag in (tags or [])},
        )

        expected_dict = {
            "name": name,
            "version": "5",
            "creation_timestamp": t1,
            "last_updated_timestamp": t2,
            "description": "version five",
            "user_id": "user 1",
            "current_stage": "Production",
            "source": source,
            "run_id": run_id,
            "run_link": run_link,
            "status": "READY",
            "status_message": "Model version #5 is ready to use.",
            "tags": {tag.key: tag.value for tag in (tags or [])},
        }
        model_version_as_dict = dict(mvd)
        self.assertEqual(model_version_as_dict, expected_dict)

        proto = mvd.to_proto()
        self.assertEqual(proto.name, name)
        self.assertEqual(proto.version, "5")
        self.assertEqual(proto.status, ModelVersionStatus.from_string("READY"))
        self.assertEqual(proto.status_message, "Model version #5 is ready to use.")
        self.assertEqual(set([tag.key for tag in proto.tags]), set(["key", "randomKey"]))
        self.assertEqual(
            set([tag.value for tag in proto.tags]), set(["value", "not a random value"])
        )
        mvd_2 = ModelVersion.from_proto(proto)
        self._check(
            mvd_2,
            name,
            "5",
            t1,
            t2,
            "version five",
            "user 1",
            "Production",
            source,
            run_id,
            "READY",
            "Model version #5 is ready to use.",
            {tag.key: tag.value for tag in (tags or [])},
        )

        expected_dict.update({"registered_model": RegisteredModel(name)})
        expected_dict["tags"] = tags
        mvd_3 = ModelVersion.from_dictionary(expected_dict)
        self._check(
            mvd_3,
            name,
            "5",
            t1,
            t2,
            "version five",
            "user 1",
            "Production",
            source,
            run_id,
            "READY",
            "Model version #5 is ready to use.",
            {tag.key: tag.value for tag in (tags or [])},
        )

    def test_string_repr(self):
        model_version = ModelVersion(
            name="myname",
            version="43",
            creation_timestamp=12,
            last_updated_timestamp=100,
            description="This is a test model.",
            user_id="user one",
            current_stage="Archived",
            source="path/to/a/notebook",
            run_id="some run",
            run_link="http://localhost:5000/path/to/run",
            status="PENDING_REGISTRATION",
            status_message="Copying!",
            tags=[],
        )

        assert (
            str(model_version) == "<ModelVersion: creation_timestamp=12, "
            "current_stage='Archived', description='This is a test "
            "model.', last_updated_timestamp=100, "
            "name='myname', "
            "run_id='some run', run_link='http://localhost:5000/path/"
            "to/run', source='path/to/a/notebook', "
            "status='PENDING_REGISTRATION', status_message='Copying!', "
            "tags={}, user_id='user one', version='43'>"
        )
