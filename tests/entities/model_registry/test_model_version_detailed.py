import unittest

import uuid

from mlflow.entities.model_registry.model_version_detailed import ModelVersionDetailed
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.entities.model_registry.registered_model import RegisteredModel
from tests.helper_functions import random_str


class TestModelVersionDetailed(unittest.TestCase):
    def _check(self, model_version_detailed, name, version,
               creation_timestamp, last_updated_timestamp,
               description, user_id, current_stage, source, run_id,
               status, status_message):
        self.assertIsInstance(model_version_detailed, ModelVersionDetailed)
        self.assertEqual(model_version_detailed.registered_model.name, name)
        self.assertEqual(model_version_detailed.get_name(), name)
        self.assertEqual(model_version_detailed.version, version)
        self.assertEqual(model_version_detailed.creation_timestamp, creation_timestamp)
        self.assertEqual(model_version_detailed.last_updated_timestamp, last_updated_timestamp)
        self.assertEqual(model_version_detailed.description, description)
        self.assertEqual(model_version_detailed.user_id, user_id)
        self.assertEqual(model_version_detailed.current_stage, current_stage)
        self.assertEqual(model_version_detailed.source, source)
        self.assertEqual(model_version_detailed.run_id, run_id)
        self.assertEqual(model_version_detailed.status, status)
        self.assertEqual(model_version_detailed.status_message, status_message)

    def test_creation_and_hydration(self):
        name = random_str()
        rm = RegisteredModel(name)
        t1, t2 = 100, 150
        source = "path/to/source"
        run_id = uuid.uuid4().hex
        mvd = ModelVersionDetailed(rm, 5, t1, t2, "version five", "user 1", "Production",
                                   source, run_id, "READY", "Model version #5 is ready to use.")
        self._check(mvd, name, 5, t1, t2, "version five", "user 1",
                    "Production", source, run_id, "READY", "Model version #5 is ready to use.")

        expected_dict = {"version": 5,
                         "creation_timestamp": t1,
                         "last_updated_timestamp": t2,
                         "description": "version five",
                         "user_id": "user 1",
                         "current_stage": "Production",
                         "source": source,
                         "run_id": run_id,
                         "status": "READY",
                         "status_message": "Model version #5 is ready to use."}
        model_version_as_dict = dict(mvd)
        self.assertIsInstance(model_version_as_dict["registered_model"], RegisteredModel)
        self.assertEqual(model_version_as_dict["registered_model"].name, name)
        model_version_as_dict.pop("registered_model")
        self.assertEqual(model_version_as_dict, expected_dict)

        proto = mvd.to_proto()
        self.assertEqual(proto.model_version.registered_model.name, name)
        self.assertEqual(proto.model_version.version, 5)
        self.assertEqual(proto.status, ModelVersionStatus.from_string("READY"))
        self.assertEqual(proto.status_message, "Model version #5 is ready to use.")
        mvd_2 = ModelVersionDetailed.from_proto(proto)
        self._check(mvd_2, name, 5, t1, t2, "version five", "user 1",
                    "Production", source, run_id, "READY", "Model version #5 is ready to use.")

        expected_dict.update({"registered_model": RegisteredModel(name)})
        mvd_3 = ModelVersionDetailed.from_dictionary(expected_dict)
        self._check(mvd_3, name, 5, t1, t2, "version five", "user 1",
                    "Production", source, run_id, "READY", "Model version #5 is ready to use.")

    def test_string_repr(self):
        model_version = ModelVersionDetailed(RegisteredModel(name="myname"),
                                             version=43,
                                             creation_timestamp=12,
                                             last_updated_timestamp=100,
                                             description="This is a test model.",
                                             user_id="user one",
                                             current_stage="Archived",
                                             source="path/to/a/notebook",
                                             run_id="some run",
                                             status="PENDING_REGISTRATION",
                                             status_message="Copying!")
        assert str(model_version) == "<ModelVersionDetailed: creation_timestamp=12, " \
                                     "current_stage='Archived', description='This is a test " \
                                     "model.', last_updated_timestamp=100, " \
                                     "registered_model=<RegisteredModel: name='myname'>, " \
                                     "run_id='some run', source='path/to/a/notebook', " \
                                     "status='PENDING_REGISTRATION', status_message='Copying!', " \
                                     "user_id='user one', version=43>"
