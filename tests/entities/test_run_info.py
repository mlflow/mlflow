import unittest
import uuid

from mlflow.entities.run_info import RunInfo
from mlflow.entities.run_tag import RunTag
from tests.helper_functions import random_str, random_int


class TestRunInfo(unittest.TestCase):
    def _check(self, ri, run_uuid, experiment_id, name, source_type, source_name,
               entry_point_name, user_id, status, start_time, end_time, source_version,
               tags, artifact_uri):
        self.assertEqual(ri.run_uuid, run_uuid)
        self.assertEqual(ri.experiment_id, experiment_id)
        self.assertEqual(ri.name, name)
        self.assertEqual(ri.source_type, source_type)
        self.assertEqual(ri.source_name, source_name)
        self.assertEqual(ri.entry_point_name, entry_point_name)
        self.assertEqual(ri.user_id, user_id)
        self.assertEqual(ri.status, status)
        self.assertEqual(ri.start_time, start_time)
        self.assertEqual(ri.end_time, end_time)
        self.assertEqual(ri.source_version, source_version)
        self.assertSequenceEqual(ri.tags, tags)
        self.assertEqual(ri.artifact_uri, artifact_uri)

    @staticmethod
    def _create():
        run_uuid = str(uuid.uuid4())
        experiment_id = random_int(10, 2000)
        name = random_str(random_int(10, 40))
        source_type = random_int(1, 4)
        source_name = random_str(random_int(100, 300))
        entry_point_name = random_str(random_int(100, 300))
        user_id = random_str(random_int(10, 25))
        status = random_int(1, 5)
        start_time = random_int(1, 10)
        end_time = start_time + random_int(1, 10)
        source_version = random_str(random_int(10, 40))
        tags = [RunTag(key=random_str(random_int(1, 5)), value=random_str(random_int(1, 5)))
                for _ in range(2)]
        artifact_uri = random_str(random_int(10, 40))
        ri = RunInfo(run_uuid=run_uuid, experiment_id=experiment_id, name=name,
                     source_type=source_type, source_name=source_name,
                     entry_point_name=entry_point_name, user_id=user_id,
                     status=status, start_time=start_time, end_time=end_time,
                     source_version=source_version, tags=tags, artifact_uri=artifact_uri)
        return (ri, run_uuid, experiment_id, name, source_type, source_name, entry_point_name,
                user_id, status, start_time, end_time, source_version, tags, artifact_uri)

    def test_creation_and_hydration(self):
        (ri1, run_uuid, experiment_id, name, source_type, source_name, entry_point_name,
         user_id, status, start_time, end_time, source_version, tags, artifact_uri) = self._create()
        self._check(ri1, run_uuid, experiment_id, name, source_type, source_name, entry_point_name,
                    user_id, status, start_time, end_time, source_version, tags, artifact_uri)
        as_dict = {
            "run_uuid": run_uuid,
            "experiment_id": experiment_id,
            "name": name,
            "source_type": source_type,
            "source_name": source_name,
            "entry_point_name": entry_point_name,
            "user_id": user_id,
            "status": status,
            "start_time": start_time,
            "end_time": end_time,
            "source_version": source_version,
            "tags": tags,
            "artifact_uri": artifact_uri,
        }
        self.assertEqual(dict(ri1), as_dict)

        proto = ri1.to_proto()
        ri2 = RunInfo.from_proto(proto)
        self._check(ri2, run_uuid, experiment_id, name, source_type, source_name, entry_point_name,
                    user_id, status, start_time, end_time, source_version, tags, artifact_uri)
        ri3 = RunInfo.from_dictionary(as_dict)
        self._check(ri3, run_uuid, experiment_id, name, source_type, source_name, entry_point_name,
                    user_id, status, start_time, end_time, source_version, tags, artifact_uri)
