import unittest
import uuid

from mlflow.entities import RunInfo, LifecycleStage
from tests.helper_functions import random_str, random_int


class TestRunInfo(object):
    @staticmethod
    def _check(ri, run_uuid, experiment_id, name, source_type, source_name,
               entry_point_name, user_id, status, start_time, end_time, source_version,
               lifecycle_stage, artifact_uri):
        assert type(ri) == RunInfo
        assert ri.run_uuid == run_uuid
        assert ri.experiment_id == experiment_id
        assert ri.name == name
        assert ri.source_type == source_type
        assert ri.source_name == source_name
        assert ri.entry_point_name == entry_point_name
        assert ri.user_id == user_id
        assert ri.status == status
        assert ri.start_time == start_time
        assert ri.end_time == end_time
        assert ri.source_version == source_version
        assert ri.lifecycle_stage == lifecycle_stage
        assert ri.artifact_uri == artifact_uri

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
        lifecycle_stage = LifecycleStage.ACTIVE
        artifact_uri = random_str(random_int(10, 40))
        ri = RunInfo(run_uuid=run_uuid, experiment_id=experiment_id, name=name,
                     source_type=source_type, source_name=source_name,
                     entry_point_name=entry_point_name, user_id=user_id,
                     status=status, start_time=start_time, end_time=end_time,
                     source_version=source_version, lifecycle_stage=lifecycle_stage,
                     artifact_uri=artifact_uri)
        return (ri, run_uuid, experiment_id, name, source_type, source_name, entry_point_name,
                user_id, status, start_time, end_time, source_version, lifecycle_stage,
                artifact_uri)

def test_creation_and_hydration():
    (ri1, run_uuid, experiment_id, name, source_type, source_name, entry_point_name,
     user_id, status, start_time, end_time, source_version, lifecycle_stage,
     artifact_uri) = _create()
    _check(ri1, run_uuid, experiment_id, name, source_type, source_name, entry_point_name,
                user_id, status, start_time, end_time, source_version, lifecycle_stage,
                artifact_uri)
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
        "lifecycle_stage": lifecycle_stage,
        "artifact_uri": artifact_uri
    }
    assert dict(ri1) == as_dict

    proto = ri1.to_proto()
    ri2 = RunInfo.from_proto(proto)
    _check(ri2, run_uuid, experiment_id, name, source_type, source_name, entry_point_name,
                user_id, status, start_time, end_time, source_version, lifecycle_stage,
                artifact_uri)
    ri3 = RunInfo.from_dictionary(as_dict)
    _check(ri3, run_uuid, experiment_id, name, source_type, source_name, entry_point_name,
                user_id, status, start_time, end_time, source_version, lifecycle_stage,
                artifact_uri)
    # Test that we can add a field to RunInfo and still deserialize it from a dictionary
    dict_copy_0 = as_dict.copy()
    dict_copy_0["my_new_field"] = "new field value"
    ri4 = RunInfo.from_dictionary(dict_copy_0)
    _check(ri4, run_uuid, experiment_id, name, source_type, source_name, entry_point_name,
                user_id, status, start_time, end_time, source_version, lifecycle_stage,
                artifact_uri)
