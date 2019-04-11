import json
import pytest

from mlflow.entities import Experiment, Metric
from mlflow.protos.service_pb2 import Experiment as ProtoExperiment
from mlflow.protos.service_pb2 import Metric as ProtoMetric

from mlflow.utils.proto_json_utils import message_to_json, parse_dict, backcompat_helper


def test_message_to_json():
    json_out = message_to_json(Experiment("123", "name", "arty", 'active').to_proto())
    assert json.loads(json_out) == {
        "experiment_id": "123",
        "name": "name",
        "artifact_location": "arty",
        "lifecycle_stage": 'active',
    }


def test_parse_dict():
    in_json = {"experiment_id": "123", "name": "name", "unknown": "field"}
    message = ProtoExperiment()
    parse_dict(in_json, message)
    experiment = Experiment.from_proto(message)
    assert experiment.experiment_id == "123"
    assert experiment.name == 'name'
    assert experiment.artifact_location == ''


def test_parse_dict_int_as_string_backcompat():
    in_json = {"timestamp": "123"}
    message = ProtoMetric()
    parse_dict(in_json, message)
    experiment = Metric.from_proto(message)
    assert experiment.timestamp == 123


def test_parse_legacy_experiment():
    in_json = {"experiment_id": 123, "name": "name", "unknown": "field"}
    message = ProtoExperiment()
    parse_dict(in_json, message)
    experiment = Experiment.from_proto(message)
    assert experiment.experiment_id == "123"
    assert experiment.name == 'name'
    assert experiment.artifact_location == ''


def test_back_compat():
    in_json = {"experiment_id": 123,
               "name": "name",
               "unknown": "field",
               "experiment_ids": [1, 2, 3, 4, 5],
               "things": {"experiment_id": 4,
                          "more_things": {"experiment_id": 7, "experiment_ids": [2, 3, 4, 5]}}}
    verify_string_experiment_ids(in_json, int)
    backcompat_helper(in_json)
    verify_string_experiment_ids(in_json, str)


def test_verify_experiment_id_type():
    in_json = {"experiment_id": 123,
               "name": "name",
               "unknown": "field",
               "experiment_ids": [1, 2, 3, 4, 5],
               "things": {"experiment_id": 4,
                          "more_things": {"experiment_id": 7, "experiment_ids": ["2", 3, 4, 5]}}}
    with pytest.raises(AssertionError):
        verify_string_experiment_ids(in_json, int)
    with pytest.raises(AssertionError):
        verify_string_experiment_ids(in_json, str)

    valid_int_json = {"experiment_id": 123,
                      "name": "name",
                      "unknown": "field",
                      "experiment_ids": [1, 2, 3, 4, 5],
                      "things": {"experiment_id": 4,
                                 "more_things": {"experiment_id": 7,
                                                 "experiment_ids": [2, 3, 4, 5]}}}
    with pytest.raises(AssertionError):
        verify_string_experiment_ids(valid_int_json, str)
    verify_string_experiment_ids(valid_int_json, int)


def verify_string_experiment_ids(js_dict, expected_type):
    for key in js_dict:
        if key == "experiment_id":
            assert type(js_dict[key]) == expected_type
        elif key == "experiment_ids":
            for val in js_dict[key]:
                assert type(val) == expected_type
        elif isinstance(js_dict[key], dict):
            verify_string_experiment_ids(js_dict[key], expected_type)
        elif isinstance(js_dict[key], list):
            for val in js_dict[key]:
                if isinstance(val, dict):
                    verify_string_experiment_ids(js_dict[key], expected_type)
