import json

from mlflow.entities import Experiment, Metric
from mlflow.protos.service_pb2 import Experiment as ProtoExperiment
from mlflow.protos.service_pb2 import Metric as ProtoMetric

from mlflow.utils.proto_json_utils import message_to_json, parse_dict


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


def test_parse_dict_string_as_int():
    in_json = {"experiment_id": 123, "name": "name", "unknown": "field"}
    message = ProtoExperiment()
    parse_dict(in_json, message)
    experiment = Experiment.from_proto(message)
    assert experiment.experiment_id == "123"
    assert experiment.name == 'name'
    assert experiment.artifact_location == ''
