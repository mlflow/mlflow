import json
import numpy as np
import pytest

from mlflow.entities import Experiment, Metric
from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import Experiment as ProtoExperiment
from mlflow.protos.service_pb2 import Metric as ProtoMetric

from mlflow.utils.proto_json_utils import (
    message_to_json,
    parse_dict,
    _stringify_all_experiment_ids,
    parse_tf_serving_input,
)


def test_message_to_json():
    json_out = message_to_json(Experiment("123", "name", "arty", "active").to_proto())
    assert json.loads(json_out) == {
        "experiment_id": "123",
        "name": "name",
        "artifact_location": "arty",
        "lifecycle_stage": "active",
    }


def test_parse_dict():
    in_json = {"experiment_id": "123", "name": "name", "unknown": "field"}
    message = ProtoExperiment()
    parse_dict(in_json, message)
    experiment = Experiment.from_proto(message)
    assert experiment.experiment_id == "123"
    assert experiment.name == "name"
    assert experiment.artifact_location == ""


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
    assert experiment.name == "name"
    assert experiment.artifact_location == ""


def test_back_compat():
    in_json = {
        "experiment_id": 123,
        "name": "name",
        "unknown": "field",
        "experiment_ids": [1, 2, 3, 4, 5],
        "things": {
            "experiment_id": 4,
            "more_things": {"experiment_id": 7, "experiment_ids": [2, 3, 4, 5]},
        },
    }

    _stringify_all_experiment_ids(in_json)
    exp_json = {
        "experiment_id": "123",
        "name": "name",
        "unknown": "field",
        "experiment_ids": ["1", "2", "3", "4", "5"],
        "things": {
            "experiment_id": "4",
            "more_things": {"experiment_id": "7", "experiment_ids": ["2", "3", "4", "5"]},
        },
    }
    assert exp_json == in_json


def test_parse_tf_serving_input():
    # instances are correctly aggregated to dict of input name -> tensor
    tfserving_input = {
        "instances": [
            {"a": "s1", "b": 1, "c": [1, 2, 3]},
            {"a": "s2", "b": 2, "c": [4, 5, 6]},
            {"a": "s3", "b": 3, "c": [7, 8, 9]},
        ]
    }
    result = parse_tf_serving_input(tfserving_input)
    assert (result["a"] == np.array(["s1", "s2", "s3"])).all()
    assert (result["b"] == np.array([1, 2, 3])).all()
    assert (result["c"] == np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])).all()

    # input is bad if a column value is missing for a row/instance
    tfserving_input = {
        "instances": [
            {"a": "s1", "b": 1},
            {"a": "s2", "b": 2, "c": [4, 5, 6]},
            {"a": "s3", "b": 3, "c": [7, 8, 9]},
        ]
    }
    with pytest.raises(
        MlflowException, match="The length of values for each input/column name are not the same"
    ):
        parse_tf_serving_input(tfserving_input)

    # values for each column are properly converted to a tensor
    arr = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
    ]
    tfserving_input = {"instances": arr}
    result = parse_tf_serving_input(tfserving_input)
    assert result.shape == (2, 3, 3)
    assert (result == np.array(arr)).all()

    # input data specified via "inputs" must be a dictionary
    tfserving_input = {"inputs": arr}
    with pytest.raises(MlflowException) as ex:
        parse_tf_serving_input(tfserving_input)
    assert "Failed to parse data as TF serving input." in str(ex)

    # input can be provided in column format
    tfserving_input = {
        "inputs": {"a": ["s1", "s2", "s3"], "b": [1, 2, 3], "c": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
    }
    result = parse_tf_serving_input(tfserving_input)
    assert (result["a"] == np.array(["s1", "s2", "s3"])).all()
    assert (result["b"] == np.array([1, 2, 3])).all()
    assert (result["c"] == np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])).all()

    # cannot specify both instance and inputs
    tfserving_input = {
        "instances": arr,
        "inputs": {"a": ["s1", "s2", "s3"], "b": [1, 2, 3], "c": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
    }
    with pytest.raises(MlflowException) as ex:
        parse_tf_serving_input(tfserving_input)
    assert (
        'Failed to parse data as TF serving input. One of "instances" and "inputs"'
        " must be specified" in str(ex)
    )

    # cannot specify signature name
    tfserving_input = {
        "signature_name": "hello",
        "inputs": {"a": ["s1", "s2", "s3"], "b": [1, 2, 3], "c": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
    }
    with pytest.raises(MlflowException) as ex:
        parse_tf_serving_input(tfserving_input)
    assert (
        'Failed to parse data as TF serving input. "signature_name" is currently not supported'
        in str(ex)
    )
