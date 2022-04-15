import base64
import json
import numpy as np
import pandas as pd
import pytest

from mlflow.entities import Experiment, Metric
from mlflow.entities.model_registry import RegisteredModel, ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import Experiment as ProtoExperiment
from mlflow.protos.service_pb2 import Metric as ProtoMetric
from mlflow.types import Schema, TensorSpec, ColSpec
from mlflow.protos.model_registry_pb2 import RegisteredModel as ProtoRegisteredModel
from tests.protos.test_message_pb2 import TestMessage
from google.protobuf.text_format import Parse as ParseTextIntoProto

from mlflow.utils.proto_json_utils import (
    message_to_json,
    parse_dict,
    _stringify_all_experiment_ids,
    parse_tf_serving_input,
    _dataframe_from_json,
)

# Prevent pytest from trying to collect TestMessage as a test class:
TestMessage.__test__ = False


def test_message_to_json():
    json_out = message_to_json(Experiment("123", "name", "arty", "active").to_proto())
    assert json.loads(json_out) == {
        "experiment_id": "123",
        "name": "name",
        "artifact_location": "arty",
        "lifecycle_stage": "active",
    }

    original_proto_message = RegisteredModel(
        name="model_1",
        creation_timestamp=111,
        last_updated_timestamp=222,
        description="Test model",
        latest_versions=[
            ModelVersion(
                name="mv-1",
                version="1",
                creation_timestamp=333,
                last_updated_timestamp=444,
                description="v 1",
                user_id="u1",
                current_stage="Production",
                source="A/B",
                run_id="9245c6ce1e2d475b82af84b0d36b52f4",
                status="READY",
                status_message=None,
            ),
            ModelVersion(
                name="mv-2",
                version="2",
                creation_timestamp=555,
                last_updated_timestamp=666,
                description="v 2",
                user_id="u2",
                current_stage="Staging",
                source="A/C",
                run_id="123",
                status="READY",
                status_message=None,
            ),
        ],
    ).to_proto()
    json_out = message_to_json(original_proto_message)
    json_dict = json.loads(json_out)
    assert json_dict == {
        "name": "model_1",
        "creation_timestamp": 111,
        "last_updated_timestamp": 222,
        "description": "Test model",
        "latest_versions": [
            {
                "name": "mv-1",
                "version": "1",
                "creation_timestamp": 333,
                "last_updated_timestamp": 444,
                "current_stage": "Production",
                "description": "v 1",
                "user_id": "u1",
                "source": "A/B",
                "run_id": "9245c6ce1e2d475b82af84b0d36b52f4",
                "status": "READY",
            },
            {
                "name": "mv-2",
                "version": "2",
                "creation_timestamp": 555,
                "last_updated_timestamp": 666,
                "current_stage": "Staging",
                "description": "v 2",
                "user_id": "u2",
                "source": "A/C",
                "run_id": "123",
                "status": "READY",
            },
        ],
    }
    new_proto_message = ProtoRegisteredModel()
    parse_dict(json_dict, new_proto_message)
    assert original_proto_message == new_proto_message

    test_message = ParseTextIntoProto(
        """
        field_int32: 11
        field_int64: 12
        field_uint32: 13
        field_uint64: 14
        field_sint32: 15
        field_sint64: 16
        field_fixed32: 17
        field_fixed64: 18
        field_sfixed32: 19
        field_sfixed64: 20
        field_bool: true
        field_string: "Im a string"
        field_with_default1: 111
        field_repeated_int64: [1, 2, 3]
        field_enum: ENUM_VALUE1
        field_inner_message {
            field_inner_int64: 101
            field_inner_repeated_int64: [102, 103]
        }
        field_inner_message {
            field_inner_int64: 104
            field_inner_repeated_int64: [105, 106]
        }
        oneof1: 207
        [mlflow.ExtensionMessage.field_extended_int64]: 100
        field_map1: [{key: 51 value: "52"}, {key: 53 value: "54"}]
        field_map2: [{key: "61" value: 62}, {key: "63" value: 64}]
        field_map3: [{key: 561 value: 562}, {key: 563 value: 564}]
        field_map4: [{key: 71
                      value: {field_inner_int64: 72
                              field_inner_repeated_int64: [81, 82]
                              field_inner_string: "str1"}},
                     {key: 73
                      value: {field_inner_int64: 74
                              field_inner_repeated_int64: 83
                              field_inner_string: "str2"}}]
    """,
        TestMessage(),
    )
    json_out = message_to_json(test_message)
    json_dict = json.loads(json_out)
    assert json_dict == {
        "field_int32": 11,
        "field_int64": 12,
        "field_uint32": 13,
        "field_uint64": 14,
        "field_sint32": 15,
        "field_sint64": 16,
        "field_fixed32": 17,
        "field_fixed64": 18,
        "field_sfixed32": 19,
        "field_sfixed64": 20,
        "field_bool": True,
        "field_string": "Im a string",
        "field_with_default1": 111,
        "field_repeated_int64": [1, 2, 3],
        "field_enum": "ENUM_VALUE1",
        "field_inner_message": [
            {"field_inner_int64": 101, "field_inner_repeated_int64": [102, 103]},
            {"field_inner_int64": 104, "field_inner_repeated_int64": [105, 106]},
        ],
        "oneof1": 207,
        # JSON doesn't support non-string keys, so the int keys will be converted to strings.
        "field_map1": {"51": "52", "53": "54"},
        "field_map2": {"63": 64, "61": 62},
        "field_map3": {"561": 562, "563": 564},
        "field_map4": {
            "73": {
                "field_inner_int64": 74,
                "field_inner_repeated_int64": [83],
                "field_inner_string": "str2",
            },
            "71": {
                "field_inner_int64": 72,
                "field_inner_repeated_int64": [81, 82],
                "field_inner_string": "str1",
            },
        },
        "[mlflow.ExtensionMessage.field_extended_int64]": "100",
    }
    new_test_message = TestMessage()
    parse_dict(json_dict, new_test_message)
    assert new_test_message == test_message


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


def test_parse_tf_serving_dictionary():
    def assert_result(result, expected_result):
        assert result.keys() == expected_result.keys()
        for key in result:
            assert (result[key] == expected_result[key]).all()
            assert result[key].dtype == expected_result[key].dtype

    # instances are correctly aggregated to dict of input name -> tensor
    tfserving_input = {
        "instances": [
            {"a": "s1", "b": 1.1, "c": [1, 2, 3]},
            {"a": "s2", "b": 2.2, "c": [4, 5, 6]},
            {"a": "s3", "b": 3.3, "c": [7, 8, 9]},
        ]
    }
    # Without Schema
    result = parse_tf_serving_input(tfserving_input)
    expected_result_no_schema = {
        "a": np.array(["s1", "s2", "s3"]),
        "b": np.array([1.1, 2.2, 3.3]),
        "c": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    }
    assert_result(result, expected_result_no_schema)

    # With schema
    schema = Schema(
        [
            TensorSpec(np.dtype("str"), [-1], "a"),
            TensorSpec(np.dtype("float32"), [-1], "b"),
            TensorSpec(np.dtype("int32"), [-1], "c"),
        ]
    )
    dfSchema = Schema([ColSpec("string", "a"), ColSpec("float", "b"), ColSpec("integer", "c")])
    result = parse_tf_serving_input(tfserving_input, schema)
    expected_result_schema = {
        "a": np.array(["s1", "s2", "s3"], dtype=np.dtype("str")),
        "b": np.array([1.1, 2.2, 3.3], dtype="float32"),
        "c": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="int32"),
    }
    assert_result(result, expected_result_schema)
    # With df Schema
    result = parse_tf_serving_input(tfserving_input, dfSchema)
    assert_result(result, expected_result_schema)

    # input provided as a dict
    tfserving_input = {
        "inputs": {
            "a": ["s1", "s2", "s3"],
            "b": [1.1, 2.2, 3.3],
            "c": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        }
    }
    # Without Schema
    result = parse_tf_serving_input(tfserving_input)
    assert_result(result, expected_result_no_schema)

    # With Schema
    result = parse_tf_serving_input(tfserving_input, schema)
    assert_result(result, expected_result_schema)

    # With df Schema
    result = parse_tf_serving_input(tfserving_input, dfSchema)
    assert_result(result, expected_result_schema)


def test_parse_tf_serving_single_array():
    def assert_result(result, expected_result):
        assert (result == expected_result).all()

    # values for each column are properly converted to a tensor
    arr = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
    ]
    tfserving_instances = {"instances": arr}
    tfserving_inputs = {"inputs": arr}

    # Without schema
    instance_result = parse_tf_serving_input(tfserving_instances)
    assert instance_result.shape == (2, 3, 3)
    assert_result(instance_result, np.array(arr, dtype="int64"))

    input_result = parse_tf_serving_input(tfserving_inputs)
    assert input_result.shape == (2, 3, 3)
    assert_result(input_result, np.array(arr, dtype="int64"))

    # Unnamed schema
    schema = Schema([TensorSpec(np.dtype("float32"), [-1])])
    instance_result = parse_tf_serving_input(tfserving_instances, schema)
    assert_result(instance_result, np.array(arr, dtype="float32"))

    input_result = parse_tf_serving_input(tfserving_inputs, schema)
    assert_result(input_result, np.array(arr, dtype="float32"))

    # named schema
    schema = Schema([TensorSpec(np.dtype("float32"), [-1], "a")])
    instance_result = parse_tf_serving_input(tfserving_instances, schema)
    assert isinstance(instance_result, dict)
    assert len(instance_result.keys()) == 1 and "a" in instance_result
    assert_result(instance_result["a"], np.array(arr, dtype="float32"))

    input_result = parse_tf_serving_input(tfserving_inputs, schema)
    assert isinstance(input_result, dict)
    assert len(input_result.keys()) == 1 and "a" in input_result
    assert_result(input_result["a"], np.array(arr, dtype="float32"))


def test_parse_tf_serving_raises_expected_errors():
    # input is bad if a column value is missing for a row/instance
    tfserving_instances = {
        "instances": [
            {"a": "s1", "b": 1},
            {"a": "s2", "b": 2, "c": [4, 5, 6]},
            {"a": "s3", "b": 3, "c": [7, 8, 9]},
        ]
    }
    with pytest.raises(
        MlflowException, match="The length of values for each input/column name are not the same"
    ):
        parse_tf_serving_input(tfserving_instances)

    tfserving_inputs = {
        "inputs": {"a": ["s1", "s2", "s3"], "b": [1, 2, 3], "c": [[1, 2, 3], [4, 5, 6]]}
    }
    with pytest.raises(
        MlflowException, match="The length of values for each input/column name are not the same"
    ):
        parse_tf_serving_input(tfserving_inputs)

    # cannot specify both instance and inputs
    tfserving_input = {
        "instances": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "inputs": {"a": ["s1", "s2", "s3"], "b": [1, 2, 3], "c": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
    }
    match = (
        'Failed to parse data as TF serving input. One of "instances" and "inputs"'
        " must be specified"
    )
    with pytest.raises(MlflowException, match=match):
        parse_tf_serving_input(tfserving_input)

    # cannot specify signature name
    tfserving_input = {
        "signature_name": "hello",
        "inputs": {"a": ["s1", "s2", "s3"], "b": [1, 2, 3], "c": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
    }
    match = 'Failed to parse data as TF serving input. "signature_name" is currently not supported'
    with pytest.raises(MlflowException, match=match):
        parse_tf_serving_input(tfserving_input)


def test_dataframe_from_json():
    source = pd.DataFrame(
        {
            "boolean": [True, False, True],
            "string": ["a", "b", "c"],
            "float": np.array([1.2, 2.3, 3.4], dtype=np.float32),
            "double": np.array([1.2, 2.3, 3.4], dtype=np.float64),
            "integer": np.array([3, 4, 5], dtype=np.int32),
            "long": np.array([3, 4, 5], dtype=np.int64),
            "binary": [bytes([1, 2, 3]), bytes([4, 5]), bytes([6])],
            "date_string": ["2018-02-03", "1996-03-02", "2021-03-05"],
        },
        columns=[
            "boolean",
            "string",
            "float",
            "double",
            "integer",
            "long",
            "binary",
            "date_string",
        ],
    )

    jsonable_df = pd.DataFrame(source, copy=True)
    jsonable_df["binary"] = jsonable_df["binary"].map(base64.b64encode)
    schema = Schema(
        [
            ColSpec("boolean", "boolean"),
            ColSpec("string", "string"),
            ColSpec("float", "float"),
            ColSpec("double", "double"),
            ColSpec("integer", "integer"),
            ColSpec("long", "long"),
            ColSpec("binary", "binary"),
            ColSpec("string", "date_string"),
        ]
    )
    parsed = _dataframe_from_json(
        jsonable_df.to_json(orient="split"), pandas_orient="split", schema=schema
    )
    assert parsed.equals(source)
    parsed = _dataframe_from_json(
        jsonable_df.to_json(orient="records"), pandas_orient="records", schema=schema
    )
    assert parsed.equals(source)
    # try parsing with tensor schema
    tensor_schema = Schema(
        [
            TensorSpec(np.dtype("bool"), [-1], "boolean"),
            TensorSpec(np.dtype("str"), [-1], "string"),
            TensorSpec(np.dtype("float32"), [-1], "float"),
            TensorSpec(np.dtype("float64"), [-1], "double"),
            TensorSpec(np.dtype("int32"), [-1], "integer"),
            TensorSpec(np.dtype("int64"), [-1], "long"),
            TensorSpec(np.dtype(bytes), [-1], "binary"),
        ]
    )
    parsed = _dataframe_from_json(
        jsonable_df.to_json(orient="split"), pandas_orient="split", schema=tensor_schema
    )

    # NB: tensor schema does not automatically decode base64 encoded bytes.
    assert parsed.equals(jsonable_df)
    parsed = _dataframe_from_json(
        jsonable_df.to_json(orient="records"), pandas_orient="records", schema=tensor_schema
    )

    # NB: tensor schema does not automatically decode base64 encoded bytes.
    assert parsed.equals(jsonable_df)

    # Test parse with TesnorSchema with a single tensor
    tensor_schema = Schema([TensorSpec(np.dtype("float32"), [-1, 3])])
    source = pd.DataFrame(
        {
            "a": np.array([1, 2, 3], dtype=np.float32),
            "b": np.array([4.1, 5.2, 6.3], dtype=np.float32),
            "c": np.array([7, 8, 9], dtype=np.float32),
        },
        columns=["a", "b", "c"],
    )
    assert source.equals(
        _dataframe_from_json(
            source.to_json(orient="split"), pandas_orient="split", schema=tensor_schema
        )
    )
    assert source.equals(
        _dataframe_from_json(
            source.to_json(orient="records"), pandas_orient="records", schema=tensor_schema
        )
    )
