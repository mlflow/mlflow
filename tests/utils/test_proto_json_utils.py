import base64
import datetime
import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from google.protobuf import descriptor_pb2, descriptor_pool
from google.protobuf.any_pb2 import Any as ProtoAny
from google.protobuf.descriptor_pb2 import MethodOptions
from google.protobuf.json_format import MessageToDict
from google.protobuf.text_format import Parse as ParseTextIntoProto
from google.protobuf.wrappers_pb2 import Int64Value

from mlflow.entities import Experiment, Metric
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import rpc
from mlflow.protos.model_registry_pb2 import RegisteredModel as ProtoRegisteredModel
from mlflow.protos.service_pb2 import Experiment as ProtoExperiment
from mlflow.protos.service_pb2 import GetTrace
from mlflow.protos.service_pb2 import Metric as ProtoMetric
from mlflow.types import ColSpec, DataType, Schema, TensorSpec
from mlflow.types.schema import Array, Map, Object, Property
from mlflow.types.utils import _infer_schema
from mlflow.utils import proto_json_utils
from mlflow.utils.proto_json_utils import (
    MlflowFailedTypeConversion,
    _CustomJsonEncoder,
    cast_df_types_according_to_schema,
    dataframe_from_parsed_json,
    dataframe_from_raw_json,
    dump_input_data,
    message_to_json,
    parse_dict,
    parse_tf_serving_input,
)

from tests.protos.test_message_pb2 import SampleMessage


@pytest.fixture(params=[True, False])
def use_int64_number_json_printer(request, monkeypatch):
    monkeypatch.setattr(proto_json_utils, "_USE_INT64_NUMBER_JSON_PRINTER", request.param)
    yield
    assert proto_json_utils._USE_INT64_NUMBER_JSON_PRINTER is request.param


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"x": np.array([1, 2])}, {"inputs": {"x": [1, 2]}}),
        (
            {"ab": np.array([[1, 2], [3, 4]]), "column": np.array([5.0])},
            {"inputs": {"ab": [[1, 2], [3, 4]], "column": [5.0]}},
        ),
    ],
)
def test_dump_input_data_dict(data, expected):
    assert json.loads(dump_input_data(data)) == expected


def test_dump_input_data_dict_with_custom_key_and_params():
    data = {"x": np.array([1, 2])}
    assert json.loads(
        dump_input_data(data, inputs_key="instances", params={"temperature": 0.5})
    ) == {
        "instances": {"x": [1, 2]},
        "params": {"temperature": 0.5},
    }


def test_dump_input_data_dict_rejects_unsupported_value():
    with pytest.raises(MlflowException, match="Incompatible input type:.* for input x"):
        dump_input_data({"x": [1, 2]})


def test_message_to_json(use_int64_number_json_printer):
    json_out = message_to_json(Experiment("123", "name", "arty", "active").to_proto())
    assert json.loads(json_out) == {
        "experiment_id": "123",
        "name": "name",
        "artifact_location": "arty",
        "lifecycle_stage": "active",
        "workspace": "default",
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

    compact_json = message_to_json(original_proto_message, pretty=False)
    assert "\n" not in compact_json
    assert json.loads(compact_json) == json_dict

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
        SampleMessage(),
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
    json_with_int64_strings = json.loads(
        message_to_json(test_message, convert_int64_to_number=False)
    )
    assert json_with_int64_strings["field_int64"] == "12"
    assert json_with_int64_strings["field_inner_message"][0]["field_inner_int64"] == "101"
    new_test_message = SampleMessage()
    parse_dict(json_dict, new_test_message)
    assert new_test_message == test_message


def test_message_to_json_preserves_special_protobuf_types(use_int64_number_json_printer):
    assert json.loads(message_to_json(Int64Value(value=42))) == "42"
    any_message = ProtoAny()
    any_message.Pack(SampleMessage(field_int64=42))
    assert json.loads(message_to_json(any_message)) == {
        "@type": "type.googleapis.com/mlflow.SampleMessage",
        "field_int64": "42",
    }


def test_message_to_json_preserves_int64_in_message_extension(use_int64_number_json_printer):
    message = MethodOptions()
    message.Extensions[rpc].rate_limit.max_burst = 42
    assert json.loads(message_to_json(message)) == {
        "[mlflow.rpc]": {"rate_limit": {"max_burst": "42"}}
    }


@pytest.fixture(scope="module")
def repeated_int64_extension():
    proto = descriptor_pb2.FileDescriptorProto()
    proto.name = "test_repeated_int64_extension.proto"
    proto.package = "mlflow"
    proto.syntax = "proto2"
    proto.dependency.append("test_message.proto")
    field = proto.extension.add()
    field.name = "field_extended_repeated_int64"
    field.extendee = ".mlflow.SampleMessage"
    field.number = 1002
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT64
    descriptor = descriptor_pool.Default().AddSerializedFile(proto.SerializeToString())
    return descriptor.extensions_by_name[field.name]


@pytest.mark.parametrize("packed", [False, True])
def test_message_to_json_preserves_repeated_int64_extension(
    use_int64_number_json_printer, repeated_int64_extension, packed
):
    message = SampleMessage()
    message.Extensions[repeated_int64_extension].extend([42, 2**53 + 1])
    if packed:
        any_message = ProtoAny()
        any_message.Pack(message)
        message = any_message
    expected = MessageToDict(message, preserving_proto_field_name=True)
    # The original merge converts unbracketed repeated extension keys to numbers,
    # but leaves packed Any messages and bracketed extension keys unchanged.
    if not packed and repeated_int64_extension.name in expected:
        expected[repeated_int64_extension.name] = [42, 2**53 + 1]
    assert json.loads(message_to_json(message)) == expected


@pytest.mark.parametrize("value", [-(2**63), -(2**53 + 1), 2**53 + 1, 2**53 + 2, 2**63 - 1])
def test_message_to_json_preserves_int64_numbers_at_precision_boundary(
    value, use_int64_number_json_printer
):
    assert json.loads(message_to_json(SampleMessage(field_int64=value))) == {"field_int64": value}


def test_message_to_json_complete_trace_int64_numbers(use_int64_number_json_printer):
    response = GetTrace.Response()
    span = response.trace.spans.add()
    span.start_time_unix_nano = 1234567890123456789
    span.end_time_unix_nano = 1234567890123456799
    span.events.add().time_unix_nano = 1234567890123456790
    attribute = span.attributes.add()
    attribute.key = "count"
    attribute.value.int_value = 42

    assert json.loads(message_to_json(response, pretty=False)) == {
        "trace": {
            "spans": [
                {
                    "start_time_unix_nano": 1234567890123456789,
                    "end_time_unix_nano": 1234567890123456799,
                    "events": [{"time_unix_nano": 1234567890123456790}],
                    "attributes": [{"key": "count", "value": {"int_value": 42}}],
                }
            ]
        }
    }


@pytest.mark.parametrize(
    "method",
    ["__init__", "_MessageToJsonObject", "_RegularMessageToJsonObject", "_FieldToJsonObject"],
)
@pytest.mark.parametrize("error", [AttributeError("API removed"), TypeError("API changed")])
def test_message_to_json_disables_optimized_printer_after_failure(
    method, error, monkeypatch, caplog
):
    monkeypatch.setattr(proto_json_utils, "_USE_INT64_NUMBER_JSON_PRINTER", True)
    message = SampleMessage(field_int64=2**53 + 1)
    with mock.patch.object(
        proto_json_utils._Int64NumberJsonPrinter, method, side_effect=error
    ) as printer:
        assert json.loads(message_to_json(message)) == {"field_int64": 2**53 + 1}
        assert not proto_json_utils._USE_INT64_NUMBER_JSON_PRINTER
        assert json.loads(message_to_json(message, pretty=False)) == {"field_int64": 2**53 + 1}
        assert json.loads(message_to_json(message, convert_int64_to_number=False)) == {
            "field_int64": str(2**53 + 1)
        }
        printer.assert_called_once()

    records = [r for r in caplog.records if r.name == proto_json_utils.__name__]
    assert len(records) == 1
    assert records[0].levelname == "ERROR"
    assert "Switching to public protobuf APIs for subsequent calls." in records[0].message
    assert records[0].exc_info[1] is error


def test_message_to_json_propagates_public_serialization_failure(monkeypatch):
    monkeypatch.setattr(proto_json_utils, "_USE_INT64_NUMBER_JSON_PRINTER", False)
    with mock.patch.object(
        proto_json_utils, "MessageToDict", side_effect=ValueError("Invalid message")
    ) as public_printer:
        with pytest.raises(ValueError, match="Invalid message"):
            message_to_json(SampleMessage())
        public_printer.assert_called_once()


def test_message_to_json_optimized_printer_skips_fallback(monkeypatch):
    monkeypatch.setattr(proto_json_utils, "_USE_INT64_NUMBER_JSON_PRINTER", True)
    with mock.patch.object(proto_json_utils, "MessageToDict") as public_printer:
        assert json.loads(message_to_json(SampleMessage(field_int64=42))) == {"field_int64": 42}
        assert proto_json_utils._USE_INT64_NUMBER_JSON_PRINTER
        public_printer.assert_not_called()


def test_message_to_json_int64_strings_skips_optimized_printer(monkeypatch):
    monkeypatch.setattr(proto_json_utils, "_USE_INT64_NUMBER_JSON_PRINTER", True)
    with mock.patch.object(proto_json_utils, "_Int64NumberJsonPrinter") as printer:
        assert json.loads(
            message_to_json(SampleMessage(field_int64=42), convert_int64_to_number=False)
        ) == {"field_int64": "42"}
        assert proto_json_utils._USE_INT64_NUMBER_JSON_PRINTER
        printer.assert_not_called()


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


def assert_result(result, expected_result):
    assert result.keys() == expected_result.keys()
    for key in result:
        assert (result[key] == expected_result[key]).all()
        assert result[key].dtype == expected_result[key].dtype


def test_parse_tf_serving_dictionary():
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
    schema = Schema([
        TensorSpec(np.dtype("str"), [-1], "a"),
        TensorSpec(np.dtype("float32"), [-1], "b"),
        TensorSpec(np.dtype("int32"), [-1], "c"),
    ])
    df_schema = Schema([ColSpec("string", "a"), ColSpec("float", "b"), ColSpec("integer", "c")])
    result = parse_tf_serving_input(tfserving_input, schema)
    expected_result_schema = {
        "a": np.array(["s1", "s2", "s3"], dtype=np.dtype("str")),
        "b": np.array([1.1, 2.2, 3.3], dtype="float32"),
        "c": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="int32"),
    }
    assert_result(result, expected_result_schema)
    # With df Schema
    result = parse_tf_serving_input(tfserving_input, df_schema)
    assert_result(result, expected_result_schema)
    # With df Schema containing array
    new_schema = _infer_schema(tfserving_input["instances"])
    result = parse_tf_serving_input(tfserving_input, new_schema)
    expected_result = {
        "a": np.array(["s1", "s2", "s3"]),
        "b": np.array([1.1, 2.2, 3.3], dtype="float64"),
        "c": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="int64"),
    }
    assert_result(result, expected_result)

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
    result = parse_tf_serving_input(tfserving_input, df_schema)
    assert_result(result, expected_result_schema)


def test_parse_tf_serving_arbitrary_input_dictionary():
    # input provided as a columnar dict with an arbitrary shape for each input, specifically a
    # different 0th dimension.
    tfserving_input_arbitrary = {
        "inputs": {
            "a": [["s1", "s2", "s3"], ["s4", "s5", "s6"]],  # [2, 3]
            "b": [1.1, 2.2, 3.3],  # [3,  ]
            "c": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],  # [4, 3]
        }
    }

    schema = Schema([
        TensorSpec(np.dtype("str"), [-1, 3], "a"),
        TensorSpec(np.dtype("float32"), [-1], "b"),
        TensorSpec(np.dtype("int32"), [-1, 4], "c"),
    ])
    df_schema = Schema([ColSpec("string", "a"), ColSpec("float", "b"), ColSpec("integer", "c")])

    expected_result_no_schema_arbitrary = {
        "a": np.array([["s1", "s2", "s3"], ["s4", "s5", "s6"]]),
        "b": np.array([1.1, 2.2, 3.3]),
        "c": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]),
    }
    expected_result_schema_arbitrary = {
        "a": np.array([["s1", "s2", "s3"], ["s4", "s5", "s6"]], dtype=np.dtype("str")),
        "b": np.array([1.1, 2.2, 3.3], dtype="float32"),
        "c": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype="int32"),
    }

    # Without Schema
    result = parse_tf_serving_input(tfserving_input_arbitrary)
    assert_result(result, expected_result_no_schema_arbitrary)

    # With Schema
    result = parse_tf_serving_input(tfserving_input_arbitrary, schema)
    assert_result(result, expected_result_schema_arbitrary)

    # With df Schema
    result = parse_tf_serving_input(tfserving_input_arbitrary, df_schema)
    assert_result(result, expected_result_schema_arbitrary)


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
    assert len(instance_result.keys()) == 1
    assert "a" in instance_result
    assert_result(instance_result["a"], np.array(arr, dtype="float32"))

    input_result = parse_tf_serving_input(tfserving_inputs, schema)
    assert isinstance(input_result, dict)
    assert len(input_result.keys()) == 1
    assert "a" in input_result
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

    # cannot specify both instance and inputs
    tfserving_input = {
        "instances": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "inputs": {"a": ["s1", "s2", "s3"], "b": [1, 2, 3], "c": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
    }
    match = 'Invalid input. One of "instances" and "inputs" must be specified'
    with pytest.raises(MlflowException, match=match):
        parse_tf_serving_input(tfserving_input)

    # cannot specify signature name
    tfserving_input = {
        "signature_name": "hello",
        "inputs": {"a": ["s1", "s2", "s3"], "b": [1, 2, 3], "c": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
    }
    match = '"signature_name" parameter is currently not supported'
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
    schema = Schema([
        ColSpec("boolean", "boolean"),
        ColSpec("string", "string"),
        ColSpec("float", "float"),
        ColSpec("double", "double"),
        ColSpec("integer", "integer"),
        ColSpec("long", "long"),
        ColSpec("binary", "binary"),
        ColSpec("string", "date_string"),
    ])
    parsed = dataframe_from_raw_json(
        jsonable_df.to_json(orient="split"), pandas_orient="split", schema=schema
    )
    pd.testing.assert_frame_equal(parsed, source)
    parsed = dataframe_from_raw_json(
        jsonable_df.to_json(orient="records"), pandas_orient="records", schema=schema
    )
    pd.testing.assert_frame_equal(parsed, source)
    # try parsing with tensor schema
    tensor_schema = Schema([
        TensorSpec(np.dtype("bool"), [-1], "boolean"),
        TensorSpec(np.dtype("str"), [-1], "string"),
        TensorSpec(np.dtype("float32"), [-1], "float"),
        TensorSpec(np.dtype("float64"), [-1], "double"),
        TensorSpec(np.dtype("int32"), [-1], "integer"),
        TensorSpec(np.dtype("int64"), [-1], "long"),
        TensorSpec(np.dtype(bytes), [-1], "binary"),
    ])
    parsed = dataframe_from_raw_json(
        jsonable_df.to_json(orient="split"), pandas_orient="split", schema=tensor_schema
    )

    # NB: tensor schema does not automatically decode base64 encoded bytes.
    pd.testing.assert_frame_equal(parsed, jsonable_df)
    parsed = dataframe_from_raw_json(
        jsonable_df.to_json(orient="records"), pandas_orient="records", schema=tensor_schema
    )

    # NB: tensor schema does not automatically decode base64 encoded bytes.
    pd.testing.assert_frame_equal(parsed, jsonable_df)

    # Test parse with TensorSchema with a single tensor
    tensor_schema = Schema([TensorSpec(np.dtype("float32"), [-1, 3])])
    source = pd.DataFrame(
        {
            "a": np.array([1, 2, 3], dtype=np.float32),
            "b": np.array([4.1, 5.2, 6.3], dtype=np.float32),
            "c": np.array([7, 8, 9], dtype=np.float32),
        },
        columns=["a", "b", "c"],
    )
    pd.testing.assert_frame_equal(
        source,
        dataframe_from_raw_json(
            source.to_json(orient="split"), pandas_orient="split", schema=tensor_schema
        ),
    )
    pd.testing.assert_frame_equal(
        source,
        dataframe_from_raw_json(
            source.to_json(orient="records"), pandas_orient="records", schema=tensor_schema
        ),
    )

    schema = Schema([ColSpec("datetime", "datetime")])
    parsed = dataframe_from_raw_json(
        """
[
    {"datetime": "2022-01-01T00:00:00"},
    {"datetime": "2022-01-02T03:04:05"}
]
    """,
        pandas_orient="records",
        schema=schema,
    )
    expected = pd.DataFrame(
        {
            "datetime": np.array(
                ["2022-01-01T00:00:00", "2022-01-02T03:04:05"],
                dtype=DataType.datetime.to_pandas(),
            )
        },
    )
    pd.testing.assert_frame_equal(parsed, expected)


@pytest.mark.parametrize(
    "value",
    ["2022-01-01T00:00:00", "2022-01-01", "2022-01-01T00:00:00+02:00"],
)
def test_dataframe_from_json_parses_datetime_as_nanoseconds(value):
    parsed = dataframe_from_raw_json(
        json.dumps([{"datetime": value}]),
        pandas_orient="records",
        schema=Schema([ColSpec("datetime", "datetime")]),
    )
    assert getattr(parsed["datetime"].dt, "unit", "ns") == "ns"


@pytest.mark.parametrize(
    ("dt", "expected"),
    [
        (datetime.datetime(2022, 1, 1), '"2022-01-01T00:00:00"'),
        (datetime.datetime(2022, 1, 2, 3, 4, 5), '"2022-01-02T03:04:05"'),
        (datetime.date(2022, 1, 1), '"2022-01-01"'),
        (datetime.time(0, 0, 0), '"00:00:00"'),
        (pd.Timestamp(2022, 1, 1), '"2022-01-01T00:00:00"'),
    ],
)
def test_datetime_encoder(dt, expected):
    assert json.dumps(dt, cls=_CustomJsonEncoder) == expected


@pytest.mark.parametrize(
    ("dataframe", "schema", "expected"),
    [
        (
            pd.DataFrame(columns=["foo"], data=[1, 2, 3]),
            Schema([TensorSpec(np.dtype("float64"), [-1], "foo")]),
            np.dtype("float64"),
        ),
        (
            pd.DataFrame(columns=["foo"], data=[[[1, 2, 3]], [[4, 5, 6]]]),
            Schema([TensorSpec(np.dtype("float64"), [-1, 1], "foo")]),
            np.dtype("object"),
        ),
        (
            pd.DataFrame(index=[1, 2, 3], columns=["foo"], data=[1, 2, 3]),
            Schema([TensorSpec(np.dtype("float64"), [-1], "foo")]),
            np.dtype("float64"),
        ),
        (
            pd.DataFrame(columns=["foo"], data=[1, 2, 3]),
            Schema([ColSpec("double", "foo")]),
            np.dtype("float64"),
        ),
    ],
)
def test_cast_df_types_according_to_schema_success(dataframe, schema, expected):
    casted_pdf = cast_df_types_according_to_schema(dataframe, schema)
    assert casted_pdf["foo"].dtype == expected


@pytest.mark.parametrize(
    ("dataframe", "schema", "error_message"),
    [
        (
            pd.DataFrame(columns=["foo"], data=[1, 2, 3]),
            Schema([ColSpec("binary", "foo")]),
            r"TypeError\('encoding without a string argument'\)",
        ),
        (
            pd.DataFrame(columns=["foo"], data=["a", "b", "c"]),
            Schema([ColSpec("double", "foo")]),
            r'ValueError\("could not convert string to float: \'a\'"\)',
        ),
    ],
)
def test_cast_df_types_according_to_schema_error_message(dataframe, schema, error_message):
    with pytest.raises(MlflowFailedTypeConversion, match=error_message):
        cast_df_types_according_to_schema(dataframe, schema)


@pytest.mark.parametrize(
    ("pandas_orient", "data"),
    [
        ("split", {"columns": ["foo"], "data": [[True], [None], [False]]}),
        ("records", [{"foo": True}, {"foo": None}, {"foo": False}]),
        ("records", [{"foo": True}, {}, {"foo": False}]),
    ],
)
def test_dataframe_from_parsed_json_keeps_missing_booleans(pandas_orient, data):
    schema = Schema([ColSpec(DataType.boolean, "foo", required=False)])
    df = dataframe_from_parsed_json(data, pandas_orient, schema)
    values = df["foo"].tolist()
    assert values[0] is True
    assert pd.isna(values[1])
    assert values[2] is False


@pytest.mark.parametrize(
    ("data_type", "value", "expected"),
    [
        (DataType.boolean, True, True),
        (DataType.string, "a", "a"),
        (DataType.binary, "a", b"a"),
    ],
)
@pytest.mark.parametrize("key", ["inputs", "instances"])
def test_parse_tf_serving_input_keeps_missing_values(data_type, value, expected, key):
    schema = Schema([ColSpec(data_type, "foo", required=False), ColSpec(DataType.long, "bar")])
    data = (
        {"foo": [value, None], "bar": [1, 2]}
        if key == "inputs"
        else [{"foo": value, "bar": 1}, {"foo": None, "bar": 2}]
    )
    result = parse_tf_serving_input({key: data}, schema)
    assert result["foo"].tolist() == [expected, None]


@pytest.mark.parametrize(
    ("pandas_orient", "data"),
    [
        ("split", {"columns": ["foo"], "data": [[1.5], [None], [0]]}),
        ("records", [{"foo": "true"}, {}, {"foo": False}]),
    ],
)
def test_dataframe_from_parsed_json_converts_non_null_booleans_with_missing_values(
    pandas_orient, data
):
    schema = Schema([ColSpec(DataType.boolean, "foo", required=False)])
    df = dataframe_from_parsed_json(data, pandas_orient, schema)
    values = df["foo"].tolist()
    assert values[0] is True
    assert pd.isna(values[1])
    assert values[2] is False


@pytest.mark.parametrize("data_type", [DataType.boolean, DataType.string, DataType.binary])
def test_parse_tf_serving_input_rejects_nested_values_with_missing_values(data_type):
    schema = Schema([ColSpec(data_type, "foo", required=False), ColSpec(DataType.long, "bar")])
    data = {"foo": [["a"], None], "bar": [1, 2]}
    with pytest.raises(MlflowException, match="setting an array element with a sequence"):
        parse_tf_serving_input({"inputs": data}, schema)


@pytest.mark.parametrize(
    ("data", "schema", "instances_data"),
    [
        ({"query": "sentence"}, Schema([ColSpec(DataType.string, name="query")]), None),
        (
            {"query": ["sentence_1", "sentence_2"]},
            Schema([ColSpec(Array(DataType.string), name="query")]),
            None,
        ),
        (
            {"query": ["sentence_1", "sentence_2"], "table": "some_table"},
            Schema([
                ColSpec(Array(DataType.string), name="query"),
                ColSpec(DataType.string, name="table"),
            ]),
            None,
        ),
        (
            {"query": [{"name": "value", "age": 10}, {"name": "value"}], "table": ["some_table"]},
            Schema([
                ColSpec(
                    Array(
                        Object([
                            Property("name", DataType.string),
                            Property("age", DataType.long, required=False),
                        ])
                    ),
                    name="query",
                ),
                ColSpec(Array(DataType.string), name="table"),
            ]),
            None,
        ),
        (
            [{"query": "sentence"}, {"query": "sentence"}],
            Schema([ColSpec(DataType.string, name="query")]),
            {"query": ["sentence", "sentence"]},
        ),
        (
            [
                {"query": ["sentence_1", "sentence_2"], "table": "some_table"},
                {"query": ["sentence_1", "sentence_2"]},
            ],
            Schema([
                ColSpec(Array(DataType.string), name="query"),
                ColSpec(DataType.string, name="table", required=False),
            ]),
            {
                "query": [["sentence_1", "sentence_2"], ["sentence_1", "sentence_2"]],
                "table": ["some_table"],
            },
        ),
        (
            [
                {"query": {"a": "sentence_1", "b": "sentence_2"}, "table": "some_table"},
                {"query": {"a": "sentence_1"}, "table": "some_table"},
            ],
            Schema([
                ColSpec(
                    Object([
                        Property("a", DataType.string),
                        Property("b", DataType.string, required=False),
                    ]),
                    name="query",
                ),
                ColSpec(DataType.string, name="table"),
            ]),
            {
                "query": [{"a": "sentence_1", "b": "sentence_2"}, {"a": "sentence_1"}],
                "table": ["some_table", "some_table"],
            },
        ),
        (
            {
                "query": [{"name": "value", "age": "10"}, {"name": "value"}],
                "table": {"k": "some_table"},
                "data": {"k1": ["a", "b"], "k2": ["c"]},
            },
            Schema([
                ColSpec(
                    Array(Map(value_type=DataType.string)),
                    name="query",
                ),
                ColSpec(Map(value_type=DataType.string), name="table"),
                ColSpec(Map(value_type=Array(DataType.string)), name="data"),
            ]),
            None,
        ),
    ],
)
def test_parse_tf_serving_input_for_dictionaries_and_lists_and_maps(data, schema, instances_data):
    np.testing.assert_equal(parse_tf_serving_input({"inputs": data}, schema), data)
    if instances_data is None:
        np.testing.assert_equal(parse_tf_serving_input({"instances": data}, schema), data)
    else:
        np.testing.assert_equal(parse_tf_serving_input({"instances": data}, schema), instances_data)
    df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
    df_split = df.to_dict(orient="split")
    pd.testing.assert_frame_equal(dataframe_from_parsed_json(df_split, "split", schema), df)
    df_records = df.to_dict(orient="records")
    pd.testing.assert_frame_equal(dataframe_from_parsed_json(df_records, "records", schema), df)
