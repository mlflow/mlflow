from copy import deepcopy

import pytest
from google.protobuf.any_pb2 import Any
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct
from google.protobuf.timestamp_pb2 import Timestamp
from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from opentelemetry.proto.trace.v1.trace_pb2 import Span

from mlflow.utils.proto_json_utils import parse_dict


@pytest.mark.parametrize("spelling", ["snake", "camel"])
@pytest.mark.parametrize("text", ["x" * 4095, "x" * 4096, "x" * 100000, "日本語😀" * 3000])
def test_large_strings_preserve_complete_span(spelling, text):
    payload = {
        "trace_id": "AAECAwQFBgcICQoLDA0ODw==",
        "span_id": "AAECAwQFBgc=",
        "name": text,
        "start_time_unix_nano": "1234567890123456789",
        "end_time_unix_nano": "1234567890123456799",
        "attributes": [
            {"key": "text", "value": {"string_value": text}},
            {
                "key": "nested",
                "value": {
                    "array_value": {
                        "values": [
                            {"string_value": text},
                            {
                                "kvlist_value": {
                                    "values": [{"key": "inner", "value": {"string_value": text}}]
                                }
                            },
                            {"int_value": "9223372036854775807"},
                            {"bool_value": True},
                            {"double_value": "NaN"},
                            {"bytes_value": "aGk="},
                        ]
                    }
                },
            },
        ],
        "status": {"code": "STATUS_CODE_ERROR", "message": text},
        "events": [
            {
                "name": text,
                "time_unix_nano": "123",
                "attributes": [
                    {"key": "event", "value": {"string_value": text}},
                ],
            }
        ],
        "links": [
            {
                "trace_id": "AAECAwQFBgcICQoLDA0ODw==",
                "span_id": "AAECAwQFBgc=",
                "attributes": [{"key": "link", "value": {"string_value": text}}],
            }
        ],
        "unknown_field": {"string_value": text},
    }
    if spelling == "camel":

        def camel(value):
            if isinstance(value, list):
                return [camel(v) for v in value]
            if not isinstance(value, dict):
                return value
            return {
                k.split("_")[0] + "".join(w.title() for w in k.split("_")[1:]): camel(v)
                for k, v in value.items()
            }

        payload = camel(payload)
    before = deepcopy(payload)
    expected = Span(name="old", attributes=[{"key": "old", "value": {"int_value": 1}}])
    actual = Span()
    actual.CopyFrom(expected)
    ParseDict(payload, expected, ignore_unknown_fields=True)
    assert parse_dict(payload, actual) is None
    assert actual.SerializeToString(deterministic=True) == expected.SerializeToString(
        deterministic=True
    )
    assert payload == before


@pytest.mark.parametrize(
    "payload",
    [
        {"string_value": "x" * 5000, "int_value": "1"},
        {"stringValue": "x" * 5000, "string_value": "y" * 5000},
        pytest.param({"string_value": "x" * 5000 + "\ud800"}, id="high-surrogate"),
        pytest.param({"string_value": "x" * 5000 + "\udc00"}, id="low-surrogate"),
        pytest.param({"string_value": "x" * 5000 + "\ud800\udc00"}, id="surrogate-pair"),
        {"array_value": {"values": [{"string_value": "x" * 5000}, None]}},
        {"array_value": {"values": [{"string_value": "x" * 5000}, {"int_value": "bad"}]}},
        {"string_value": None},
        {"string_value": 100},
        {"string_value": "x" * 5000, "unknown": True},
        {},
    ],
)
def test_large_string_errors_defaults_and_partial_messages_match(payload):
    expected = AnyValue(int_value=42)
    actual = AnyValue(int_value=42)
    expected_error = None
    try:
        ParseDict(payload, expected, ignore_unknown_fields=True)
    except Exception as exc:
        expected_error = exc
    if expected_error is None:
        parse_dict(payload, actual)
    else:
        with pytest.raises(type(expected_error)) as error:
            parse_dict(payload, actual)
        assert str(error.value) == str(expected_error)
    assert actual == expected


@pytest.mark.parametrize(
    ("message", "payload"),
    [
        (Struct(), {"nested": {"large": "x" * 5000}, "number": 1}),
        (Timestamp(), "2026-01-01T00:00:00Z"),
        (Any(), {"@type": "type.googleapis.com/google.protobuf.StringValue", "value": "x" * 5000}),
        (Span(), {"name": "x" * 5000, "status": None, "attributes": []}),
        (Span(), {"name": "x" * 5000, "startTimeUnixNano": "bad"}),
        (Span(), {"name": "x" * 5000, "startTimeUnixNano": "1", "start_time_unix_nano": "2"}),
    ],
)
def test_special_messages_and_aliases_match(message, payload):
    actual = type(message)()
    expected_error = None
    try:
        ParseDict(payload, message, ignore_unknown_fields=True)
    except Exception as exc:
        expected_error = exc
    if expected_error is None:
        parse_dict(payload, actual)
    else:
        with pytest.raises(type(expected_error)) as error:
            parse_dict(payload, actual)
        assert str(error.value) == str(expected_error)
    assert actual == message


@pytest.mark.parametrize("depth", [30, 120, 600])
def test_nested_message_recursion_matches_parser(depth):
    payload = {"string_value": "x" * 5000}
    for _ in range(depth):
        payload = {"array_value": {"values": [payload]}}
    expected = AnyValue()
    actual = AnyValue()
    expected_error = None
    try:
        ParseDict(payload, expected, ignore_unknown_fields=True)
    except Exception as exc:
        expected_error = exc
    if expected_error is None:
        parse_dict(payload, actual)
    else:
        with pytest.raises(type(expected_error)) as error:
            parse_dict(payload, actual)
        assert str(error.value) == str(expected_error)
    assert actual == expected
