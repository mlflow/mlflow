import pytest

from mlflow.entities import LiveSpan
from mlflow.exceptions import MlflowException
from mlflow.tracing.utils import (
    deduplicate_span_names_in_place,
    encode_span_id,
    maybe_get_request_id,
)
from mlflow.tracing.utils.search import _FieldParser, _parse_fields, _ParsedField

from tests.tracing.helper import create_mock_otel_span


def test_deduplicate_span_names():
    span_names = ["red", "red", "blue", "red", "green", "blue"]

    spans = [
        LiveSpan(create_mock_otel_span("trace_id", span_id=i, name=span_name), request_id="tr-123")
        for i, span_name in enumerate(span_names)
    ]
    deduplicate_span_names_in_place(spans)

    assert [span.name for span in spans] == [
        "red_1",
        "red_2",
        "blue_1",
        "red_3",
        "green",
        "blue_2",
    ]
    # Check if the span order is preserved
    assert [span.span_id for span in spans] == [encode_span_id(i) for i in [0, 1, 2, 3, 4, 5]]


def test_maybe_get_request_id():
    assert maybe_get_request_id(is_evaluate=True) is None

    try:
        from mlflow.pyfunc.context import Context, set_prediction_context
    except ImportError:
        pytest.skip("Skipping the rest of tests as mlflow.pyfunc module is not available.")

    with set_prediction_context(Context(request_id="eval", is_evaluate=True)):
        assert maybe_get_request_id(is_evaluate=True) == "eval"

    with set_prediction_context(Context(request_id="non_eval", is_evaluate=False)):
        assert maybe_get_request_id(is_evaluate=True) is None

    with pytest.raises(MlflowException, match="Missing request_id for context"):
        with set_prediction_context(Context(request_id=None, is_evaluate=True)):
            maybe_get_request_id(is_evaluate=True)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        # no dot
        ("span.inputs", _ParsedField("span", "inputs", None)),
        ("span.outputs", _ParsedField("span", "outputs", None)),
        ("`span`.inputs", _ParsedField("span", "inputs", None)),
        ("`span`.outputs", _ParsedField("span", "outputs", None)),
        ("span.inputs.field", _ParsedField("span", "inputs", "field")),
        ("`span`.inputs.field", _ParsedField("span", "inputs", "field")),
        ("span.inputs.`field`", _ParsedField("span", "inputs", "field")),
        ("`span`.inputs.`field`", _ParsedField("span", "inputs", "field")),
        # dot in span name
        ("`span.name`.inputs.field", _ParsedField("span.name", "inputs", "field")),
        ("`span.inputs.name`.inputs.field", _ParsedField("span.inputs.name", "inputs", "field")),
        (
            "`span.outputs.name`.outputs.field",
            _ParsedField("span.outputs.name", "outputs", "field"),
        ),
        # dot in field name
        ("span.inputs.`field.name`", _ParsedField("span", "inputs", "field.name")),
        ("span.inputs.`field.inputs.name`", _ParsedField("span", "inputs", "field.inputs.name")),
        (
            "span.outputs.`field.outputs.name`",
            _ParsedField("span", "outputs", "field.outputs.name"),
        ),
        # dot in both span and field name
        ("`span.name`.inputs.`field.name`", _ParsedField("span.name", "inputs", "field.name")),
        (
            "`span.inputs.name`.inputs.`field.inputs.name`",
            _ParsedField("span.inputs.name", "inputs", "field.inputs.name"),
        ),
        (
            "`span.outputs.name`.outputs.`field.outputs.name`",
            _ParsedField("span.outputs.name", "outputs", "field.outputs.name"),
        ),
    ],
)
def test_field_parser(field, expected):
    assert _FieldParser(field).parse() == expected


@pytest.mark.parametrize(
    ("input_string", "error_message"),
    [
        ("`span.inputs.field", "Expected closing backtick"),
        ("`span`a.inputs.field", "Expected dot after span name"),
        ("span.foo.field", "Invalid field type"),
        ("span.inputs.`field", "Expected closing backtick"),
        ("span.inputs.`field`name", "Unexpected characters after closing backtick"),
    ],
)
def test_field_parser_invalid_value(input_string, error_message):
    with pytest.raises(MlflowException, match=error_message):
        _FieldParser(input_string).parse()


def test_parse_fields():
    fields = ["span1.inputs", "span2.outputs.field1", "span3.outputs"]
    parsed_fields = _parse_fields(fields)

    assert len(parsed_fields) == 3

    assert parsed_fields[0].span_name == "span1"
    assert parsed_fields[0].field_type == "inputs"
    assert parsed_fields[0].field_name is None

    assert parsed_fields[1].span_name == "span2"
    assert parsed_fields[1].field_type == "outputs"
    assert parsed_fields[1].field_name == "field1"

    assert parsed_fields[2].span_name == "span3"
    assert parsed_fields[2].field_type == "outputs"
    assert parsed_fields[2].field_name is None
