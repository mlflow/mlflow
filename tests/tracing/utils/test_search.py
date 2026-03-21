import pytest

from mlflow.exceptions import MlflowException
from mlflow.tracing.utils.search import _FieldParser, _parse_fields, _ParsedField


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
