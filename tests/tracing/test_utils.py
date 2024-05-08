import re

import pytest

from mlflow.entities import LiveSpan
from mlflow.exceptions import MlflowException
from mlflow.tracing.utils import (
    _parse_fields,
    deduplicate_span_names_in_place,
    encode_span_id,
    maybe_get_evaluation_request_id,
)

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


def test_maybe_get_evaluation_request_id():
    assert maybe_get_evaluation_request_id() is None

    try:
        from mlflow.pyfunc.context import Context, set_prediction_context
    except ImportError:
        pytest.skip("Skipping the rest of tests as mlflow.pyfunc module is not available.")

    with set_prediction_context(Context(request_id="eval", is_evaluate=True)):
        assert maybe_get_evaluation_request_id() == "eval"

    with set_prediction_context(Context(request_id="non_eval", is_evaluate=False)):
        assert maybe_get_evaluation_request_id() is None

    with pytest.raises(MlflowException, match="When prediction request context"):
        with set_prediction_context(Context(request_id=None, is_evaluate=True)):
            maybe_get_evaluation_request_id()


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

    # Test invalid fields
    with pytest.raises(
        MlflowException,
        match=re.escape("Field must be of the form 'span_name.[inputs|outputs]'"),
    ):
        _parse_fields(["span1"])
