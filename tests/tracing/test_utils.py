from mlflow.entities import LiveSpan
from mlflow.tracing.utils import deduplicate_span_names_in_place, encode_span_id

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
