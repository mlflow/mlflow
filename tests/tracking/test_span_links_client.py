from mlflow.entities import Link
from mlflow.tracking.client import MlflowClient

from tests.tracing.helper import get_traces


def test_client_start_trace_with_links():
    client = MlflowClient()
    links = [
        Link(trace_id="tr-0123456789abcdef0123456789abcdef", span_id="0123456789abcdef"),
    ]

    root = client.start_trace("my_trace", links=links)
    client.end_trace(root.trace_id)

    traces = get_traces()
    assert len(traces) == 1
    assert len(traces[0].data.spans[0].links) == 1
    assert traces[0].data.spans[0].links[0].span_id == "0123456789abcdef"


def test_client_start_span_with_links():
    client = MlflowClient()

    root = client.start_trace("my_trace")

    links = [
        Link(
            trace_id="tr-0123456789abcdef0123456789abcdef",
            span_id="0123456789abcdef",
            attributes={"reason": "depends_on"},
        ),
    ]
    child = client.start_span(
        "child_span",
        trace_id=root.trace_id,
        parent_id=root.span_id,
        links=links,
    )
    client.end_span(trace_id=root.trace_id, span_id=child.span_id)
    client.end_trace(root.trace_id)

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    child_span = next(s for s in spans if s.name == "child_span")
    assert len(child_span.links) == 1
    assert child_span.links[0].attributes == {"reason": "depends_on"}
