import mlflow
from mlflow.entities import Link
from mlflow.entities.span import NoOpSpan
from mlflow.tracing.fluent import start_span_no_context

from tests.tracing.helper import get_traces


# --- Integration tests for start_span with links ---


def test_start_span_with_links():
    links = [
        Link(trace_id="tr-0123456789abcdef0123456789abcdef", span_id="0123456789abcdef"),
        Link(
            trace_id="tr-abcdef0123456789abcdef0123456789",
            span_id="abcdef0123456789",
            attributes={"kind": "follows_from"},
        ),
    ]

    with mlflow.start_span(name="linked_span", links=links) as span:
        span.set_outputs("done")

    traces = get_traces()
    assert len(traces) == 1

    linked_span = traces[0].data.spans[0]
    assert len(linked_span.links) == 2
    assert linked_span.links[0].trace_id == "tr-0123456789abcdef0123456789abcdef"
    assert linked_span.links[0].span_id == "0123456789abcdef"
    assert linked_span.links[1].attributes == {"kind": "follows_from"}


def test_start_span_no_context_with_links():
    links = [
        Link(trace_id="tr-0123456789abcdef0123456789abcdef", span_id="0123456789abcdef"),
    ]

    root = start_span_no_context("root_span", links=links)
    root.end()

    traces = get_traces()
    assert len(traces) == 1
    assert len(traces[0].data.spans[0].links) == 1
    assert traces[0].data.spans[0].links[0].span_id == "0123456789abcdef"


def test_start_span_without_links():
    with mlflow.start_span(name="no_links_span") as span:
        span.set_outputs("done")

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].data.spans[0].links == []


# --- Decorator tests ---


def test_trace_decorator_with_static_links():
    links = [
        Link(trace_id="tr-0123456789abcdef0123456789abcdef", span_id="0123456789abcdef"),
    ]

    @mlflow.trace(links=links)
    def my_func(x):
        return x * 2

    my_func(5)

    traces = get_traces()
    assert len(traces) == 1
    assert len(traces[0].data.spans[0].links) == 1
    assert traces[0].data.spans[0].links[0].span_id == "0123456789abcdef"


def test_trace_decorator_with_callable_links():
    call_count = 0

    def get_links():
        nonlocal call_count
        call_count += 1
        return [
            Link(
                trace_id="tr-0123456789abcdef0123456789abcdef",
                span_id="0123456789abcdef",
                attributes={"call": call_count},
            ),
        ]

    @mlflow.trace(links=get_links)
    def my_func(x):
        return x * 2

    my_func(5)
    my_func(10)

    traces = get_traces()
    assert len(traces) == 2
    assert call_count == 2
    # Each invocation should have gotten fresh links from the callable
    for trace_obj in traces:
        assert len(trace_obj.data.spans[0].links) == 1


def test_trace_decorator_with_generator_and_links():
    links = [
        Link(trace_id="tr-0123456789abcdef0123456789abcdef", span_id="0123456789abcdef"),
    ]

    @mlflow.trace(links=links)
    def my_generator():
        yield 1
        yield 2
        yield 3

    result = list(my_generator())
    assert result == [1, 2, 3]

    traces = get_traces()
    assert len(traces) == 1
    assert len(traces[0].data.spans[0].links) == 1


# --- Client API tests ---


def test_client_start_trace_with_links():
    client = mlflow.MlflowClient()
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
    client = mlflow.MlflowClient()

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


# --- NoOpSpan safety ---


def test_noop_span_add_link_does_not_raise():
    noop = NoOpSpan()
    link = Link(trace_id="tr-0123456789abcdef0123456789abcdef", span_id="0123456789abcdef")
    noop.add_link(link)


# --- End-to-end: links persist through store round-trip ---


def test_links_persist_through_store_roundtrip():
    links = [
        Link(
            trace_id="tr-0123456789abcdef0123456789abcdef",
            span_id="0123456789abcdef",
            attributes={"type": "caused_by"},
        ),
        Link(trace_id="tr-fedcba9876543210fedcba9876543210", span_id="fedcba9876543210"),
    ]

    with mlflow.start_span(name="roundtrip_span", links=links) as span:
        span.set_outputs("test")

    traces = get_traces()
    assert len(traces) == 1
    stored_links = traces[0].data.spans[0].links
    assert len(stored_links) == 2
    assert stored_links[0].trace_id == "tr-0123456789abcdef0123456789abcdef"
    assert stored_links[0].span_id == "0123456789abcdef"
    assert stored_links[0].attributes == {"type": "caused_by"}
    assert stored_links[1].trace_id == "tr-fedcba9876543210fedcba9876543210"
    assert stored_links[1].span_id == "fedcba9876543210"
    assert stored_links[1].attributes is None
