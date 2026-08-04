import logging

import pytest

import mlflow
from mlflow.entities import SpanLogLevel
from mlflow.entities.span import Span, SpanType
from mlflow.entities.span_event import SpanEvent
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.utils.default_log_level import default_log_level_for_span_type

from tests.tracing.helper import get_traces


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (SpanLogLevel.DEBUG, SpanLogLevel.DEBUG),
        (SpanLogLevel.CRITICAL, SpanLogLevel.CRITICAL),
        ("DEBUG", SpanLogLevel.DEBUG),
        ("info", SpanLogLevel.INFO),
        ("Warning", SpanLogLevel.WARNING),
        ("  ERROR  ", SpanLogLevel.ERROR),
    ],
)
def test_from_value_accepts_enum_and_string_forms(value, expected):
    assert SpanLogLevel.from_value(value) is expected


@pytest.mark.parametrize("value", ["NOPE", "TRACE", "FATAL", "INFOO", "", "WARN", "warn"])
def test_from_value_rejects_invalid_string(value):
    # "WARN" is not a valid alias; only the full names are accepted.
    with pytest.raises(MlflowException, match="Invalid SpanLogLevel"):
        SpanLogLevel.from_value(value)


@pytest.mark.parametrize("value", [0, 7, 10, 20, 30, 40, 50, 100, -1, logging.INFO])
def test_from_value_rejects_int(value):
    # Raw integers — including the canonical 10/20/30/40/50 and `logging.*` —
    # are rejected: the API surface is `SpanLogLevel | str` only. Use the enum
    # member or its name string instead.
    with pytest.raises(MlflowException, match="must be"):
        SpanLogLevel.from_value(value)


@pytest.mark.parametrize("value", [None, 1.5, ["INFO"], object(), True, False])
def test_from_value_rejects_invalid_type(value):
    with pytest.raises(MlflowException, match="must be"):
        SpanLogLevel.from_value(value)


def test_log_level_constructor_default_for_unknown_span():
    # No span_type provided -> defaults to UNKNOWN -> DEBUG via the constructor.
    with mlflow.start_span("s"):
        pass
    persisted = get_traces()[0].data.spans[0]
    assert persisted.log_level is SpanLogLevel.DEBUG
    assert persisted.attributes[SpanAttributeKey.LOG_LEVEL] == int(SpanLogLevel.DEBUG)


def test_log_level_constructor_default_for_info_span_type():
    # CHAT_MODEL is in the INFO set -> constructor stamps INFO automatically.
    with mlflow.start_span("s", span_type=SpanType.CHAT_MODEL):
        pass
    persisted = get_traces()[0].data.spans[0]
    assert persisted.log_level is SpanLogLevel.INFO


@pytest.mark.parametrize(
    ("set_value", "expected"),
    [
        (SpanLogLevel.WARNING, SpanLogLevel.WARNING),
        (SpanLogLevel.ERROR, SpanLogLevel.ERROR),
        ("info", SpanLogLevel.INFO),
        ("CRITICAL", SpanLogLevel.CRITICAL),
    ],
)
def test_set_log_level_normalizes_input(set_value, expected):
    with mlflow.start_span("s") as span:
        span.set_log_level(set_value)
    persisted = get_traces()[0].data.spans[0]
    assert persisted.log_level is expected
    # Stored as the raw int under the reserved attribute key for portability.
    assert persisted.attributes[SpanAttributeKey.LOG_LEVEL] == int(expected)


def test_set_log_level_rejects_invalid_input():
    with (
        mlflow.start_span("s") as span,
        pytest.raises(MlflowException, match="Invalid SpanLogLevel"),
    ):
        span.set_log_level("NOPE")


def test_start_span_kwarg_overrides_constructor_default():
    # CHAT_MODEL would default to INFO; explicit kwarg should win.
    with mlflow.start_span("s", span_type=SpanType.CHAT_MODEL, log_level="WARNING"):
        pass
    persisted = get_traces()[0].data.spans[0]
    assert persisted.log_level is SpanLogLevel.WARNING


def test_start_span_no_context_kwarg():
    span = mlflow.start_span_no_context("s", log_level=SpanLogLevel.ERROR)
    span.end()
    persisted = get_traces()[0].data.spans[0]
    assert persisted.log_level is SpanLogLevel.ERROR


def test_trace_decorator_kwarg_sync():
    @mlflow.trace(log_level="DEBUG")
    def fn(x):
        return x + 1

    fn(1)
    persisted = get_traces()[0].data.spans[0]
    assert persisted.log_level is SpanLogLevel.DEBUG


@pytest.mark.asyncio
async def test_trace_decorator_kwarg_async():
    @mlflow.trace(log_level=SpanLogLevel.INFO)
    async def fn(x):
        return x + 1

    await fn(1)
    persisted = get_traces()[0].data.spans[0]
    assert persisted.log_level is SpanLogLevel.INFO


def test_trace_decorator_kwarg_generator():
    @mlflow.trace(log_level="WARNING")
    def gen():
        yield 1
        yield 2

    list(gen())
    persisted = get_traces()[0].data.spans[0]
    assert persisted.log_level is SpanLogLevel.WARNING


def test_log_level_round_trips_through_to_dict_from_dict():
    with mlflow.start_span("s", log_level=SpanLogLevel.ERROR):
        pass
    persisted = get_traces()[0].data.spans[0]

    rebuilt = Span.from_dict(persisted.to_dict())

    assert rebuilt.log_level is SpanLogLevel.ERROR


@pytest.mark.parametrize(
    ("span_type", "expected"),
    [
        # INFO set: user-visible semantic operations.
        (SpanType.LLM, SpanLogLevel.INFO),
        (SpanType.CHAT_MODEL, SpanLogLevel.INFO),
        (SpanType.AGENT, SpanLogLevel.INFO),
        (SpanType.TOOL, SpanLogLevel.INFO),
        (SpanType.RETRIEVER, SpanLogLevel.INFO),
        (SpanType.EMBEDDING, SpanLogLevel.INFO),
        # DEBUG set: internal/glue work and unclassified types.
        (SpanType.CHAIN, SpanLogLevel.DEBUG),
        (SpanType.PARSER, SpanLogLevel.DEBUG),
        (SpanType.RERANKER, SpanLogLevel.DEBUG),
        (SpanType.MEMORY, SpanLogLevel.DEBUG),
        (SpanType.WORKFLOW, SpanLogLevel.DEBUG),
        (SpanType.TASK, SpanLogLevel.DEBUG),
        (SpanType.GUARDRAIL, SpanLogLevel.DEBUG),
        (SpanType.EVALUATOR, SpanLogLevel.DEBUG),
        (SpanType.UNKNOWN, SpanLogLevel.DEBUG),
        # Custom (non-built-in) span types fall through to DEBUG.
        ("MY_CUSTOM_TYPE", SpanLogLevel.DEBUG),
        (None, SpanLogLevel.DEBUG),
    ],
)
def test_default_log_level_for_span_type_mapping(span_type, expected):
    assert default_log_level_for_span_type(span_type) is expected


def test_constructor_stamps_default_for_info_span_type():
    span = mlflow.start_span_no_context("s", span_type=SpanType.CHAT_MODEL)
    span.end()
    persisted = get_traces()[0].data.spans[0]
    assert persisted.log_level is SpanLogLevel.INFO


def test_constructor_stamps_default_for_debug_span_type():
    span = mlflow.start_span_no_context("s", span_type=SpanType.PARSER)
    span.end()
    persisted = get_traces()[0].data.spans[0]
    assert persisted.log_level is SpanLogLevel.DEBUG


def test_multi_span_trace_carries_per_span_levels():
    @mlflow.trace(log_level="DEBUG", name="inner")
    def inner():
        return 1

    @mlflow.trace(log_level="WARNING", name="root")
    def root():
        return inner()

    root()
    spans = {s.name: s for s in get_traces()[0].data.spans}
    assert spans["root"].log_level is SpanLogLevel.WARNING
    assert spans["inner"].log_level is SpanLogLevel.DEBUG


# ---- Exception → ERROR bump --------------------------------------------------


def test_exception_event_bumps_debug_span_to_error():
    # PARSER defaults to DEBUG via the constructor; an exception event should
    # promote it to ERROR so users with the filter at INFO/WARNING still see it.
    with mlflow.start_span("s", span_type=SpanType.PARSER) as span:
        span.add_event(SpanEvent.from_exception(ValueError("boom")))
    persisted = get_traces()[0].data.spans[0]
    assert persisted.log_level is SpanLogLevel.ERROR


def test_exception_event_bumps_info_span_to_error():
    with mlflow.start_span("s", span_type=SpanType.CHAT_MODEL) as span:
        span.add_event(SpanEvent.from_exception(RuntimeError("boom")))
    persisted = get_traces()[0].data.spans[0]
    assert persisted.log_level is SpanLogLevel.ERROR


def test_exception_event_does_not_demote_critical():
    # User-set CRITICAL must be preserved when an exception fires.
    with mlflow.start_span("s", log_level=SpanLogLevel.CRITICAL) as span:
        span.add_event(SpanEvent.from_exception(RuntimeError("boom")))
    persisted = get_traces()[0].data.spans[0]
    assert persisted.log_level is SpanLogLevel.CRITICAL


def test_record_exception_bumps_to_error():
    # record_exception() goes through add_event under the hood, so the bump
    # should fire here too.
    span = mlflow.start_span_no_context("s", span_type=SpanType.PARSER)
    span.record_exception(ValueError("boom"))
    span.end()
    persisted = get_traces()[0].data.spans[0]
    assert persisted.log_level is SpanLogLevel.ERROR


def test_traced_function_that_raises_is_promoted_to_error():
    # A plain @mlflow.trace function that throws records an exception event via
    # the decorator's error-handling path, which should promote the span.
    @mlflow.trace
    def fn():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        fn()

    persisted = get_traces()[0].data.spans[0]
    assert persisted.log_level is SpanLogLevel.ERROR


def test_non_exception_event_does_not_bump_log_level():
    # Plain (non-exception) events must not move the level.
    with mlflow.start_span("s", span_type=SpanType.CHAT_MODEL) as span:
        span.add_event(SpanEvent(name="my_event", attributes={"k": "v"}))
    persisted = get_traces()[0].data.spans[0]
    assert persisted.log_level is SpanLogLevel.INFO
