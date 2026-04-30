import logging

import pytest

import mlflow
from mlflow.entities import SpanLogLevel
from mlflow.entities.span import Span, SpanType
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.utils.default_log_level import default_log_level_for_span_type

from tests.tracing.helper import get_traces


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (SpanLogLevel.DEBUG, SpanLogLevel.DEBUG),
        (SpanLogLevel.CRITICAL, SpanLogLevel.CRITICAL),
        (10, SpanLogLevel.DEBUG),
        (20, SpanLogLevel.INFO),
        (30, SpanLogLevel.WARNING),
        (40, SpanLogLevel.ERROR),
        (50, SpanLogLevel.CRITICAL),
        ("DEBUG", SpanLogLevel.DEBUG),
        ("info", SpanLogLevel.INFO),
        ("Warning", SpanLogLevel.WARNING),
        ("WARN", SpanLogLevel.WARNING),
        ("warn", SpanLogLevel.WARNING),
        ("  ERROR  ", SpanLogLevel.ERROR),
        (logging.INFO, SpanLogLevel.INFO),
        (logging.CRITICAL, SpanLogLevel.CRITICAL),
    ],
)
def test_from_value_accepts_enum_int_and_string_forms(value, expected):
    assert SpanLogLevel.from_value(value) is expected


@pytest.mark.parametrize("value", ["NOPE", "TRACE", "FATAL", "INFOO", ""])
def test_from_value_rejects_invalid_string(value):
    with pytest.raises(MlflowException, match="Invalid SpanLogLevel"):
        SpanLogLevel.from_value(value)


@pytest.mark.parametrize("value", [0, 7, 100, -1])
def test_from_value_rejects_invalid_int(value):
    with pytest.raises(MlflowException, match="Invalid SpanLogLevel"):
        SpanLogLevel.from_value(value)


@pytest.mark.parametrize("value", [None, 1.5, ["INFO"], object(), True, False])
def test_from_value_rejects_invalid_type(value):
    # bool is rejected even though it's an int subclass.
    with pytest.raises(MlflowException, match="must be"):
        SpanLogLevel.from_value(value)


def test_log_level_unset_by_default():
    with mlflow.start_span("s"):
        pass
    persisted = get_traces()[0].data.spans[0]
    assert persisted.log_level is None
    assert SpanAttributeKey.LOG_LEVEL not in persisted.attributes


@pytest.mark.parametrize(
    ("set_value", "expected"),
    [
        (SpanLogLevel.WARNING, SpanLogLevel.WARNING),
        (40, SpanLogLevel.ERROR),
        ("info", SpanLogLevel.INFO),
        ("WARN", SpanLogLevel.WARNING),
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


def test_start_span_kwarg():
    with mlflow.start_span("s", log_level="WARNING"):
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
        (SpanType.LLM, SpanLogLevel.INFO),
        (SpanType.CHAT_MODEL, SpanLogLevel.INFO),
        (SpanType.AGENT, SpanLogLevel.INFO),
        (SpanType.TOOL, SpanLogLevel.INFO),
        (SpanType.RETRIEVER, SpanLogLevel.INFO),
        (SpanType.EMBEDDING, SpanLogLevel.INFO),
        (SpanType.MEMORY, SpanLogLevel.INFO),
        (SpanType.WORKFLOW, SpanLogLevel.INFO),
        (SpanType.TASK, SpanLogLevel.INFO),
        (SpanType.GUARDRAIL, SpanLogLevel.INFO),
        (SpanType.EVALUATOR, SpanLogLevel.INFO),
        (SpanType.UNKNOWN, SpanLogLevel.INFO),
        (SpanType.CHAIN, SpanLogLevel.DEBUG),
        (SpanType.PARSER, SpanLogLevel.DEBUG),
        (SpanType.RERANKER, SpanLogLevel.DEBUG),
        # Custom (non-built-in) span types fall through to INFO.
        ("MY_CUSTOM_TYPE", SpanLogLevel.INFO),
        (None, SpanLogLevel.INFO),
    ],
)
def test_default_log_level_for_span_type_mapping(span_type, expected):
    assert default_log_level_for_span_type(span_type) is expected


def test_default_log_level_helper_threads_through_start_span_no_context():
    # Mirrors the autolog pattern: stamp the helper's default at span creation time.
    span = mlflow.start_span_no_context(
        "s",
        span_type=SpanType.CHAT_MODEL,
        log_level=default_log_level_for_span_type(SpanType.CHAT_MODEL),
    )
    span.end()
    persisted = get_traces()[0].data.spans[0]
    assert persisted.log_level is SpanLogLevel.INFO


def test_default_log_level_helper_threads_debug_for_internal_span_types():
    span = mlflow.start_span_no_context(
        "s",
        span_type=SpanType.PARSER,
        log_level=default_log_level_for_span_type(SpanType.PARSER),
    )
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
