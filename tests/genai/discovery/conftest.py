from unittest.mock import MagicMock

import pytest

from mlflow.entities import Trace, TraceData, TraceInfo
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.span import Span
from mlflow.entities.span_status import SpanStatus, SpanStatusCode


@pytest.fixture
def make_mock_span():
    def _make(
        name="test_span",
        status_code=SpanStatusCode.OK,
        span_id="span-1",
        parent_id=None,
        span_type="UNKNOWN",
        start_time_ns=0,
        end_time_ns=100_000_000,
        model_name=None,
        events=None,
        inputs=None,
        outputs=None,
        status_description="",
    ):
        span = MagicMock(spec=Span)
        span.name = name
        span.span_id = span_id
        span.parent_id = parent_id
        span.span_type = span_type
        span.start_time_ns = start_time_ns
        span.end_time_ns = end_time_ns
        span.model_name = model_name
        span.status = SpanStatus(status_code=status_code, description=status_description)
        span.events = events or []
        span.inputs = inputs
        span.outputs = outputs
        return span

    return _make


@pytest.fixture
def make_trace(make_mock_span):
    def _make(
        trace_id="trace-1",
        request_preview="What is MLflow?",
        response_preview="MLflow is an ML platform.",
        execution_duration=500,
        spans=None,
        assessments=None,
        tags=None,
    ):
        info = MagicMock(spec=TraceInfo)
        info.trace_id = trace_id
        info.request_preview = request_preview
        info.response_preview = response_preview
        info.execution_duration = execution_duration
        info.assessments = assessments or []
        info.tags = tags or {}
        info.trace_metadata = {}

        data = MagicMock(spec=TraceData)
        data.spans = spans or [make_mock_span()]

        trace = MagicMock(spec=Trace)
        trace.info = info
        trace.data = data
        return trace

    return _make


@pytest.fixture
def make_assessment():
    def _make(name, value):
        return Feedback(
            name=name,
            value=value,
            source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="test"),
        )

    return _make
