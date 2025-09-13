"""Shared test fixtures for optimizer tests."""

import json
import time
from typing import Any
from unittest.mock import Mock

import dspy
import pytest
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.span import Span
from mlflow.entities.trace import Trace, TraceData, TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges.base import Judge, JudgeField
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY
from mlflow.tracing.utils import build_otel_context


class MockJudge(Judge):
    """Mock judge implementation for testing."""

    def __init__(self, name="mock_judge", instructions=None, model=None, **kwargs):
        super().__init__(name=name, **kwargs)
        # Use a proper template with variables for testing
        self._instructions = (
            instructions or "Evaluate if the {{outputs}} properly addresses {{inputs}}"
        )
        self._model = model

    @property
    def instructions(self) -> str:
        return self._instructions

    @property
    def model(self) -> str:
        """Get the model for this judge."""
        return self._model

    def __call__(self, inputs, outputs, expectations=None, trace=None):
        # Simple mock implementation
        return "pass" if "good" in str(outputs).lower() else "fail"

    def get_input_fields(self) -> list[JudgeField]:
        """Get the input fields for this mock judge."""
        return [
            JudgeField(name="inputs", description="Test inputs"),
            JudgeField(name="outputs", description="Test outputs"),
        ]


def _create_trace_helper(
    trace_id: str,
    assessments: list[Feedback] | None = None,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    context_trace_id: int = 12345,
    context_span_id: int = 111,
) -> Trace:
    """Helper function to create traces with less duplication."""
    current_time_ns = int(time.time() * 1e9)

    # Build attributes dict
    attributes = {"mlflow.traceRequestId": json.dumps(trace_id)}
    if inputs is not None:
        attributes["mlflow.spanInputs"] = json.dumps(inputs)
    if outputs is not None:
        attributes["mlflow.spanOutputs"] = json.dumps(outputs)
    attributes["mlflow.spanType"] = json.dumps("CHAIN")

    # Create OpenTelemetry span
    otel_span = OTelReadableSpan(
        name="root_span",
        context=build_otel_context(context_trace_id, context_span_id),
        parent=None,
        start_time=current_time_ns,
        end_time=current_time_ns + 1000000,
        attributes=attributes,
    )

    # Create real Span object
    real_span = Span(otel_span)

    # Create real TraceInfo
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=int(time.time() * 1000),
        state=TraceState.OK,
        execution_duration=1000,
        trace_metadata={TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)},
        tags={},
        assessments=assessments or [],
        request_preview=json.dumps(inputs) if inputs else None,
        response_preview=json.dumps(outputs) if outputs else None,
    )

    # Create real TraceData and Trace
    trace_data = TraceData(spans=[real_span])
    return Trace(info=trace_info, data=trace_data)


@pytest.fixture
def mock_judge():
    """Create a mock judge for testing."""
    return MockJudge(model="openai:/gpt-3.5-turbo")


@pytest.fixture
def sample_trace_with_assessment():
    """Create a sample trace with human assessment for testing."""
    # Create a real assessment object (Feedback) with mixed case/whitespace to test sanitization
    assessment = Feedback(
        name="  Mock_JUDGE  ",
        value="pass",
        rationale="This looks good",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="test_user"),
    )

    return _create_trace_helper(
        trace_id="test_trace_001",
        assessments=[assessment],
        inputs={"inputs": "test input"},
        outputs={"outputs": "test output"},
        context_trace_id=12345,
        context_span_id=111,
    )


@pytest.fixture
def sample_trace_without_assessment():
    """Create a sample trace without human assessment for testing."""
    return _create_trace_helper(
        trace_id="test_trace_001",
        assessments=[],  # No assessments
        inputs={"inputs": "test input"},
        outputs={"outputs": "test output"},
        context_trace_id=12346,
        context_span_id=112,
    )


@pytest.fixture
def trace_with_expectations():
    """Create a trace with expectations."""
    from mlflow.entities import Expectation

    # Create assessments
    judge_assessment = Feedback(
        name="mock_judge",
        value="pass",
        rationale="Meets expectations",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="test_user"),
    )

    expectation1 = Expectation(
        trace_id="test_trace_with_expectations",
        name="accuracy",
        value="Should be at least 90% accurate",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="test_user"),
    )

    expectation2 = Expectation(
        trace_id="test_trace_with_expectations",
        name="format",
        value="Should return JSON format",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="test_user"),
    )

    return _create_trace_helper(
        trace_id="test_trace_with_expectations",
        assessments=[judge_assessment, expectation1, expectation2],
        inputs={"inputs": "test input"},
        outputs={"outputs": "test output"},
        context_trace_id=12352,
        context_span_id=118,
    )


@pytest.fixture
def trace_without_inputs():
    """Create a trace without inputs (only outputs)."""
    assessment = Feedback(
        name="mock_judge",
        value="pass",
        rationale="Output only",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="test_user"),
    )

    return _create_trace_helper(
        trace_id="test_trace_no_inputs",
        assessments=[assessment],
        inputs=None,  # No inputs
        outputs={"outputs": "test output"},
        context_trace_id=12350,
        context_span_id=116,
    )


@pytest.fixture
def trace_without_outputs():
    """Create a trace without outputs (only inputs)."""
    assessment = Feedback(
        name="mock_judge",
        value="pass",
        rationale="Input only",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="test_user"),
    )

    return _create_trace_helper(
        trace_id="test_trace_no_outputs",
        assessments=[assessment],
        inputs={"inputs": "test input"},
        outputs=None,  # No outputs
        context_trace_id=12351,
        context_span_id=117,
    )


@pytest.fixture
def trace_with_nested_request_response():
    """Create a trace with nested request/response structure."""
    assessment = Feedback(
        name="mock_judge",
        value="pass",
        rationale="Complex nested structure handled well",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="test_user"),
        create_time_ms=int(time.time() * 1000),
    )

    nested_inputs = {"query": {"text": "nested input", "context": {"key": "value"}}}
    nested_outputs = {"result": {"answer": "nested output", "metadata": {"score": 0.9}}}

    return _create_trace_helper(
        trace_id="test_trace_nested",
        assessments=[assessment],
        inputs=nested_inputs,
        outputs=nested_outputs,
        context_trace_id=12347,
        context_span_id=113,
    )


@pytest.fixture
def trace_with_list_request_response():
    """Create a trace with list-based request/response."""
    # Create actual trace with real MLflow objects

    # Create a real assessment object (Feedback)
    assessment = Feedback(
        name="mock_judge",
        value="fail",
        rationale="List processing needs improvement",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="test_user"),
        create_time_ms=int(time.time() * 1000),
    )

    # Create OpenTelemetry span with list structure
    current_time_ns = int(time.time() * 1e9)
    list_inputs = {"items": ["item1", "item2", "item3"]}
    list_outputs = {"results": ["result1", "result2"]}

    otel_span = OTelReadableSpan(
        name="root_span",
        context=build_otel_context(12348, 114),
        parent=None,
        start_time=current_time_ns,
        end_time=current_time_ns + 1000000,
        attributes={
            "mlflow.traceRequestId": json.dumps("test_trace_list"),
            "mlflow.spanInputs": json.dumps(list_inputs),
            "mlflow.spanOutputs": json.dumps(list_outputs),
            "mlflow.spanType": json.dumps("CHAIN"),
        },
    )

    # Create real Span object
    real_span = Span(otel_span)

    # Create real TraceInfo
    trace_info = TraceInfo(
        trace_id="test_trace_list",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=int(time.time() * 1000),
        state=TraceState.OK,
        execution_duration=1000,
        trace_metadata={TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)},
        tags={},
        assessments=[assessment],
        request_preview='["item1", "item2", "item3"]',
        response_preview='["result1", "result2"]',
    )

    # Create real TraceData
    trace_data = TraceData(spans=[real_span])

    # Create real Trace object
    return Trace(info=trace_info, data=trace_data)


@pytest.fixture
def trace_with_string_request_response():
    """Create a trace with simple string request/response."""
    # Create actual trace with real MLflow objects

    # Create a real assessment object (Feedback)
    assessment = Feedback(
        name="mock_judge",
        value="pass",
        rationale="Simple string handled correctly",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="test_user"),
        create_time_ms=int(time.time() * 1000),
    )

    # Create OpenTelemetry span with string inputs/outputs
    current_time_ns = int(time.time() * 1e9)
    string_input = "What is the capital of France?"
    string_output = "Paris"

    otel_span = OTelReadableSpan(
        name="root_span",
        context=build_otel_context(12349, 115),
        parent=None,
        start_time=current_time_ns,
        end_time=current_time_ns + 1000000,
        attributes={
            "mlflow.traceRequestId": json.dumps("test_trace_string"),
            "mlflow.spanInputs": json.dumps(string_input),
            "mlflow.spanOutputs": json.dumps(string_output),
            "mlflow.spanType": json.dumps("CHAIN"),
        },
    )

    # Create real Span object
    real_span = Span(otel_span)

    # Create real TraceInfo
    trace_info = TraceInfo(
        trace_id="test_trace_string",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=int(time.time() * 1000),
        state=TraceState.OK,
        execution_duration=1000,
        trace_metadata={TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)},
        tags={},
        assessments=[assessment],
        request_preview=json.dumps(string_input),
        response_preview=json.dumps(string_output),
    )

    # Create real TraceData
    trace_data = TraceData(spans=[real_span])

    # Create real Trace object
    return Trace(info=trace_info, data=trace_data)


@pytest.fixture
def trace_with_mixed_types():
    """Create a trace with mixed data types in request/response."""
    # Create actual trace with real MLflow objects

    # Create a real assessment object (Feedback)
    assessment = Feedback(
        name="mock_judge",
        value="pass",
        rationale="Mixed types processed successfully",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="test_user"),
        create_time_ms=int(time.time() * 1000),
    )

    # Create OpenTelemetry span with mixed types
    current_time_ns = int(time.time() * 1e9)
    mixed_inputs = {"prompt": "test", "temperature": 0.7, "max_tokens": 100}
    mixed_outputs = {"text": "response", "tokens_used": 50, "success": True}

    otel_span = OTelReadableSpan(
        name="root_span",
        context=build_otel_context(12350, 116),
        parent=None,
        start_time=current_time_ns,
        end_time=current_time_ns + 1000000,
        attributes={
            "mlflow.traceRequestId": json.dumps("test_trace_mixed"),
            "mlflow.spanInputs": json.dumps(mixed_inputs),
            "mlflow.spanOutputs": json.dumps(mixed_outputs),
            "mlflow.spanType": json.dumps("CHAIN"),
        },
    )

    # Create real Span object
    real_span = Span(otel_span)

    # Create real TraceInfo
    trace_info = TraceInfo(
        trace_id="test_trace_mixed",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=int(time.time() * 1000),
        state=TraceState.OK,
        execution_duration=1000,
        trace_metadata={TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)},
        tags={},
        assessments=[assessment],
        request_preview=json.dumps(mixed_inputs),
        response_preview=json.dumps(mixed_outputs),
    )

    # Create real TraceData
    trace_data = TraceData(spans=[real_span])

    # Create real Trace object
    return Trace(info=trace_info, data=trace_data)


@pytest.fixture
def sample_traces_with_assessments():
    """Create multiple sample traces with assessments."""
    traces = []
    # Create actual traces with real MLflow objects

    for i in range(5):
        # Create a real assessment object (Feedback)
        assessment = Feedback(
            name="mock_judge",
            value="pass" if i % 2 == 0 else "fail",
            rationale=f"Rationale for trace {i}",
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="test_user"),
            create_time_ms=int(time.time() * 1000) + i,  # Slightly different times
        )

        # Create OpenTelemetry span
        current_time_ns = int(time.time() * 1e9) + i * 1000000
        inputs = {"inputs": f"test input {i}"}
        outputs = {"outputs": f"test output {i}"}

        otel_span = OTelReadableSpan(
            name="root_span",
            context=build_otel_context(12351 + i, 117 + i),
            parent=None,
            start_time=current_time_ns,
            end_time=current_time_ns + 1000000,
            attributes={
                "mlflow.traceRequestId": json.dumps(f"test_trace_{i:03d}"),
                "mlflow.spanInputs": json.dumps(inputs),
                "mlflow.spanOutputs": json.dumps(outputs),
                "mlflow.spanType": json.dumps("CHAIN"),
            },
        )

        # Create real Span object
        real_span = Span(otel_span)

        # Create real TraceInfo
        trace_info = TraceInfo(
            trace_id=f"test_trace_{i:03d}",
            trace_location=TraceLocation.from_experiment_id("0"),
            request_time=int(time.time() * 1000) + i,
            state=TraceState.OK,
            execution_duration=1000,
            trace_metadata={TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)},
            tags={},
            assessments=[assessment],
            request_preview=json.dumps(inputs),
            response_preview=json.dumps(outputs),
        )

        # Create real TraceData
        trace_data = TraceData(spans=[real_span])

        # Create real Trace object
        trace = Trace(info=trace_info, data=trace_data)
        traces.append(trace)

    return traces


@pytest.fixture
def mock_dspy_example():
    """Create a DSPy example for testing."""
    # Return an actual dspy.Example
    example = dspy.Example(
        inputs="test inputs", outputs="test outputs", result="pass", rationale="test rationale"
    )
    return example.with_inputs("inputs", "outputs")


@pytest.fixture
def mock_dspy_program():
    """Create a mock DSPy program for testing."""
    mock_program = Mock()
    mock_program.signature = Mock()
    mock_program.signature.instructions = "Optimized instructions"
    return mock_program


@pytest.fixture
def mock_dspy_optimizer():
    """Create a mock DSPy optimizer for testing."""

    def create_mock_optimizer():
        mock_optimizer = Mock()
        mock_optimizer.__class__.__name__ = "MockOptimizer"
        mock_optimizer.compile.return_value = Mock()
        return mock_optimizer

    return create_mock_optimizer


@pytest.fixture
def trace_with_two_human_assessments():
    """Create a trace with two HUMAN assessments with different timestamps."""
    # Create two real assessment objects (Feedback) with different timestamps
    older_assessment = Feedback(
        name="mock_judge",
        value="fail",
        rationale="First assessment - should not be used",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user1"),
        create_time_ms=int(time.time() * 1000) - 1000,  # 1 second older
    )

    newer_assessment = Feedback(
        name="mock_judge",
        value="pass",
        rationale="Second assessment - should be used (more recent)",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user2"),
        create_time_ms=int(time.time() * 1000),  # Current time (newer)
    )

    # Create OpenTelemetry span
    current_time_ns = int(time.time() * 1e9)
    otel_span = OTelReadableSpan(
        name="root_span",
        context=build_otel_context(12360, 160),
        parent=None,
        start_time=current_time_ns,
        end_time=current_time_ns + 1000000,
        attributes={
            "mlflow.traceRequestId": json.dumps("test_trace_two_human"),
            "mlflow.spanInputs": json.dumps({"inputs": "test input"}),
            "mlflow.spanOutputs": json.dumps({"outputs": "test output"}),
            "mlflow.spanType": json.dumps("CHAIN"),
        },
    )

    real_span = Span(otel_span)

    trace_info = TraceInfo(
        trace_id="test_trace_two_human",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=int(time.time() * 1000),
        state=TraceState.OK,
        execution_duration=1000,
        trace_metadata={TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)},
        tags={},
        assessments=[older_assessment, newer_assessment],  # Order shouldn't matter due to sorting
        request_preview='{"inputs": "test input"}',
        response_preview='{"outputs": "test output"}',
    )

    trace_data = TraceData(spans=[real_span])
    return Trace(info=trace_info, data=trace_data)


@pytest.fixture
def trace_with_human_and_llm_assessments():
    """Create a trace with both HUMAN and LLM_JUDGE assessments (HUMAN should be prioritized)."""
    # Create HUMAN assessment (older timestamp)
    human_assessment = Feedback(
        name="mock_judge",
        value="fail",
        rationale="Human assessment - should be prioritized",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="human_user"),
        create_time_ms=int(time.time() * 1000) - 2000,  # 2 seconds older
    )

    # Create LLM_JUDGE assessment (newer timestamp)
    llm_assessment = Feedback(
        name="mock_judge",
        value="pass",
        rationale="LLM assessment - should not be used despite being newer",
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4"),
        create_time_ms=int(time.time() * 1000),  # Current time (newer)
    )

    # Create OpenTelemetry span
    current_time_ns = int(time.time() * 1e9)
    otel_span = OTelReadableSpan(
        name="root_span",
        context=build_otel_context(12361, 161),
        parent=None,
        start_time=current_time_ns,
        end_time=current_time_ns + 1000000,
        attributes={
            "mlflow.traceRequestId": json.dumps("test_trace_human_llm"),
            "mlflow.spanInputs": json.dumps({"inputs": "test input"}),
            "mlflow.spanOutputs": json.dumps({"outputs": "test output"}),
            "mlflow.spanType": json.dumps("CHAIN"),
        },
    )

    real_span = Span(otel_span)

    trace_info = TraceInfo(
        trace_id="test_trace_human_llm",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=int(time.time() * 1000),
        state=TraceState.OK,
        execution_duration=1000,
        trace_metadata={TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)},
        tags={},
        # Order shouldn't matter - HUMAN should be chosen
        assessments=[llm_assessment, human_assessment],
        request_preview='{"inputs": "test input"}',
        response_preview='{"outputs": "test output"}',
    )

    trace_data = TraceData(spans=[real_span])
    return Trace(info=trace_info, data=trace_data)


class MockDSPyLM(dspy.BaseLM):
    """Mock DSPy LM class for testing that inherits from DSPy's BaseLM."""

    def __init__(self, model_name):
        super().__init__(model_name)
        self.model = model_name
        self.name = model_name
        self._context_calls = []

    def basic_request(self, prompt, **kwargs):
        # Track that this LM was called
        self._context_calls.append(
            {
                "model": self.model,
                "prompt": prompt,
                "kwargs": kwargs,
                "context": "lm_basic_request_called",
            }
        )

        # Return a default answer
        return [{"text": '{"result": "pass", "rationale": "test rationale"}'}]

    def __call__(self, *args, **kwargs):
        return self.basic_request(str(args), **kwargs)

    @property
    def context_calls(self):
        return self._context_calls
