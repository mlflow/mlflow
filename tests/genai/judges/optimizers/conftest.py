"""Shared test fixtures for optimizer tests."""

from unittest.mock import Mock

import dspy
import pytest

from mlflow.entities.trace import Trace, TraceData, TraceInfo
from mlflow.genai.judges.base import Judge, JudgeField


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
        ]


@pytest.fixture
def mock_judge():
    """Create a mock judge for testing."""
    return MockJudge(model="openai:/gpt-3.5-turbo")


@pytest.fixture
def sample_trace_with_assessment():
    """Create a sample trace with human assessment for testing."""
    # Mock assessment
    mock_assessment = Mock()
    mock_assessment.name = "mock_judge"
    mock_assessment.source.source_type = "HUMAN"
    mock_assessment.feedback.value = "pass"
    mock_assessment.rationale = "This looks good"

    # Mock trace info
    mock_trace_info = Mock(spec=TraceInfo)
    mock_trace_info.trace_id = "test_trace_001"
    mock_trace_info.assessments = [mock_assessment]
    mock_trace_info.request_preview = '{"inputs": "test input"}'
    mock_trace_info.response_preview = '{"outputs": "test output"}'

    # Mock trace data
    mock_trace_data = Mock(spec=TraceData)
    mock_trace_data.request = {"inputs": "test input"}
    mock_trace_data.response = {"outputs": "test output"}

    # Mock trace
    mock_trace = Mock(spec=Trace)
    mock_trace.info = mock_trace_info
    mock_trace.data = mock_trace_data

    return mock_trace


@pytest.fixture
def sample_trace_without_assessment():
    """Create a sample trace without human assessment for testing."""
    # Mock trace info
    mock_trace_info = Mock(spec=TraceInfo)
    mock_trace_info.trace_id = "test_trace_001"
    mock_trace_info.request_preview = '{"inputs": "test input"}'
    mock_trace_info.response_preview = '{"outputs": "test output"}'

    # Mock trace data
    mock_trace_data = Mock(spec=TraceData)
    mock_trace_data.request = {"inputs": "test input"}
    mock_trace_data.response = {"outputs": "test output"}

    # Mock trace
    mock_trace = Mock(spec=Trace)
    mock_trace.info = mock_trace_info
    mock_trace.data = mock_trace_data

    return mock_trace


@pytest.fixture
def trace_with_nested_request_response():
    """Create a trace with nested request/response structure."""
    mock_assessment = Mock()
    mock_assessment.name = "mock_judge"
    mock_assessment.source.source_type = "HUMAN"
    mock_assessment.feedback.value = "pass"
    mock_assessment.rationale = "Complex nested structure handled well"

    mock_trace_info = Mock(spec=TraceInfo)
    mock_trace_info.trace_id = "test_trace_nested"
    mock_trace_info.assessments = [mock_assessment]
    mock_trace_info.request_preview = (
        '{"query": {"text": "nested input", "context": {"key": "value"}}}'
    )
    mock_trace_info.response_preview = (
        '{"result": {"answer": "nested output", "metadata": {"score": 0.9}}}'
    )

    mock_trace_data = Mock(spec=TraceData)
    mock_trace_data.request = {"query": {"text": "nested input", "context": {"key": "value"}}}
    mock_trace_data.response = {"result": {"answer": "nested output", "metadata": {"score": 0.9}}}

    mock_trace = Mock(spec=Trace)
    mock_trace.info = mock_trace_info
    mock_trace.data = mock_trace_data

    return mock_trace


@pytest.fixture
def trace_with_list_request_response():
    """Create a trace with list-based request/response."""
    mock_assessment = Mock()
    mock_assessment.name = "mock_judge"
    mock_assessment.source.source_type = "HUMAN"
    mock_assessment.feedback.value = "fail"
    mock_assessment.rationale = "List processing needs improvement"

    mock_trace_info = Mock(spec=TraceInfo)
    mock_trace_info.trace_id = "test_trace_list"
    mock_trace_info.assessments = [mock_assessment]
    mock_trace_info.request_preview = '["item1", "item2", "item3"]'
    mock_trace_info.response_preview = '["result1", "result2"]'

    mock_trace_data = Mock(spec=TraceData)
    mock_trace_data.request = ["item1", "item2", "item3"]
    mock_trace_data.response = ["result1", "result2"]

    mock_trace = Mock(spec=Trace)
    mock_trace.info = mock_trace_info
    mock_trace.data = mock_trace_data

    return mock_trace


@pytest.fixture
def trace_with_string_request_response():
    """Create a trace with simple string request/response."""
    mock_assessment = Mock()
    mock_assessment.name = "mock_judge"
    mock_assessment.source.source_type = "HUMAN"
    mock_assessment.feedback.value = "pass"
    mock_assessment.rationale = "Simple string handled correctly"

    mock_trace_info = Mock(spec=TraceInfo)
    mock_trace_info.trace_id = "test_trace_string"
    mock_trace_info.assessments = [mock_assessment]
    mock_trace_info.request_preview = '"What is the capital of France?"'
    mock_trace_info.response_preview = '"Paris"'

    mock_trace_data = Mock(spec=TraceData)
    mock_trace_data.request = "What is the capital of France?"
    mock_trace_data.response = "Paris"

    mock_trace = Mock(spec=Trace)
    mock_trace.info = mock_trace_info
    mock_trace.data = mock_trace_data

    return mock_trace


@pytest.fixture
def trace_with_mixed_types():
    """Create a trace with mixed data types in request/response."""
    mock_assessment = Mock()
    mock_assessment.name = "mock_judge"
    mock_assessment.source.source_type = "HUMAN"
    mock_assessment.feedback.value = "pass"
    mock_assessment.rationale = "Mixed types processed successfully"

    mock_trace_info = Mock(spec=TraceInfo)
    mock_trace_info.trace_id = "test_trace_mixed"
    mock_trace_info.assessments = [mock_assessment]
    mock_trace_info.request_preview = '{"prompt": "test", "temperature": 0.7, "max_tokens": 100}'
    mock_trace_info.response_preview = '{"text": "response", "tokens_used": 50, "success": true}'

    mock_trace_data = Mock(spec=TraceData)
    mock_trace_data.request = {"prompt": "test", "temperature": 0.7, "max_tokens": 100}
    mock_trace_data.response = {"text": "response", "tokens_used": 50, "success": True}

    mock_trace = Mock(spec=Trace)
    mock_trace.info = mock_trace_info
    mock_trace.data = mock_trace_data

    return mock_trace


@pytest.fixture
def sample_traces_with_assessments():
    """Create multiple sample traces with assessments."""
    traces = []

    for i in range(5):
        # Mock assessment
        mock_assessment = Mock()
        mock_assessment.name = "mock_judge"
        mock_assessment.source.source_type = "HUMAN"
        mock_assessment.feedback.value = "pass" if i % 2 == 0 else "fail"
        mock_assessment.rationale = f"Rationale for trace {i}"

        # Mock trace info
        mock_trace_info = Mock(spec=TraceInfo)
        mock_trace_info.trace_id = f"test_trace_{i:03d}"
        mock_trace_info.assessments = [mock_assessment]
        mock_trace_info.request_preview = f'{{"inputs": "test input {i}"}}'
        mock_trace_info.response_preview = f'{{"outputs": "test output {i}"}}'

        # Mock trace data
        mock_trace_data = Mock(spec=TraceData)
        mock_trace_data.request = {"inputs": f"test input {i}"}
        mock_trace_data.response = {"outputs": f"test output {i}"}

        # Mock trace
        mock_trace = Mock(spec=Trace)
        mock_trace.info = mock_trace_info
        mock_trace.data = mock_trace_data

        traces.append(mock_trace)

    return traces


@pytest.fixture
def mock_dspy_example():
    """Create a mock DSPy example for testing."""
    mock_example = Mock()
    mock_example.inputs = "test inputs"
    mock_example.outputs = "test outputs"
    mock_example.result = "pass"
    mock_example.rationale = "test rationale"

    def mock_with_inputs(*args):
        return mock_example

    mock_example.with_inputs = mock_with_inputs
    return mock_example


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
