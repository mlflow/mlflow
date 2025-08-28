"""Shared test fixtures for optimizer tests."""

from unittest.mock import Mock
import pytest

from mlflow.entities.trace import Trace, TraceData, TraceInfo
from mlflow.genai.judges.base import Judge


class MockJudge(Judge):
    """Mock judge implementation for testing."""
    
    def __init__(self, name="mock_judge", description="A mock judge for testing", **kwargs):
        super().__init__(name=name, **kwargs)
        self._description = description
    
    @property
    def description(self) -> str:
        return self._description
    
    def __call__(self, inputs, outputs, expectations=None, trace=None):
        # Simple mock implementation
        return "pass" if "good" in str(outputs).lower() else "fail"


@pytest.fixture
def mock_judge():
    """Create a mock judge for testing."""
    return MockJudge()


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