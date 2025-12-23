from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities.assessment import Assessment, AssessmentSource, Feedback
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.entities.trace import Trace, TraceInfo
from mlflow.exceptions import MlflowException
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.optimizers import MemAlignOptimizer


@pytest.fixture
def sample_judge():
    return make_judge(
        name="test_judge",
        instructions="Evaluate if {{ outputs }} correctly answers {{ inputs }}",
        model="openai:/gpt-4",
    )


@pytest.fixture
def sample_traces():
    traces = []
    for i in range(5):
        trace_info = TraceInfo(
            request_id=f"req_{i}",
            experiment_id="exp_1",
            timestamp_ms=1000 + i,
            execution_time_ms=100,
            status="OK",
            request_metadata={},
            tags={},
            trace_id=f"trace_{i}",
        )

        assessment = Assessment(
            name="test_judge",
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user1"),
            timestamp_ms=1000 + i,
            feedback=Feedback(value="yes" if i % 2 == 0 else "no", rationale=f"Reason {i}"),
        )
        trace_info.assessments = [assessment]

        trace_data = MagicMock()
        trace_data.spans = []

        trace = MagicMock(spec=Trace)
        trace.info = trace_info
        trace.data = trace_data
        traces.append(trace)

    return traces


def test_init_default_config():
    optimizer = MemAlignOptimizer()
    assert optimizer._retrieval_k == 5
    assert optimizer._embedder_name == "openai/text-embedding-3-small"
    assert optimizer._embed_dim == 512


def test_init_custom_config():
    optimizer = MemAlignOptimizer(
        distillation_model="openai:/gpt-4",
        retrieval_k=3,
        embed_dim=256,
    )
    assert optimizer._distillation_model == "openai:/gpt-4"
    assert optimizer._retrieval_k == 3
    assert optimizer._embed_dim == 256


def test_align_empty_traces_raises_error(sample_judge):
    optimizer = MemAlignOptimizer()
    with pytest.raises(MlflowException, match="No traces provided"):
        optimizer.align(sample_judge, [])


def test_align_no_valid_feedback_raises_error(sample_judge):
    with (
        patch(
            "mlflow.genai.judges.optimizers.memalign.optimizer.trace_to_dspy_example"
        ) as mock_trace_to_example,
    ):
        mock_trace_to_example.return_value = None

        mock_trace = MagicMock()
        mock_trace.info.trace_id = "test_trace"

        optimizer = MemAlignOptimizer()
        with pytest.raises(MlflowException, match="No valid feedback records found"):
            optimizer.align(sample_judge, [mock_trace])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
