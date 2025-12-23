from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities.assessment import Assessment, AssessmentSource, Feedback
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.entities.trace import Trace, TraceInfo
from mlflow.exceptions import MlflowException
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.optimizers import MemAlignOptimizer
from mlflow.genai.judges.optimizers.memalign import MemoryAugmentedJudge


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
    assert optimizer._config.retrieval_k == 5
    assert optimizer._config.disable_semantic_memory is False
    assert optimizer._config.disable_episodic_memory is False


def test_init_custom_config():
    optimizer = MemAlignOptimizer(
        model="openai:/gpt-4",
        retrieval_k=3,
        disable_semantic_memory=True,
        embed_dim=256,
    )
    assert optimizer._config.model == "openai:/gpt-4"
    assert optimizer._config.retrieval_k == 3
    assert optimizer._config.disable_semantic_memory is True
    assert optimizer._config.embed_dim == 256


def test_both_memories_disabled_raises_error():
    with pytest.raises(MlflowException, match="At least one memory system must be enabled"):
        MemAlignOptimizer(disable_semantic_memory=True, disable_episodic_memory=True)


def test_align_empty_traces_raises_error(sample_judge):
    with (
        patch("mlflow.genai.judges.optimizers.memalign.construct_dspy_lm"),
        patch("mlflow.genai.judges.optimizers.memalign.create_dspy_signature"),
        patch("mlflow.genai.judges.optimizers.memalign_utils.extract_request_from_trace"),
        patch("mlflow.genai.judges.optimizers.memalign_utils.extract_response_from_trace"),
    ):
        optimizer = MemAlignOptimizer()
        with pytest.raises(MlflowException, match="No traces provided"):
            optimizer.align(sample_judge, [])


def test_align_no_valid_feedback_raises_error(sample_judge):
    with (
        patch("mlflow.genai.judges.optimizers.memalign.construct_dspy_lm"),
        patch("mlflow.genai.judges.optimizers.memalign.create_dspy_signature"),
        patch(
            "mlflow.genai.judges.optimizers.memalign_utils.extract_request_from_trace"
        ) as mock_extract_request,
        patch(
            "mlflow.genai.judges.optimizers.memalign_utils.extract_response_from_trace"
        ) as mock_extract_response,
    ):
        mock_trace = MagicMock(spec=Trace)
        mock_trace.info.trace_id = "test_trace"
        mock_trace.info.assessments = []

        mock_extract_request.return_value = {"question": "test"}
        mock_extract_response.return_value = {"answer": "test"}

        optimizer = MemAlignOptimizer()
        with pytest.raises(MlflowException, match="No valid feedback records found"):
            optimizer.align(sample_judge, [mock_trace])


def test_memory_augmented_judge_init(sample_judge):
    from mlflow.genai.judges.optimizers.dspy_utils import create_dspy_signature
    from mlflow.genai.judges.optimizers.memalign_config import MemAlignConfig

    with (
        patch("mlflow.genai.judges.optimizers.memalign.construct_dspy_lm") as mock_construct_lm,
        patch("mlflow.genai.judges.optimizers.memalign.dspy.Predict"),
        patch("mlflow.genai.judges.optimizers.memalign.dspy.Embedder"),
    ):
        config = MemAlignConfig(model="openai:/gpt-4")
        signature = create_dspy_signature(sample_judge)

        mock_lm = MagicMock()
        mock_construct_lm.return_value = mock_lm

        judge = MemoryAugmentedJudge(
            base_judge=sample_judge,
            config=config,
            base_signature=signature,
            feedback_records=[],
        )

        assert judge.name == sample_judge.name
        assert judge.model == sample_judge.model
        assert isinstance(judge._feedback_history, dict)


def test_memory_augmented_judge_properties(sample_judge):
    from mlflow.genai.judges.optimizers.dspy_utils import create_dspy_signature
    from mlflow.genai.judges.optimizers.memalign_config import MemAlignConfig

    with (
        patch("mlflow.genai.judges.optimizers.memalign.construct_dspy_lm") as mock_construct_lm,
        patch("mlflow.genai.judges.optimizers.memalign.dspy.Predict"),
        patch("mlflow.genai.judges.optimizers.memalign.dspy.Embedder"),
    ):
        config = MemAlignConfig(model="openai:/gpt-4")
        signature = create_dspy_signature(sample_judge)

        mock_lm = MagicMock()
        mock_construct_lm.return_value = mock_lm

        judge = MemoryAugmentedJudge(
            base_judge=sample_judge,
            config=config,
            base_signature=signature,
            feedback_records=[],
        )

        assert judge.name == "test_judge"
        assert "openai:/gpt-4" in judge.model
        assert "Evaluate if" in judge.instructions
        assert len(judge.get_input_fields()) > 0


def test_get_guideline_texts_empty(sample_judge):
    from mlflow.genai.judges.optimizers.dspy_utils import create_dspy_signature
    from mlflow.genai.judges.optimizers.memalign_config import MemAlignConfig

    with (
        patch("mlflow.genai.judges.optimizers.memalign.construct_dspy_lm"),
        patch("mlflow.genai.judges.optimizers.memalign.dspy.Predict"),
    ):
        config = MemAlignConfig(model="openai:/gpt-4", disable_semantic_memory=True)
        signature = create_dspy_signature(sample_judge)

        judge = MemoryAugmentedJudge(
            base_judge=sample_judge,
            config=config,
            base_signature=signature,
            feedback_records=[],
        )

        guidelines = judge._get_guideline_texts()
        assert guidelines == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
