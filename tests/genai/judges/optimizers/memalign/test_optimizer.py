from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import mlflow
from mlflow.entities.assessment import Assessment, AssessmentSource, Feedback
from mlflow.entities.assessment_source import AssessmentSourceType
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
def mock_embedder():
    with patch("dspy.Embedder") as mock_embedder_class:
        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder
        yield mock_embedder


@pytest.fixture
def mock_search():
    with patch("dspy.retrievers.Embeddings") as mock_embeddings_class:
        mock_search = MagicMock()
        mock_embeddings_class.return_value = mock_search
        yield mock_search


@pytest.fixture
def mock_distillation_lm():
    with patch(
        "mlflow.genai.judges.optimizers.memalign.utils.construct_dspy_lm"
    ) as mock_construct_lm:
        mock_lm = MagicMock()
        mock_construct_lm.return_value = mock_lm
        yield mock_lm


@contextmanager
def mock_apis(guidelines=None):
    """Context manager for mocking API calls with optional guideline configuration."""
    if guidelines is None:
        guidelines = []

    with (
        patch("dspy.retrievers.Embeddings") as mock_embeddings_class,
        patch("dspy.Embedder") as mock_embedder_class,
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils.construct_dspy_lm"
        ) as mock_construct_lm,
    ):
        # Mock distillation LM
        mock_lm = MagicMock()
        guidelines_json = {"guidelines": [{"guideline_text": g} for g in guidelines]}
        mock_lm.return_value = [f"{guidelines_json}".replace("'", '"')]
        mock_construct_lm.return_value = mock_lm

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        mock_search = MagicMock()
        mock_embeddings_class.return_value = mock_search

        yield {
            "lm": mock_lm,
            "embedder": mock_embedder,
            "search": mock_search,
            "construct_lm": mock_construct_lm,
            "embedder_class": mock_embedder_class,
            "embeddings_class": mock_embeddings_class,
        }


@pytest.fixture
def sample_traces():
    trace_ids = []
    for i in range(5):
        with mlflow.start_span(name=f"test_span_{i}") as span:
            span.set_inputs({"inputs": f"input_{i}"})
            span.set_outputs({"outputs": f"output_{i}"})
            trace_ids.append(mlflow.get_last_active_trace_id())

    traces = mlflow.search_traces(filter_string=None, return_type="list")
    traces = [t for t in traces if t.info.trace_id in trace_ids]

    for i, trace in enumerate(traces):
        assessment = Assessment(
            name="test_judge",
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user1"),
            feedback=Feedback(value="yes", rationale=f"Reason {i}"),
        )
        trace.info.assessments = [assessment]

    return traces


def test_init_default_config():
    optimizer = MemAlignOptimizer()
    assert optimizer._retrieval_k == 5
    assert optimizer._embedding_model == "openai/text-embedding-3-small"
    assert optimizer._embedding_dim == 512


def test_init_custom_config():
    optimizer = MemAlignOptimizer(
        reflection_lm="openai:/gpt-4",
        retrieval_k=3,
        embedding_dim=256,
    )
    assert optimizer._reflection_lm == "openai:/gpt-4"
    assert optimizer._retrieval_k == 3
    assert optimizer._embedding_dim == 256


def test_align_empty_traces_raises_error(sample_judge):
    optimizer = MemAlignOptimizer()
    with pytest.raises(MlflowException, match="No traces provided"):
        optimizer.align(sample_judge, [])


def test_align_no_valid_feedback_raises_error(sample_judge):
    # Create a trace without any assessments - trace_to_dspy_example will return None
    with mlflow.start_span(name="test_span") as span:
        span.set_inputs({"inputs": "test input"})
        span.set_outputs({"outputs": "test output"})

    trace_id = mlflow.get_last_active_trace_id()
    traces = mlflow.search_traces(filter_string=None, return_type="list")
    trace = [t for t in traces if t.info.trace_id == trace_id][0]

    optimizer = MemAlignOptimizer()
    with pytest.raises(MlflowException, match="No valid feedback records found"):
        optimizer.align(sample_judge, [trace])


def test_align_creates_memory_augmented_judge(sample_judge, sample_traces):
    with mock_apis(guidelines=["Guideline 1", "Guideline 2"]):
        optimizer = MemAlignOptimizer(retrieval_k=3)
        aligned_judge = optimizer.align(sample_judge, sample_traces[:3])

        assert aligned_judge is not None
        assert aligned_judge.name == sample_judge.name
        assert aligned_judge.model == sample_judge.model
        assert len(aligned_judge._examples) == 3
        assert len(aligned_judge._semantic_memory) == 2


def test_unalign_removes_traces(sample_judge, sample_traces):
    with mock_apis(guidelines=["Guideline 1"]):
        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces)

        # Verify all examples are present
        num_examples = len(aligned_judge._examples)
        assert num_examples == len(sample_traces)

        traces_to_remove = [sample_traces[1], sample_traces[3]]
        unaligned_judge = aligned_judge.unalign(traces=traces_to_remove)

        # Verify examples for traces 1 and 3 are removed
        assert len(unaligned_judge._examples) == num_examples - 2
        remaining_trace_ids = {
            ex._trace_id for ex in unaligned_judge._examples if hasattr(ex, "_trace_id")
        }
        expected_remaining_trace_ids = {
            sample_traces[i].info.trace_id for i in range(len(sample_traces)) if i not in [1, 3]
        }
        assert remaining_trace_ids == expected_remaining_trace_ids


def test_unalign_no_matching_traces_returns_same_judge(sample_judge, sample_traces):
    with mock_apis(guidelines=[]):
        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces[:3])

        # Create trace with non-existent ID
        mock_trace = MagicMock()
        mock_trace.info.trace_id = "trace_999"

        unaligned_judge = aligned_judge.unalign(traces=[mock_trace])
        assert unaligned_judge is aligned_judge
        assert len(unaligned_judge._examples) == 3


def test_judge_call_uses_semantic_memory(sample_judge, sample_traces):
    with mock_apis(guidelines=["Be concise", "Be clear"]):
        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces[:1])

        assert len(aligned_judge._semantic_memory) == 2
        assert "Be concise" in aligned_judge._semantic_memory
        assert "Be clear" in aligned_judge._semantic_memory


def test_judge_call_retrieves_relevant_examples(sample_judge, sample_traces):
    with mock_apis(guidelines=[]) as mocks:
        # Configure search to return specific indices
        search_results = MagicMock()
        search_results.indices = [0, 2]
        mocks["search"].return_value = search_results

        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces[:3])

        # Mock the predict module to return a result
        mock_prediction = MagicMock()
        mock_prediction.result = "yes"
        mock_prediction.rationale = "Test rationale"
        aligned_judge._predict_module = MagicMock(return_value=mock_prediction)

        assessment = aligned_judge(inputs="test input", outputs="test output")
        mocks["search"].assert_called_once()
        assert "retrieved_example_indices" in assessment.metadata
        assert assessment.metadata["retrieved_example_indices"] == [0, 2]


def test_memory_augmented_judge_properties(sample_judge, sample_traces):
    with mock_apis(guidelines=["Guideline 1"]):
        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces[:1])

        assert aligned_judge.name == sample_judge.name
        assert aligned_judge.model == sample_judge.model
        assert aligned_judge.get_input_fields() == sample_judge.get_input_fields()

        assert sample_judge.instructions in aligned_judge.instructions
        assert "Distilled Guidelines" in aligned_judge.instructions
        assert "Guideline 1" in aligned_judge.instructions


def test_incremental_alignment_preserves_examples(sample_judge, sample_traces):
    with mock_apis(guidelines=["Guideline 1"]):
        optimizer = MemAlignOptimizer()

        # First alignment with 2 traces
        judge_v2 = optimizer.align(sample_judge, sample_traces[:2])
        assert len(judge_v2._examples) == 2
        assert judge_v2._base_judge is sample_judge

        # Second alignment with 2 more traces - should preserve previous examples
        judge_v3 = optimizer.align(judge_v2, sample_traces[2:4])
        assert len(judge_v3._examples) == 4
        assert judge_v3._base_judge is sample_judge  # Should unwrap to original

        # Verify all trace IDs are present
        trace_ids_in_v3 = {ex._trace_id for ex in judge_v3._examples if hasattr(ex, "_trace_id")}
        expected_trace_ids = {sample_traces[i].info.trace_id for i in range(4)}
        assert trace_ids_in_v3 == expected_trace_ids


def test_incremental_alignment_redistills_guidelines(sample_judge, sample_traces):
    # First alignment: distills "Guideline A"
    with mock_apis(guidelines=["Guideline A"]):
        optimizer = MemAlignOptimizer()
        judge_v2 = optimizer.align(sample_judge, sample_traces[:2])
        assert len(judge_v2._semantic_memory) == 1
        assert "Guideline A" in judge_v2._semantic_memory

    # Second alignment: distills "Guideline B" from ALL examples (old + new)
    with mock_apis(guidelines=["Guideline B"]):
        judge_v3 = optimizer.align(judge_v2, sample_traces[2:4])
        # Should have both old + new guidelines (re-distilled from all examples)
        assert len(judge_v3._semantic_memory) == 2
        assert "Guideline A" in judge_v3._semantic_memory
        assert "Guideline B" in judge_v3._semantic_memory


def test_unalign_redistills_guidelines(sample_judge, sample_traces):
    # First alignment: distills "Guideline 1" and "Guideline 2"
    with mock_apis(guidelines=["Guideline 1", "Guideline 2"]):
        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces)
        assert len(aligned_judge._semantic_memory) == 2

    # Unalign some traces and redistill - should get "Guideline 3" from remaining examples
    with mock_apis(guidelines=["Guideline 3"]):
        traces_to_remove = [sample_traces[1], sample_traces[3]]
        unaligned_judge = aligned_judge.unalign(traces=traces_to_remove)
        # Should only have "Guideline 3" (redistilled from remaining examples)
        assert len(unaligned_judge._semantic_memory) == 1
        assert "Guideline 3" in unaligned_judge._semantic_memory
