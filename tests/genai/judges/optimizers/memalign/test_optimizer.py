import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import mlflow
from mlflow.entities.assessment import Assessment, AssessmentSource, Feedback
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.instructions_judge import InstructionsJudge
from mlflow.genai.judges.optimizers import MemAlignOptimizer
from mlflow.genai.judges.optimizers.memalign.optimizer import (
    _DATABRICKS_EMBEDDING_BATCH_SIZE,
    _DEFAULT_EMBEDDING_BATCH_SIZE,
    MemoryAugmentedJudge,
    _build_embedder,
)
from mlflow.genai.judges.optimizers.memalign.prompts import (
    EXAMPLES_SECTION_HEADER,
    GUIDELINES_SECTION_HEADER,
)
from mlflow.genai.scorers import get_scorer
from mlflow.genai.scorers.base import Scorer, ScorerKind, SerializedScorer
from mlflow.tracking.fluent import _get_experiment_id

_HUMAN_SOURCE = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user1")


def _start_test_trace(span_name: str, inputs: str = "input", outputs: str = "output") -> str:
    with mlflow.start_span(name=span_name) as span:
        span.set_inputs({"inputs": inputs})
        span.set_outputs({"outputs": outputs})
    return mlflow.get_last_active_trace_id()


def _log_human_feedback(trace_id: str, value: str, rationale: str = "") -> Assessment:
    return mlflow.log_feedback(
        trace_id=trace_id,
        name="test_judge",
        value=value,
        rationale=rationale,
        source=_HUMAN_SOURCE,
    )


def _update_human_feedback(
    trace_id: str, assessment_id: str, value: str, rationale: str = ""
) -> Assessment:
    return mlflow.update_assessment(
        trace_id=trace_id,
        assessment_id=assessment_id,
        assessment=Feedback(
            name="test_judge", value=value, rationale=rationale, source=_HUMAN_SOURCE
        ),
    )


def _refresh(trace: Trace) -> Trace:
    return mlflow.get_trace(trace.info.trace_id)


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
def mock_apis(guidelines=None, batch_size=50):
    """Context manager for mocking API calls with optional guideline configuration."""
    if guidelines is None:
        guidelines = []

    # _create_batches returns list of batches; mock returns single batch with all indices
    # based on actual input size
    def create_batches_side_effect(examples_data, indices, **kwargs):
        # Return single batch containing all indices
        return [list(indices)]

    with (
        patch("dspy.retrievers.Embeddings") as mock_embeddings_class,
        patch("dspy.Embedder") as mock_embedder_class,
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils.construct_dspy_lm"
        ) as mock_construct_lm,
        patch(
            "mlflow.genai.judges.optimizers.memalign.utils._create_batches",
            side_effect=create_batches_side_effect,
        ) as mock_create_batches,
    ):
        # Mock distillation LM - include source_trace_ids for guidelines to be retained
        mock_lm = MagicMock()
        guidelines_json = {
            "guidelines": [
                {"guideline_text": g, "source_trace_ids": list(range(batch_size))}
                for g in guidelines
            ]
        }
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
            "create_batches": mock_create_batches,
        }


@pytest.fixture
def sample_traces():
    traces = []
    for i in range(5):
        trace_id = _start_test_trace(f"test_span_{i}", f"input_{i}", f"output_{i}")
        _log_human_feedback(trace_id, value="yes", rationale=f"Reason {i}")
        traces.append(mlflow.get_trace(trace_id))
    return traces


def test_init_default_config():
    optimizer = MemAlignOptimizer()
    assert optimizer._retrieval_k == 5
    assert optimizer._embedding_model == "openai:/text-embedding-3-small"
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

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

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
        assert len(aligned_judge._episodic_memory) == 3
        assert len(aligned_judge._semantic_memory) == 2


def test_unalign_removes_traces(sample_judge, sample_traces):
    with mock_apis(guidelines=["Guideline 1"]):
        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces)

        # Verify all examples are present
        num_examples = len(aligned_judge._episodic_memory)
        assert num_examples == len(sample_traces)

        traces_to_remove = [sample_traces[1], sample_traces[3]]
        unaligned_judge = aligned_judge.unalign(traces=traces_to_remove)

        # Verify examples for traces 1 and 3 are removed
        assert len(unaligned_judge._episodic_memory) == num_examples - 2
        remaining_trace_ids = {
            ex._trace_id for ex in unaligned_judge._episodic_memory if hasattr(ex, "_trace_id")
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
        assert len(unaligned_judge._episodic_memory) == 3


def test_judge_call_uses_semantic_memory(sample_judge, sample_traces):
    with mock_apis(guidelines=["Be concise", "Be clear"]):
        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces[:1])

        assert len(aligned_judge._semantic_memory) == 2
        guideline_texts = [g.guideline_text for g in aligned_judge._semantic_memory]
        assert "Be concise" in guideline_texts
        assert "Be clear" in guideline_texts


def test_judge_call_retrieves_relevant_examples(sample_judge, sample_traces):
    with mock_apis(guidelines=[]) as mocks:
        # Configure search to return specific indices
        search_results = MagicMock()
        search_results.indices = [0, 2]
        mocks["search"].return_value = search_results

        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces[:3])

        mock_feedback = Feedback(
            name=sample_judge.name,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=sample_judge.model,
            ),
            value="yes",
            rationale="Test rationale",
        )
        # Patch on the class: __call__ scores through a per-call copy of the scoring
        # judge, so an instance-level mock would not be the object that gets invoked.
        with patch.object(
            InstructionsJudge, "__call__", return_value=mock_feedback
        ) as mock_scoring_call:
            assessment = aligned_judge(inputs="test input", outputs="test output")

        mock_scoring_call.assert_called_once()
        mocks["search"].assert_called_once()
        assert "retrieved_example_trace_ids" in assessment.metadata
        # Should return trace IDs, not indices
        retrieved_trace_ids = assessment.metadata["retrieved_example_trace_ids"]
        assert len(retrieved_trace_ids) == 2
        # Verify they're actual trace IDs from the sample traces
        expected_trace_ids = [sample_traces[0].info.trace_id, sample_traces[2].info.trace_id]
        assert retrieved_trace_ids == expected_trace_ids


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

        judge_v2 = optimizer.align(sample_judge, sample_traces[:2])
        assert len(judge_v2._episodic_memory) == 2
        assert judge_v2._base_judge is sample_judge

        judge_v3 = optimizer.align(judge_v2, sample_traces[2:4])
        assert len(judge_v3._episodic_memory) == 4
        assert judge_v3._base_judge is sample_judge

        trace_ids_in_v3 = {
            ex._trace_id for ex in judge_v3._episodic_memory if hasattr(ex, "_trace_id")
        }
        expected_trace_ids = {sample_traces[i].info.trace_id for i in range(4)}
        assert trace_ids_in_v3 == expected_trace_ids


def test_incremental_alignment_preserves_trace_ids(sample_judge, sample_traces):
    with mock_apis(guidelines=["Guideline 1"]):
        optimizer = MemAlignOptimizer()

        judge_v2 = optimizer.align(sample_judge, sample_traces[:2])
        batch1_ids = {t.info.trace_id for t in sample_traces[:2]}
        assert set(judge_v2._episodic_trace_ids) == batch1_ids

        judge_v3 = optimizer.align(judge_v2, sample_traces[2:4])
        all_ids = batch1_ids | {t.info.trace_id for t in sample_traces[2:4]}
        assert set(judge_v3._episodic_trace_ids) == all_ids


def test_incremental_alignment_with_single_example(sample_judge, sample_traces):
    with mock_apis(guidelines=[]):
        optimizer = MemAlignOptimizer()

        judge_v2 = optimizer.align(sample_judge, sample_traces[:1])
        assert len(judge_v2._episodic_memory) == 1

        judge_v3 = optimizer.align(judge_v2, sample_traces[1:3])
        assert len(judge_v3._episodic_memory) == 3


def test_incremental_alignment_after_deserialization(sample_judge, sample_traces):
    with mock_apis(guidelines=["Guideline 1"]):
        optimizer = MemAlignOptimizer()

        aligned_v1 = optimizer.align(sample_judge, sample_traces[:3])
        assert len(aligned_v1._episodic_memory) == 3

        dumped = aligned_v1.model_dump()
        serialized = SerializedScorer(**dumped)
        deserialized = MemoryAugmentedJudge._from_serialized(serialized)

        assert deserialized._episodic_memory == []
        assert len(deserialized._episodic_trace_ids) == 3

        trace_map = {t.info.trace_id: t for t in sample_traces[:3]}
        with patch(
            "mlflow.genai.judges.optimizers.memalign.optimizer.mlflow.get_trace",
            side_effect=lambda tid, **kwargs: trace_map.get(tid),
        ):
            aligned_v2 = optimizer.align(deserialized, sample_traces[3:5])

        assert len(aligned_v2._episodic_memory) == 5


def test_incremental_alignment_redistills_guidelines(sample_judge, sample_traces):
    # First alignment: distills "Guideline A"
    with mock_apis(guidelines=["Guideline A"]):
        optimizer = MemAlignOptimizer()
        judge_v2 = optimizer.align(sample_judge, sample_traces[:2])
        assert len(judge_v2._semantic_memory) == 1
        guideline_texts = [g.guideline_text for g in judge_v2._semantic_memory]
        assert "Guideline A" in guideline_texts

    # Second alignment: distills "Guideline B" from ALL examples (old + new)
    with mock_apis(guidelines=["Guideline B"]):
        judge_v3 = optimizer.align(judge_v2, sample_traces[2:4])
        # Should have both old + new guidelines (re-distilled from all examples)
        assert len(judge_v3._semantic_memory) == 2
        guideline_texts = [g.guideline_text for g in judge_v3._semantic_memory]
        assert "Guideline A" in guideline_texts
        assert "Guideline B" in guideline_texts


def test_unalign_filters_guidelines_by_source_ids(sample_judge, sample_traces):
    # Test that unalign() filters guidelines based on source_ids
    with mock_apis(guidelines=["Guideline 1", "Guideline 2"]):
        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces)
        assert len(aligned_judge._semantic_memory) == 2

        # Unalign some traces - should filter guidelines based on source_trace_ids
        traces_to_remove = [sample_traces[1], sample_traces[3]]
        unaligned_judge = aligned_judge.unalign(traces=traces_to_remove)
        # Unalign doesn't redistill, it filters guidelines based on source_trace_ids
        # Guidelines without source_trace_ids are retained
        # Guidelines are deleted only if ALL source traces were removed
        # Since mock_apis doesn't provide source_trace_ids, all guidelines are retained
        assert len(unaligned_judge._episodic_memory) == 3  # 5 - 2 removed
        assert len(unaligned_judge._semantic_memory) == 2


# =============================================================================
# Serialization Tests
# =============================================================================


def test_memory_augmented_judge_kind_property(sample_judge, sample_traces):
    with mock_apis(guidelines=[]):
        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces[:1])

        assert aligned_judge.kind == ScorerKind.MEMORY_AUGMENTED


def test_memory_augmented_judge_model_dump(sample_judge, sample_traces):
    with mock_apis(guidelines=["Guideline A", "Guideline B"]):
        optimizer = MemAlignOptimizer(
            reflection_lm="openai:/gpt-4o-mini",
            retrieval_k=3,
            embedding_model="openai:/text-embedding-3-small",
            embedding_dim=256,
        )
        aligned_judge = optimizer.align(sample_judge, sample_traces[:3])

        dumped = aligned_judge.model_dump()

        # Verify top-level structure
        assert "memory_augmented_judge_data" in dumped
        assert dumped["name"] == sample_judge.name

        data = dumped["memory_augmented_judge_data"]
        assert "base_judge" in data
        assert "episodic_trace_ids" in data
        assert "semantic_memory" in data

        # Verify config fields
        assert data["reflection_lm"] == "openai:/gpt-4o-mini"
        assert data["retrieval_k"] == 3
        assert data["embedding_model"] == "openai:/text-embedding-3-small"
        assert data["embedding_dim"] == 256

        # Verify episodic trace IDs are extracted
        expected_trace_ids = [t.info.trace_id for t in sample_traces[:3]]
        assert set(data["episodic_trace_ids"]) == set(expected_trace_ids)

        # Verify semantic memory is serialized
        assert len(data["semantic_memory"]) == 2
        guideline_texts = [g["guideline_text"] for g in data["semantic_memory"]]
        assert "Guideline A" in guideline_texts
        assert "Guideline B" in guideline_texts


def test_memory_augmented_judge_from_serialized(sample_judge, sample_traces):
    with mock_apis(guidelines=["Be concise", "Be accurate"]):
        optimizer = MemAlignOptimizer(
            reflection_lm="openai:/gpt-4",
            retrieval_k=7,
            embedding_model="openai:/text-embedding-3-large",
            embedding_dim=1024,
        )
        aligned_judge = optimizer.align(sample_judge, sample_traces[:2])

        dumped = aligned_judge.model_dump()
        serialized = SerializedScorer(**dumped)
        restored = MemoryAugmentedJudge._from_serialized(serialized)

        # Verify config fields are restored
        assert restored._reflection_lm == "openai:/gpt-4"
        assert restored._retrieval_k == 7
        assert restored._embedding_model == "openai:/text-embedding-3-large"
        assert restored._embedding_dim == 1024

        # Verify semantic memory is restored
        assert len(restored._semantic_memory) == 2
        guideline_texts = [g.guideline_text for g in restored._semantic_memory]
        assert "Be concise" in guideline_texts
        assert "Be accurate" in guideline_texts

        # Verify lazy initialization state (_embedder is None means deferred)
        assert restored._embedder is None
        assert restored._episodic_memory == []
        assert len(restored._episodic_trace_ids) == 2

        # Verify deferred components are None
        assert restored._base_signature is None
        assert restored._retriever is None


def test_scorer_model_validate_routes_to_memory_augmented_judge(sample_judge, sample_traces):
    with mock_apis(guidelines=[]):
        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces[:1])

        dumped = aligned_judge.model_dump()
        restored = Scorer.model_validate(dumped)

        assert isinstance(restored, MemoryAugmentedJudge)
        assert restored.name == sample_judge.name


def test_scorer_model_validate_json_routes_to_memory_augmented_judge(sample_judge, sample_traces):
    with mock_apis(guidelines=[]):
        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces[:1])

        dumped = aligned_judge.model_dump()
        restored = Scorer.model_validate_json(json.dumps(dumped))

        assert isinstance(restored, MemoryAugmentedJudge)
        assert restored.name == sample_judge.name


def test_memory_augmented_judge_round_trip_serialization(sample_judge, sample_traces):
    with mock_apis(guidelines=["Test guideline"]):
        optimizer = MemAlignOptimizer(
            reflection_lm="openai:/gpt-4o-mini",
            retrieval_k=5,
            embedding_model="openai:/text-embedding-3-small",
            embedding_dim=512,
        )
        original_judge = optimizer.align(sample_judge, sample_traces[:3])

        dumped = original_judge.model_dump()
        serialized = SerializedScorer(**dumped)
        restored_judge = MemoryAugmentedJudge._from_serialized(serialized)

        # Verify config matches
        assert restored_judge.name == original_judge.name
        assert restored_judge._reflection_lm == original_judge._reflection_lm
        assert restored_judge._retrieval_k == original_judge._retrieval_k
        assert restored_judge._embedding_model == original_judge._embedding_model
        assert restored_judge._embedding_dim == original_judge._embedding_dim

        # Verify semantic memory matches
        original_guidelines = [g.guideline_text for g in original_judge._semantic_memory]
        restored_guidelines = [g.guideline_text for g in restored_judge._semantic_memory]
        assert original_guidelines == restored_guidelines

        # Verify episodic trace IDs match
        assert set(restored_judge._episodic_trace_ids) == set(original_judge._episodic_trace_ids)


def test_memory_augmented_judge_lazy_init_triggered_on_call(sample_judge, sample_traces):
    with mock_apis(guidelines=[]):
        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces[:2])

        dumped = aligned_judge.model_dump()
        serialized = SerializedScorer(**dumped)
        restored = MemoryAugmentedJudge._from_serialized(serialized)

        # Verify deferred state (_embedder is None means not initialized)
        assert restored._embedder is None

        # Mock mlflow.get_trace and predict module for the call
        trace_map = {t.info.trace_id: t for t in sample_traces[:2]}

        mock_feedback = Feedback(
            name=sample_judge.name,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=sample_judge.model,
            ),
            value="yes",
            rationale="Test",
        )

        with (
            patch(
                "mlflow.genai.judges.optimizers.memalign.optimizer.mlflow.get_trace",
                side_effect=lambda tid, **kwargs: trace_map.get(tid),
            ) as mock_get_trace,
            patch("dspy.Embedder") as mock_embedder_class,
            patch("dspy.retrievers.Embeddings"),
        ):
            mock_embedder_class.return_value = MagicMock()

            # Trigger lazy init, then mock the scoring judge before call completes
            restored._lazy_init()
            assert restored._embedder is not None
            assert mock_get_trace.call_count == 2

            # Patch on the class: __call__ copies the base judge per call, so an
            # instance-level mock would not be the object that gets invoked.
            with patch.object(
                InstructionsJudge, "__call__", return_value=mock_feedback
            ) as mock_scoring_call:
                restored(inputs="test", outputs="test")
                mock_scoring_call.assert_called_once()


def test_memory_augmented_judge_lazy_init_logs_warning_for_missing_traces(
    sample_judge, sample_traces
):
    with mock_apis(guidelines=[]):
        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces[:3])

        dumped = aligned_judge.model_dump()
        serialized = SerializedScorer(**dumped)
        restored = MemoryAugmentedJudge._from_serialized(serialized)

        # Mock get_trace to return only 1 of 3 traces (simulating missing traces)
        first_trace = sample_traces[0]

        def mock_get_trace_fn(tid, **kwargs):
            if tid == first_trace.info.trace_id:
                return first_trace
            return None

        mock_feedback = Feedback(
            name=sample_judge.name,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=sample_judge.model,
            ),
            value="yes",
            rationale="Test",
        )

        with (
            patch(
                "mlflow.genai.judges.optimizers.memalign.optimizer.mlflow.get_trace",
                side_effect=mock_get_trace_fn,
            ),
            patch("dspy.Embedder"),
            patch("dspy.retrievers.Embeddings"),
            patch("mlflow.genai.judges.optimizers.memalign.optimizer._logger") as mock_logger,
            patch.object(
                InstructionsJudge, "__call__", return_value=mock_feedback
            ) as mock_scoring_call,
        ):
            restored._lazy_init()
            restored(inputs="test", outputs="test")

            mock_scoring_call.assert_called_once()
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "Could not find 2 traces" in warning_msg
            assert "Judge will operate with partial memory" in warning_msg


def test_memory_augmented_judge_create_copy_preserves_trace_ids(sample_judge, sample_traces):
    with mock_apis(guidelines=["Test guideline"]):
        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces[:3])

        assert len(aligned_judge._episodic_trace_ids) == 3

        judge_copy = aligned_judge._create_copy()

        # Copy should have trace IDs and be in deferred state
        assert judge_copy._embedder is None
        assert judge_copy._episodic_memory == []
        assert set(judge_copy._episodic_trace_ids) == set(aligned_judge._episodic_trace_ids)


def test_judge_call_delegates_to_base_judge_copy(sample_judge, sample_traces):
    with mock_apis(guidelines=["Be concise"]) as mocks:
        mocks["search"].return_value = MagicMock(indices=[0])

        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces[:1])

        mock_feedback = Feedback(
            name=sample_judge.name,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=sample_judge.model,
            ),
            value="yes",
            rationale="Test rationale",
        )
        invoked_instructions = []

        def capture(self, **kwargs):
            invoked_instructions.append(self._instructions)
            return mock_feedback

        with patch.object(InstructionsJudge, "__call__", capture):
            aligned_judge(inputs="test input", outputs="test output")

        # The judge that was invoked carries instructions augmented with the guidelines.
        assert len(invoked_instructions) == 1
        assert "Be concise" in invoked_instructions[0]
        # The shared base judge is left untouched, so context cannot bleed across calls.
        assert "Be concise" not in aligned_judge._base_judge._instructions


def test_memory_augmented_judge_extracts_inputs_outputs_from_trace(sample_judge, sample_traces):
    with mock_apis(guidelines=[]) as mocks:
        mocks["search"].return_value = MagicMock(indices=[])

        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces[:1])

        mock_feedback = Feedback(
            name=sample_judge.name,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=sample_judge.model,
            ),
            value="yes",
            rationale="Test rationale",
        )
        # Call with only trace - inputs/outputs should be extracted from trace
        test_trace = sample_traces[0]
        with patch.object(
            InstructionsJudge, "__call__", return_value=mock_feedback
        ) as mock_scoring_call:
            aligned_judge(trace=test_trace)

        # Verify scoring judge was called with extracted inputs/outputs and the trace
        call_kwargs = mock_scoring_call.call_args.kwargs
        assert call_kwargs["inputs"] == {"inputs": "input_0"}
        assert call_kwargs["outputs"] == {"outputs": "output_0"}
        assert call_kwargs["trace"] is test_trace


# =============================================================================
# Trace-based (agentic) scoring preservation tests
#
# An aligned judge must keep the base judge's invocation mode. For a
# {{ trace }} judge that means the Trace object still reaches
# invoke_judge_model() so the judge tools can inspect child spans; the old
# implementation flattened the trace to root-span text and scored through
# dspy.Predict, so anything below the root span was invisible at scoring time.
# =============================================================================


@pytest.fixture
def trace_judge():
    return make_judge(
        name="test_judge",
        instructions="Evaluate whether {{ trace }} used the calculator tool correctly",
        model="openai:/gpt-4",
    )


def _start_trace_with_tool_span(root_name: str, tool_output: str) -> str:
    with mlflow.start_span(name=root_name) as root:
        root.set_inputs({"question": "what is 2+2?"})
        with mlflow.start_span(name="calculator", span_type="TOOL") as tool:
            tool.set_inputs({"expression": "2+2"})
            tool.set_outputs({"result": tool_output})
        root.set_outputs({"answer": "the answer is 4"})
    return mlflow.get_last_active_trace_id()


@pytest.fixture
def tool_traces():
    traces = []
    for i in range(2):
        trace_id = _start_trace_with_tool_span(f"agent_{i}", tool_output="4")
        _log_human_feedback(trace_id, value="yes", rationale=f"Tool used correctly {i}")
        traces.append(mlflow.get_trace(trace_id))
    return traces


def test_aligned_trace_judge_passes_trace_to_invoke_judge_model(trace_judge, tool_traces):
    # The Trace object itself must reach invoke_judge_model so that the judge tools can
    # read child spans. The pre-fix implementation scored via dspy.Predict and passed only
    # value_to_embedding_text(trace) (root request/response), so trace=None reached the
    # model layer and child TOOL spans were unreachable.
    with mock_apis(guidelines=["Check the tool result"]) as mocks:
        mocks["search"].return_value = MagicMock(indices=[0])

        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(trace_judge, tool_traces)

        target_trace = tool_traces[0]
        feedback = Feedback(
            name=trace_judge.name,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id=trace_judge.model
            ),
            value="yes",
            rationale="Tool call verified",
        )
        with patch(
            "mlflow.genai.judges.instructions_judge.invoke_judge_model",
            return_value=feedback,
        ) as mock_invoke:
            aligned_judge(trace=target_trace)

        mock_invoke.assert_called_once()
        assert mock_invoke.call_args.kwargs["trace"] is target_trace


def test_aligned_trace_judge_keeps_agentic_system_prompt(trace_judge, tool_traces):
    # Trace-based judges get the agentic system prompt that instructs the model to inspect
    # the trace via tools. Delegating to a copy of the base judge preserves it; scoring
    # through dspy.Predict did not produce this prompt at all.
    with mock_apis(guidelines=["Check the tool result"]) as mocks:
        mocks["search"].return_value = MagicMock(indices=[0])

        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(trace_judge, tool_traces)

        feedback = Feedback(
            name=trace_judge.name,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id=trace_judge.model
            ),
            value="yes",
            rationale="Tool call verified",
        )
        with patch(
            "mlflow.genai.judges.instructions_judge.invoke_judge_model",
            return_value=feedback,
        ) as mock_invoke:
            aligned_judge(trace=tool_traces[0])

        system_message = mock_invoke.call_args.kwargs["prompt"][0].content
        assert "To read the actual trace, you will need to use the" in system_message
        # Memory context rides along in the same system prompt.
        assert "Check the tool result" in system_message
        assert "Example Judgements" in system_message


def test_aligned_judge_preserves_feedback_value_type(sample_traces):
    # A non-str feedback_value_type is enforced by the base judge's response_format. The
    # pre-fix path built its own DSPy signature, so the aligned judge could not be relied
    # on to honor the base judge's declared output type.
    bool_judge = make_judge(
        name="test_judge",
        instructions="Is {{ outputs }} a correct answer to {{ inputs }}?",
        model="openai:/gpt-4",
        feedback_value_type=bool,
    )

    with mock_apis(guidelines=[]) as mocks:
        mocks["search"].return_value = MagicMock(indices=[])

        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(bool_judge, sample_traces[:1])

        feedback = Feedback(
            name=bool_judge.name,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id=bool_judge.model
            ),
            value=True,
            rationale="Correct",
        )
        with patch(
            "mlflow.genai.judges.instructions_judge.invoke_judge_model",
            return_value=feedback,
        ) as mock_invoke:
            result = aligned_judge(inputs="test input", outputs="test output")

        response_format = mock_invoke.call_args.kwargs["response_format"]
        assert response_format.model_fields["result"].annotation is bool
        assert result.value is True


def test_aligned_judge_does_not_mutate_base_judge_instructions(sample_judge, sample_traces):
    with mock_apis(guidelines=["Be concise"]) as mocks:
        mocks["search"].return_value = MagicMock(indices=[0])

        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces[:1])

        original_instructions = sample_judge.instructions

        feedback = Feedback(
            name=sample_judge.name,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id=sample_judge.model
            ),
            value="yes",
            rationale="Test",
        )
        with patch(
            "mlflow.genai.judges.instructions_judge.invoke_judge_model",
            return_value=feedback,
        ) as mock_invoke:
            aligned_judge(inputs="test input", outputs="test output")

        mock_invoke.assert_called_once()
        assert sample_judge.instructions == original_instructions


def test_incremental_alignment_scores_through_unwrapped_base_judge(sample_judge, sample_traces):
    # Re-aligning passes the previous MemoryAugmentedJudge in as base_judge. The judge
    # scored through must be the unwrapped InstructionsJudge, not that wrapper: delegating
    # to the wrapper would score through its stale inner memory and drop the per-call
    # instructions the outer judge applies.
    with mock_apis(guidelines=["Guideline A"]) as mocks:
        mocks["search"].return_value = MagicMock(indices=[0])

        optimizer = MemAlignOptimizer()
        judge_v2 = optimizer.align(sample_judge, sample_traces[:2])
        judge_v3 = optimizer.align(judge_v2, sample_traces[2:4])

        assert isinstance(judge_v3._base_judge, InstructionsJudge)
        assert not isinstance(judge_v3._base_judge, MemoryAugmentedJudge)

        # The re-aligned judge scores with its combined memory, applied once.
        invoked_instructions = []
        feedback = Feedback(
            name=sample_judge.name,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id=sample_judge.model
            ),
            value="yes",
            rationale="Test",
        )

        def capture(self, **kwargs):
            invoked_instructions.append(self._instructions)
            return feedback

        with patch.object(InstructionsJudge, "__call__", capture):
            judge_v3(inputs="test input", outputs="test output")

        assert len(invoked_instructions) == 1
        assert "Guideline A" in invoked_instructions[0]
        assert "Example Judgements" in invoked_instructions[0]


def test_augmented_instructions_preserve_dspy_section_headers(sample_judge, sample_traces):
    # The section headers are carried over verbatim from the DSPy InputField descriptions
    # that used to carry this framing, and examples precede guidelines as they did in the
    # prepended signature. Pinned so the prompt the judge sees does not drift silently.
    with mock_apis(guidelines=["Be concise"]) as mocks:
        mocks["search"].return_value = MagicMock(indices=[0])

        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces[:1])

        instructions = aligned_judge._build_augmented_instructions(
            ["Be concise"], aligned_judge._episodic_memory[:1]
        )

        assert GUIDELINES_SECTION_HEADER in instructions
        assert EXAMPLES_SECTION_HEADER in instructions
        assert instructions.startswith(sample_judge.instructions)
        # Examples precede guidelines, matching the order the DSPy fields were prepended in.
        assert instructions.index("Example Judgements (1):") < instructions.index(
            "Distilled Guidelines (1):"
        )


def test_aligned_judge_reports_base_criterion_in_guideline_metadata(sample_judge, sample_traces):
    # InstructionsJudge stamps metadata["guideline"] with the instructions it ran on, and the
    # UI renders that as a short per-assertion label (TestCaseDetail.tsx, rendererFunctions.tsx).
    # It must stay the base criterion: the augmented text embeds retrieved examples, i.e. other
    # traces' inputs/outputs, which would then be persisted onto every assessment.
    with mock_apis(guidelines=["Be concise"]) as mocks:
        mocks["search"].return_value = MagicMock(indices=[0])

        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces[:2])

        feedback = Feedback(
            name=sample_judge.name,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id=sample_judge.model
            ),
            value="yes",
            rationale="Test",
        )
        with patch(
            "mlflow.genai.judges.instructions_judge.invoke_judge_model",
            return_value=feedback,
        ) as mock_invoke:
            result = aligned_judge(inputs="test input", outputs="test output")

        # The augmented context still reaches the model.
        assert "Be concise" in mock_invoke.call_args.kwargs["prompt"][0].content

        guideline_metadata = result.metadata["guideline"]
        assert guideline_metadata == sample_judge.instructions
        assert "Example Judgements" not in guideline_metadata
        assert "input_0" not in guideline_metadata
        # Retrieval metadata is still attached alongside.
        assert result.metadata["retrieved_example_trace_ids"] == [sample_traces[0].info.trace_id]


def test_concurrent_calls_do_not_leak_retrieved_examples_across_rows(sample_judge, sample_traces):
    # The eval harness scores rows on a thread pool sharing one scorer instance
    # (mlflow/genai/evaluation/harness.py:422-454), so two rows can be inside __call__ at
    # once. Each row's prompt must carry only the examples retrieved for that row.
    #
    # The barrier sits at the start of _build_system_message, i.e. after both threads have
    # published their per-call instructions but before either reads them back — the exact
    # interleaving that a shared mutable _instructions attribute cannot survive.
    with mock_apis(guidelines=[]) as mocks:
        optimizer = MemAlignOptimizer()
        aligned_judge = optimizer.align(sample_judge, sample_traces[:3])
        assert mocks["embedder"] is not None

    # Row 0 retrieves episodic example 0 ("input_0"); row 1 retrieves example 1 ("input_1").
    index_by_row = {"row-0": [0], "row-1": [1]}
    expected_example_text = {"row-0": "input_0", "row-1": "input_1"}
    other_example_text = {"row-0": "input_1", "row-1": "input_0"}

    barrier = threading.Barrier(2)
    prompts: dict[str, str] = {}
    prompts_lock = threading.Lock()

    def retriever_side_effect(query, **kwargs):
        row = "row-0" if "row-0" in str(query) else "row-1"
        return MagicMock(indices=index_by_row[row])

    original_build_system_message = InstructionsJudge._build_system_message

    def barriered_build_system_message(self, is_trace_based):
        barrier.wait(timeout=10)
        return original_build_system_message(self, is_trace_based)

    def invoke_side_effect(**kwargs):
        user_message = kwargs["prompt"][1].content
        row = "row-0" if "row-0" in user_message else "row-1"
        with prompts_lock:
            prompts[row] = kwargs["prompt"][0].content
        return Feedback(
            name=sample_judge.name,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id=sample_judge.model
            ),
            value="yes",
            rationale="Test",
        )

    aligned_judge._retriever = MagicMock(side_effect=retriever_side_effect)

    with (
        patch(
            "mlflow.genai.judges.instructions_judge.invoke_judge_model",
            side_effect=invoke_side_effect,
        ),
        patch.object(InstructionsJudge, "_build_system_message", barriered_build_system_message),
        ThreadPoolExecutor(max_workers=2, thread_name_prefix="MemAlignRaceTest") as pool,
    ):
        futures = [
            pool.submit(aligned_judge, inputs=row, outputs="out") for row in ("row-0", "row-1")
        ]
        for future in futures:
            future.result(timeout=30)

    assert set(prompts) == {"row-0", "row-1"}
    for row, system_message in prompts.items():
        assert expected_example_text[row] in system_message
        assert other_example_text[row] not in system_message


@pytest.mark.parametrize(
    ("embedding_model", "expected_batch_size"),
    [
        ("endpoints:/databricks-bge-large-en", _DATABRICKS_EMBEDDING_BATCH_SIZE),
        ("databricks:/my-embedding-endpoint", _DATABRICKS_EMBEDDING_BATCH_SIZE),
        ("openai:/text-embedding-3-small", _DEFAULT_EMBEDDING_BATCH_SIZE),
    ],
)
def test_embedder_batch_size(sample_judge, sample_traces, embedding_model, expected_batch_size):
    with mock_apis(guidelines=[]) as mocks:
        optimizer = MemAlignOptimizer(embedding_model=embedding_model)
        optimizer.align(sample_judge, sample_traces[:1])

        _, kwargs = mocks["embedder_class"].call_args
        assert kwargs["batch_size"] == expected_batch_size


@pytest.mark.parametrize(
    ("embedding_model", "expected_api_base"),
    [
        (
            "databricks:/system.ai.gte-large-en",
            "https://my-workspace.databricks.com/ai-gateway/mlflow/v1",
        ),
        (
            "endpoints:/main.default.my_embeddings",
            "https://my-workspace.databricks.com/ai-gateway/mlflow/v1",
        ),
        (
            "databricks:/databricks-gte-large-en",
            "https://my-workspace.databricks.com/serving-endpoints",
        ),
    ],
)
def test_build_embedder_routes_databricks_models(monkeypatch, embedding_model, expected_api_base):
    monkeypatch.setenv("DATABRICKS_HOST", "https://my-workspace.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-test-token")

    with patch("dspy.Embedder") as mock_embedder_class:
        _build_embedder(embedding_model, 512)

    _, kwargs = mock_embedder_class.call_args
    assert kwargs["api_base"] == expected_api_base
    assert kwargs["api_key"] == "dapi-test-token"


def test_build_embedder_non_databricks_has_no_api_base(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://my-workspace.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-test-token")

    with patch("dspy.Embedder") as mock_embedder_class:
        _build_embedder("openai:/text-embedding-3-small", 512)

    _, kwargs = mock_embedder_class.call_args
    assert "api_base" not in kwargs
    assert "api_key" not in kwargs


# =============================================================================
# Re-alignment / deduplication tests
# =============================================================================


def test_realign_on_same_traces_does_not_duplicate_memory(sample_judge, sample_traces):
    with mock_apis(guidelines=["Guideline A"]) as mocks:
        optimizer = MemAlignOptimizer()
        judge_v1 = optimizer.align(sample_judge, sample_traces[:3])

        v1_episodic_count = len(judge_v1._episodic_memory)
        v1_trace_ids = sorted(judge_v1._episodic_trace_ids)
        v1_semantic_count = len(judge_v1._semantic_memory)
        v1_lm_calls = mocks["lm"].call_count

        judge_v2 = optimizer.align(judge_v1, sample_traces[:3])

        assert len(judge_v2._episodic_memory) == v1_episodic_count
        assert sorted(judge_v2._episodic_trace_ids) == v1_trace_ids
        assert len(judge_v2._semantic_memory) == v1_semantic_count
        # No additional reflection-LM calls — the all-skipped path short-circuits.
        assert mocks["lm"].call_count == v1_lm_calls


def test_realign_replaces_changed_trace_content(sample_judge, sample_traces):
    with mock_apis(guidelines=[]) as mocks:
        optimizer = MemAlignOptimizer()
        judge_v1 = optimizer.align(sample_judge, sample_traces[:3])
        v1_lm_calls = mocks["lm"].call_count

        target = sample_traces[0]
        [original] = target.info.assessments
        _update_human_feedback(
            trace_id=target.info.trace_id,
            assessment_id=original.assessment_id,
            value="no",
            rationale="Updated reason",
        )
        target = _refresh(target)

        judge_v2 = optimizer.align(judge_v1, [target, sample_traces[1], sample_traces[2]])

        assert len(judge_v2._episodic_memory) == 3
        examples_for_t0 = [
            ex
            for ex in judge_v2._episodic_memory
            if getattr(ex, "_trace_id", None) == target.info.trace_id
        ]
        assert len(examples_for_t0) == 1
        assert examples_for_t0[0].result == "no"
        assert examples_for_t0[0].rationale == "Updated reason"
        # One additional reflection-LM call: the refreshed example was re-distilled.
        assert mocks["lm"].call_count == v1_lm_calls + 1


def test_realign_updates_majority_resolution(sample_judge):
    trace_id = _start_test_trace("majority_trace")
    _log_human_feedback(trace_id, value="yes", rationale="ra")
    _log_human_feedback(trace_id, value="yes", rationale="rb")
    feedback_c = _log_human_feedback(trace_id, value="yes", rationale="rc")
    feedback_d = _log_human_feedback(trace_id, value="no", rationale="rd")
    _log_human_feedback(trace_id, value="no", rationale="re")
    trace = mlflow.get_trace(trace_id)

    with mock_apis(guidelines=[]):
        optimizer = MemAlignOptimizer()
        judge_v1 = optimizer.align(sample_judge, [trace])

        assert len(judge_v1._episodic_memory) == 3
        assert sorted(ex.rationale for ex in judge_v1._episodic_memory) == ["ra", "rb", "rc"]
        for ex in judge_v1._episodic_memory:
            assert ex.result == "yes"

        # Edit: c flips to "no", d flips to "yes". Majority is still "yes" (a, b, d).
        _update_human_feedback(
            trace_id=trace_id,
            assessment_id=feedback_c.assessment_id,
            value="no",
            rationale="rc-updated",
        )
        _update_human_feedback(
            trace_id=trace_id,
            assessment_id=feedback_d.assessment_id,
            value="yes",
            rationale="rd-updated",
        )
        trace = mlflow.get_trace(trace_id)

        judge_v2 = optimizer.align(judge_v1, [trace])

        assert len(judge_v2._episodic_memory) == 3
        for ex in judge_v2._episodic_memory:
            assert ex.result == "yes"
        assert sorted(ex.rationale for ex in judge_v2._episodic_memory) == [
            "ra",
            "rb",
            "rd-updated",
        ]


def test_realign_after_assessment_removal(sample_judge):
    trace_id = _start_test_trace("removal_trace")
    _log_human_feedback(trace_id, value="yes", rationale="ra")
    _log_human_feedback(trace_id, value="yes", rationale="rb")
    feedback_c = _log_human_feedback(trace_id, value="yes", rationale="rc")
    trace = mlflow.get_trace(trace_id)

    with mock_apis(guidelines=[]):
        optimizer = MemAlignOptimizer()
        judge_v1 = optimizer.align(sample_judge, [trace])
        assert len(judge_v1._episodic_memory) == 3

        mlflow.delete_assessment(trace_id=trace_id, assessment_id=feedback_c.assessment_id)
        trace = mlflow.get_trace(trace_id)

        judge_v2 = optimizer.align(judge_v1, [trace])

        assert len(judge_v2._episodic_memory) == 2
        assert sorted(ex.rationale for ex in judge_v2._episodic_memory) == ["ra", "rb"]


def test_realign_after_deserialization(sample_judge, sample_traces):
    with mock_apis(guidelines=["Guideline A"]):
        optimizer = MemAlignOptimizer()
        judge_v1 = optimizer.align(sample_judge, sample_traces[:3])

        dumped = judge_v1.model_dump()
        serialized = SerializedScorer(**dumped)
        deserialized = MemoryAugmentedJudge._from_serialized(serialized)

        assert deserialized._episodic_memory == []
        assert set(deserialized._episodic_trace_ids) == {t.info.trace_id for t in sample_traces[:3]}

        # Re-align with 3 overlapping + 1 new trace; mock get_trace for reconstruction
        trace_map = {t.info.trace_id: t for t in sample_traces[:4]}
        with patch(
            "mlflow.genai.judges.optimizers.memalign.optimizer.mlflow.get_trace",
            side_effect=lambda tid, **kwargs: trace_map.get(tid),
        ):
            judge_v2 = optimizer.align(deserialized, sample_traces[:4])

        assert len(judge_v2._episodic_memory) == 4
        assert set(judge_v2._episodic_trace_ids) == {t.info.trace_id for t in sample_traces[:4]}


def test_realign_mixed_unchanged_changed_and_new(sample_judge, sample_traces):
    with mock_apis(guidelines=[]):
        optimizer = MemAlignOptimizer()
        judge_v1 = optimizer.align(sample_judge, sample_traces[:3])

        t1 = sample_traces[1]
        [original] = t1.info.assessments
        _update_human_feedback(
            trace_id=t1.info.trace_id,
            assessment_id=original.assessment_id,
            value="no",
            rationale="t1 updated",
        )
        t1_updated = _refresh(t1)

        judge_v2 = optimizer.align(judge_v1, [sample_traces[0], t1_updated, sample_traces[3]])

        # Final memory: t0 (unchanged), t1 (updated), t2 (preserved — not in this call), t3 (new)
        assert len(judge_v2._episodic_memory) == 4
        assert set(judge_v2._episodic_trace_ids) == {
            sample_traces[i].info.trace_id for i in [0, 1, 2, 3]
        }

        examples_for_t1 = [
            ex
            for ex in judge_v2._episodic_memory
            if getattr(ex, "_trace_id", None) == t1_updated.info.trace_id
        ]
        assert len(examples_for_t1) == 1
        assert examples_for_t1[0].result == "no"
        assert examples_for_t1[0].rationale == "t1 updated"


def test_realign_after_unalign_roundtrip(sample_judge, sample_traces):
    with mock_apis(guidelines=["Guideline A"]):
        optimizer = MemAlignOptimizer()
        judge_v1 = optimizer.align(sample_judge, sample_traces[:3])
        v1_trace_ids = set(judge_v1._episodic_trace_ids)
        v1_episodic_count = len(judge_v1._episodic_memory)

        unaligned = judge_v1.unalign(traces=sample_traces[:3])
        assert unaligned._episodic_memory == []
        assert unaligned._episodic_trace_ids == []

        judge_v3 = optimizer.align(unaligned, sample_traces[:3])

        assert set(judge_v3._episodic_trace_ids) == v1_trace_ids
        assert len(judge_v3._episodic_memory) == v1_episodic_count


# =============================================================================
# End-to-end lifecycle tests
#
# These drive the sequences a user actually performs and score at the end. The
# older round-trip tests stop at state assertions, which no longer implies the
# judge scores correctly now that scoring runs through a per-call copy of the
# base judge.
# =============================================================================


def _mock_scoring_call(judge_name: str, model: str, value="yes"):
    """Patch InstructionsJudge.__call__ and capture the instructions it ran on."""
    feedback = Feedback(
        name=judge_name,
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id=model),
        value=value,
        rationale="Test rationale",
    )
    captured: list[str] = []

    def capture(self, **kwargs):
        captured.append(self._instructions)
        return feedback

    return patch.object(InstructionsJudge, "__call__", capture), captured


def test_unalign_then_realign_then_score(sample_judge, sample_traces):
    with mock_apis(guidelines=["Guideline A"]) as mocks:
        mocks["search"].return_value = MagicMock(indices=[0])
        optimizer = MemAlignOptimizer()

        judge_v1 = optimizer.align(sample_judge, sample_traces[:3])
        unaligned = judge_v1.unalign(traces=sample_traces[:3])
        assert unaligned._episodic_memory == []

        judge_v3 = optimizer.align(unaligned, sample_traces[:3])
        assert len(judge_v3._episodic_memory) == 3

        patcher, captured = _mock_scoring_call(sample_judge.name, sample_judge.model)
        with patcher:
            result = judge_v3(inputs="test input", outputs="test output")

        # Memory rebuilt by the re-align reaches the prompt, and scoring still works.
        assert len(captured) == 1
        assert "Guideline A" in captured[0]
        assert "Example Judgements" in captured[0]
        assert result.value == "yes"


def test_register_then_load_then_score(sample_judge, sample_traces):
    with mock_apis(guidelines=["Guideline A"]) as mocks:
        mocks["search"].return_value = MagicMock(indices=[0])
        optimizer = MemAlignOptimizer()
        aligned = optimizer.align(sample_judge, sample_traces[:3])

        aligned.register()
        loaded = get_scorer(name=sample_judge.name, experiment_id=_get_experiment_id())
        assert isinstance(loaded, MemoryAugmentedJudge)
        # Episodic memory is reconstructed lazily from the persisted trace IDs.
        assert loaded._embedder is None
        assert len(loaded._episodic_trace_ids) == 3

        patcher, captured = _mock_scoring_call(sample_judge.name, sample_judge.model)
        with patcher:
            result = loaded(inputs="test input", outputs="test output")

        # Semantic memory survived the round trip and episodic memory was rebuilt from
        # the persisted trace IDs, so both reach the prompt the loaded judge scores with.
        assert len(captured) == 1
        assert "Guideline A" in captured[0]
        assert "Example Judgements" in captured[0]
        assert result.value == "yes"
        assert result.metadata["retrieved_example_trace_ids"] == [sample_traces[0].info.trace_id]


def test_register_then_load_then_unalign_then_score(sample_judge, sample_traces):
    # unalign() must reconstruct episodic memory before filtering. A freshly loaded judge
    # holds only trace IDs, so reading _episodic_memory directly would find nothing to
    # remove and silently hand back an unchanged judge.
    with mock_apis(guidelines=["Guideline A"]) as mocks:
        mocks["search"].return_value = MagicMock(indices=[0])
        optimizer = MemAlignOptimizer()
        aligned = optimizer.align(sample_judge, sample_traces[:3])

        aligned.register()
        loaded = get_scorer(name=sample_judge.name, experiment_id=_get_experiment_id())
        assert loaded._episodic_memory == []

        updated = loaded.unalign(traces=[sample_traces[0]])

        assert updated is not loaded
        assert set(updated._episodic_trace_ids) == {
            sample_traces[1].info.trace_id,
            sample_traces[2].info.trace_id,
        }
        assert len(updated._episodic_memory) == 2
        assert all(
            ex._trace_id != sample_traces[0].info.trace_id for ex in updated._episodic_memory
        )

        patcher, captured = _mock_scoring_call(sample_judge.name, sample_judge.model)
        with patcher:
            result = updated(inputs="test input", outputs="test output")

        assert len(captured) == 1
        assert result.value == "yes"


def test_align_dedupes_traces_within_a_single_call(sample_judge, sample_traces):
    with mock_apis(guidelines=[]) as mocks:
        optimizer = MemAlignOptimizer()
        judge = optimizer.align(sample_judge, [sample_traces[0], sample_traces[0]])

        assert len(judge._episodic_memory) == 1
        assert judge._episodic_trace_ids == [sample_traces[0].info.trace_id]
        # In-batch dedup: distillation runs once for the single resolved example,
        # not twice for the duplicate trace.
        assert mocks["lm"].call_count == 1


def test_realign_with_all_empty_assessments_raises(sample_judge, sample_traces):
    with mock_apis(guidelines=[]):
        optimizer = MemAlignOptimizer()
        judge_v1 = optimizer.align(sample_judge, sample_traces[:3])

        for trace in sample_traces[:3]:
            for assessment in trace.info.assessments:
                mlflow.delete_assessment(
                    trace_id=trace.info.trace_id, assessment_id=assessment.assessment_id
                )
        refreshed = [_refresh(t) for t in sample_traces[:3]]

        with pytest.raises(MlflowException, match="Cannot retract feedback"):
            optimizer.align(judge_v1, refreshed)


def test_realign_with_partial_empty_assessments_raises(sample_judge, sample_traces):
    # Even when some incoming traces are valid, any trace whose previously-aligned
    # assessments have been emptied must block the entire call — retractions go
    # through unalign(), not through a side-effect of align().
    with mock_apis(guidelines=[]):
        optimizer = MemAlignOptimizer()
        judge_v1 = optimizer.align(sample_judge, sample_traces[:3])
        v1_trace_ids = set(judge_v1._episodic_trace_ids)
        v1_episodic_count = len(judge_v1._episodic_memory)

        target = sample_traces[0]
        for assessment in target.info.assessments:
            mlflow.delete_assessment(
                trace_id=target.info.trace_id, assessment_id=assessment.assessment_id
            )
        emptied = _refresh(target)

        with pytest.raises(MlflowException, match="Cannot retract feedback"):
            optimizer.align(judge_v1, [emptied, sample_traces[1], sample_traces[3]])

        # No state mutation on the prior judge.
        assert set(judge_v1._episodic_trace_ids) == v1_trace_ids
        assert len(judge_v1._episodic_memory) == v1_episodic_count


def test_align_with_partial_no_feedback_traces_raises(sample_judge, sample_traces):
    # A trace that was never previously aligned and has no human assessments is
    # treated as user error (likely wrong trace IDs / missing feedback) and blocks
    # the call regardless of whether other valid traces are present.
    no_feedback_trace_id = _start_test_trace("no_feedback_span")
    no_feedback_trace = mlflow.get_trace(no_feedback_trace_id)

    with mock_apis(guidelines=[]):
        optimizer = MemAlignOptimizer()
        with pytest.raises(MlflowException, match="No valid feedback records found"):
            optimizer.align(sample_judge, [no_feedback_trace, sample_traces[0]])
