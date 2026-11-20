import copy
import json
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import dspy
import numpy as np
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
        assert restored._retrievers is None


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

    aligned_judge._retrievers = {
        "inputs": (MagicMock(side_effect=retriever_side_effect), [0, 1, 2])
    }

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


def test_concurrent_first_call_on_deferred_multi_field_judge_does_not_raise(sample_judge):
    # A second call that lands while the first is still building the per-field indexes must
    # not iterate a half-built retriever dict. As on master, it may retrieve nothing.
    building = threading.Event()
    release = threading.Event()
    errors = []

    class FakeEmbeddings:
        def __init__(self, embedder, corpus, k):
            self.embedder = lambda texts: np.ones((len(texts), 1))
            self.corpus_embeddings = np.ones((len(corpus), 1))
            if "answer" in corpus[0]:  # the second field's index: hold the first call here
                building.set()
                assert release.wait(timeout=30)

    def first_call():
        try:
            judge(inputs="q", outputs="a")
        except Exception as e:
            errors.append(e)

    examples = [dspy.Example(inputs="question", outputs="answer", result="yes", rationale="r")]
    with (
        patch("dspy.retrievers.Embeddings", FakeEmbeddings),
        patch("mlflow.genai.judges.optimizers.memalign.optimizer._build_embedder"),
        patch.object(MemoryAugmentedJudge, "_reconstruct_episodic_memory", return_value=examples),
        patch.object(InstructionsJudge, "__call__", return_value=MagicMock(metadata={})),
    ):
        judge = MemoryAugmentedJudge(
            sample_judge, retrieval_fields=["inputs", "outputs"], _defer_init=True
        )
        judge._episodic_trace_ids = ["t1"]
        first = threading.Thread(target=first_call, name="memalign-first-call")
        first.start()
        assert building.wait(timeout=30)
        try:
            judge(inputs="q", outputs="a")  # the first call is mid-build: indexes unpublished
            assert judge._retrievers is None
        finally:
            release.set()
            first.join(timeout=30)

    assert errors == []
    assert list(judge._retrievers) == ["inputs", "outputs"]


def test_deferred_judge_from_deserialization_supports_deepcopy_and_pickle(sample_judge):
    deferred = MemoryAugmentedJudge(sample_judge, retrieval_fields=["inputs"], _defer_init=True)
    restored = Scorer.model_validate(deferred.model_dump())

    assert copy.deepcopy(restored)._retrieval_fields == ["inputs"]
    assert pickle.loads(pickle.dumps(restored))._retrieval_fields == ["inputs"]


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


def test_memory_augmented_judge_round_trip_preserves_base_judge_session_level():
    # The wrapper itself is single-turn, but the wrapped judge's session_level must
    # survive a dump/restore cycle through the nested base-judge payload.
    base_judge = make_judge(
        name="conversation_judge",
        instructions="Evaluate whether the {{ conversation }} addresses the user's requests.",
        model="openai:/gpt-4",
        session_level="episode",
    )
    aligned = MemoryAugmentedJudge(base_judge=base_judge, _defer_init=True)

    dumped = aligned.model_dump()
    base_judge_data = dumped["memory_augmented_judge_data"]["base_judge"]
    assert base_judge_data["session_level"] == "episode"

    restored = Scorer.model_validate(dumped)
    assert isinstance(restored, MemoryAugmentedJudge)
    assert restored._base_judge.session_level == "episode"


# =============================================================================
# retrieval_fields tests
# =============================================================================


def _last_corpus(mocks):
    return mocks["embeddings_class"].call_args.kwargs["corpus"]


def _mock_base_judge_call():
    return patch.object(InstructionsJudge, "__call__", return_value=MagicMock(metadata={}))


@pytest.mark.parametrize(
    ("retrieval_fields", "match"),
    [
        (["context"], "unsupported value 'context'. Allowed values: inputs, outputs, expectations"),
        (["conversation"], "unsupported value 'conversation'"),
        ([], "non-empty list"),
        ("inputs", "non-empty list"),
        (["inputs", "inputs"], "duplicates"),
        ([1], "unsupported value 1"),
    ],
)
def test_retrieval_fields_validation(sample_judge, retrieval_fields, match):
    with pytest.raises(MlflowException, match=match) as exc_info:
        MemAlignOptimizer(retrieval_fields=retrieval_fields)
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"

    with pytest.raises(MlflowException, match=match) as exc_info:
        MemoryAugmentedJudge(sample_judge, retrieval_fields=retrieval_fields, _defer_init=True)
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"


def test_retrieval_fields_not_in_instructions_raises_at_align(sample_traces):
    judge = make_judge(
        name="test_judge", instructions="Is {{ inputs }} polite?", model="openai:/gpt-4"
    )
    with mock_apis(guidelines=[]):
        optimizer = MemAlignOptimizer(retrieval_fields=["outputs"])
        with pytest.raises(
            MlflowException,
            match="retrieval_fields includes 'outputs' but the judge's instructions only "
            "reference: inputs",
        ) as exc_info:
            optimizer.align(judge, sample_traces[:2])
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"

    # Deferred construction (e.g. deserialization) validates too, before any first call
    with pytest.raises(MlflowException, match="retrieval_fields includes 'outputs'") as exc_info:
        MemoryAugmentedJudge(judge, retrieval_fields=["outputs"], _defer_init=True)
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"


def test_retrieval_fields_single_field_builds_corpus_and_queries_that_field(
    sample_judge, sample_traces
):
    with mock_apis(guidelines=[]) as mocks:
        mocks["search"].return_value = MagicMock(indices=[0])
        optimizer = MemAlignOptimizer(retrieval_fields=["outputs"])
        aligned_judge = optimizer.align(sample_judge, sample_traces[:3])

        corpus = _last_corpus(mocks)
        assert len(corpus) == 3
        assert all("output_" in text and "input_" not in text for text in corpus)
        assert list(aligned_judge._retrievers) == ["outputs"]

        with _mock_base_judge_call():
            aligned_judge(inputs="the finding", outputs="the reply")
        mocks["search"].assert_called_once_with("the reply")


def test_retrieval_fields_multi_field_builds_one_index_per_field(sample_judge):
    with mock_apis(guidelines=[]) as mocks:
        judge = MemoryAugmentedJudge(sample_judge, retrieval_fields=["inputs", "outputs"])
        judge._episodic_memory = [
            dspy.Example(inputs="in_0", outputs="out_0"),
            dspy.Example(inputs="in_1", outputs=None),
            dspy.Example(inputs=None, outputs="out_2"),
            dspy.Example(inputs=None, outputs=None),
        ]
        judge._build_episodic_memory()

        corpora = [c.kwargs["corpus"] for c in mocks["embeddings_class"].call_args_list]
        assert corpora == [["in_0", "in_1"], ["out_0", "out_2"]]
        # The example with neither field is dropped; positions map corpus -> memory index
        assert len(judge._episodic_memory) == 3
        assert {field: positions for field, (_, positions) in judge._retrievers.items()} == {
            "inputs": [0, 1],
            "outputs": [0, 2],
        }


def _reply_embedder(texts):
    # Findings ("finding_<i>") sit at angle 0.1 * i in dims 0-1; replies are orthogonal one-hots
    # in dims 2-3, so every repeated reply embeds identically.
    vectors = []
    for text in texts:
        if text.startswith("finding_"):
            angle = 0.1 * int(text.removeprefix("finding_"))
            vectors.append([np.cos(angle), np.sin(angle), 0.0, 0.0])
        else:
            vectors.append([0.0, 0.0, float(text == "Fixed"), float(text == "Won't fix")])
    return np.array(vectors)


def test_multi_field_retrieval_matches_all_fields_with_repeated_values(sample_judge):
    # Labels follow the reply. The "Won't fix" findings are the closest to the query finding,
    # so a judge that only interleaves per-field rankings retrieves opposite-reply examples.
    replies = ["Won't fix"] * 6 + ["Fixed"] * 6
    examples = [
        dspy.Example(
            inputs=f"finding_{i}",
            outputs=reply,
            result="no" if reply == "Won't fix" else "yes",
            rationale="r",
        )
        for i, reply in enumerate(replies)
    ]
    with (
        patch(
            "mlflow.genai.judges.optimizers.memalign.optimizer._build_embedder",
            return_value=_reply_embedder,
        ),
        _mock_base_judge_call(),
    ):
        judge = MemoryAugmentedJudge(
            sample_judge, retrieval_k=5, retrieval_fields=["inputs", "outputs"]
        )
        judge._episodic_memory = list(examples)
        judge._build_episodic_memory()
        assessment = judge(inputs="finding_0", outputs="Fixed")

    retrieved = [
        int(trace_id.removeprefix("example_"))
        for trace_id in assessment.metadata["retrieved_example_trace_ids"]
    ]
    assert len(retrieved) == 5
    assert [replies[i] for i in retrieved] == ["Fixed"] * 5
    # Among same-reply examples, the closest findings win
    assert retrieved == [6, 7, 8, 9, 10]


def _build_single_field_index_and_score(judge, field, example, **call_kwargs):
    with mock_apis(guidelines=[]) as mocks:
        mocks["search"].return_value = MagicMock(indices=[0])
        memory_judge = MemoryAugmentedJudge(judge, retrieval_fields=[field])
        memory_judge._episodic_memory = [example]
        memory_judge._build_episodic_memory()
        assert list(memory_judge._retrievers) == [field]
        with _mock_base_judge_call():
            memory_judge(**call_kwargs)
    return _last_corpus(mocks), mocks["search"]


def test_retrieval_fields_expectations_builds_corpus_and_query():
    judge = make_judge(
        name="test_judge",
        instructions="Does {{ outputs }} match {{ expectations }}?",
        model="openai:/gpt-4",
    )
    example = dspy.Example(outputs="out_0", expectations={"answer": "42"})
    corpus, search = _build_single_field_index_and_score(
        judge, "expectations", example, outputs="out_q", expectations={"answer": "7"}
    )
    assert corpus == ["{'answer': '42'}"]
    search.assert_called_once_with("{'answer': '7'}")


def test_retrieval_fields_trace_builds_corpus_and_query(sample_traces):
    judge = make_judge(
        name="test_judge", instructions="Evaluate {{ trace }}", model="openai:/gpt-4"
    )
    trace = sample_traces[0]
    corpus, search = _build_single_field_index_and_score(
        judge, "trace", dspy.Example(trace=trace), trace=trace
    )
    assert len(corpus) == 1
    assert "input_0" in corpus[0]
    search.assert_called_once_with(corpus[0])


def test_retrieval_fields_corpus_uses_truncation(sample_judge):
    with (
        mock_apis(guidelines=[]) as mocks,
        patch(
            "mlflow.genai.judges.optimizers.memalign.optimizer.truncate_to_token_limit",
            side_effect=lambda text, model, model_type: text[:3],
        ) as mock_truncate,
    ):
        judge = MemoryAugmentedJudge(sample_judge, retrieval_fields=["outputs"])
        judge._episodic_memory = [dspy.Example(inputs="in_0", outputs="long output")]
        judge._build_episodic_memory()

    mock_truncate.assert_called_once_with(
        "long output", "openai:/text-embedding-3-small", model_type="embedding"
    )
    assert _last_corpus(mocks) == ["lon"]


def test_retrieval_fields_serialization_round_trip(sample_judge, sample_traces):
    with mock_apis(guidelines=[]):
        optimizer = MemAlignOptimizer(retrieval_fields=["outputs", "inputs"])
        aligned_judge = optimizer.align(sample_judge, sample_traces[:2])

    dumped = aligned_judge.model_dump()
    assert dumped["memory_augmented_judge_data"]["retrieval_fields"] == ["outputs", "inputs"]
    restored = Scorer.model_validate(dumped)
    assert restored._retrieval_fields == ["outputs", "inputs"]
    assert restored._create_copy()._retrieval_fields == ["outputs", "inputs"]

    # Payloads written before retrieval_fields existed load with the default
    del dumped["memory_augmented_judge_data"]["retrieval_fields"]
    assert Scorer.model_validate(dumped)._retrieval_fields is None


def test_deserialized_judge_builds_configured_index_on_first_call(sample_judge, sample_traces):
    with mock_apis(guidelines=[]):
        aligned_judge = MemAlignOptimizer(retrieval_fields=["outputs"]).align(
            sample_judge, sample_traces[:2]
        )
    restored = Scorer.model_validate(aligned_judge.model_dump())
    assert restored._retrievers is None

    examples = [dspy.Example(inputs="question", outputs="answer", result="yes", rationale="r")]
    with (
        mock_apis(guidelines=[]) as mocks,
        patch.object(MemoryAugmentedJudge, "_reconstruct_episodic_memory", return_value=examples),
        _mock_base_judge_call(),
    ):
        mocks["search"].return_value = MagicMock(indices=[0])
        restored(inputs="q", outputs="a")

    assert list(restored._retrievers) == ["outputs"]
    assert _last_corpus(mocks) == ["answer"]


def test_unalign_preserves_retrieval_fields(sample_judge, sample_traces):
    with mock_apis(guidelines=[]):
        optimizer = MemAlignOptimizer(retrieval_fields=["outputs"])
        aligned_judge = optimizer.align(sample_judge, sample_traces[:3])
        unaligned = aligned_judge.unalign(sample_traces[:1])
    assert unaligned._retrieval_fields == ["outputs"]
    assert list(unaligned._retrievers) == ["outputs"]


def test_realign_unchanged_traces_with_new_retrieval_fields_rebuilds_index(
    sample_judge, sample_traces
):
    with mock_apis(guidelines=["Guideline A"]) as mocks:
        judge_v1 = MemAlignOptimizer().align(sample_judge, sample_traces[:3])
        assert list(judge_v1._retrievers) == ["inputs"]
        lm_calls = mocks["lm"].call_count

        # Identical config keeps today's skip: the input judge is returned unchanged
        assert MemAlignOptimizer().align(judge_v1, sample_traces[:3]) is judge_v1

        judge_v2 = MemAlignOptimizer(retrieval_fields=["outputs"]).align(
            judge_v1, sample_traces[:3]
        )

    assert judge_v2 is not judge_v1
    assert judge_v2._retrieval_fields == ["outputs"]
    assert list(judge_v2._retrievers) == ["outputs"]
    assert all("output_" in text for text in _last_corpus(mocks))
    assert len(judge_v2._episodic_memory) == len(judge_v1._episodic_memory)
    assert [g.guideline_text for g in judge_v2._semantic_memory] == ["Guideline A"]
    # Memory is unchanged, so no re-distillation
    assert mocks["lm"].call_count == lm_calls


@pytest.mark.parametrize(
    "realign",
    [
        lambda judge, traces: MemAlignOptimizer().align(judge, traces),
        # Judge.align with no optimizer goes through get_default_optimizer()
        lambda judge, traces: judge.align(traces=traces),
    ],
    ids=["default_optimizer", "judge_align"],
)
def test_default_realign_keeps_retrieval_fields(sample_judge, sample_traces, realign):
    with mock_apis(guidelines=[]):
        aligned = MemAlignOptimizer(retrieval_fields=["outputs"]).align(
            sample_judge, sample_traces[:2]
        )
        # Unchanged traces: the input judge comes back as-is
        assert realign(aligned, sample_traces[:2]) is aligned
        realigned = realign(aligned, sample_traces[:4])

    assert realigned._retrieval_fields == ["outputs"]
    assert list(realigned._retrievers) == ["outputs"]


@pytest.mark.parametrize("num_traces", [2, 4], ids=["skip_unchanged", "add_new"])
def test_explicit_optimizer_retrieval_fields_override_judge_fields(
    sample_judge, sample_traces, num_traces
):
    with mock_apis(guidelines=[]) as mocks:
        aligned = MemAlignOptimizer(retrieval_fields=["outputs"]).align(
            sample_judge, sample_traces[:2]
        )
        realigned = MemAlignOptimizer(retrieval_fields=["inputs"]).align(
            aligned, sample_traces[:num_traces]
        )

    assert realigned is not aligned
    assert realigned._retrieval_fields == ["inputs"]
    assert list(realigned._retrievers) == ["inputs"]
    corpus = _last_corpus(mocks)
    assert len(corpus) == num_traces
    assert all("input_" in text and "output_" not in text for text in corpus)
    assert aligned._retrieval_fields == ["outputs"]


def test_realign_unchanged_traces_with_explicit_default_field_skips_rebuild(
    sample_judge, sample_traces
):
    with mock_apis(guidelines=[]) as mocks:
        judge_v1 = MemAlignOptimizer().align(sample_judge, sample_traces[:3])
        index_builds = mocks["embeddings_class"].call_count

        # sample_judge's default field is "inputs", so explicit ["inputs"] is not a change
        same = MemAlignOptimizer(retrieval_fields=["inputs"]).align(judge_v1, sample_traces[:3])

    assert same is judge_v1
    assert mocks["embeddings_class"].call_count == index_builds


def test_default_inputs_corpus_embeds_only_the_inputs_value(sample_judge, sample_traces):
    # getattr(example, "inputs") is dspy.Example's bound ``inputs`` method, whose repr would
    # leak the whole example (including the human label and rationale) into the corpus.
    with mock_apis(guidelines=[]) as mocks:
        MemAlignOptimizer().align(sample_judge, sample_traces[:2])

    corpus = _last_corpus(mocks)
    assert len(corpus) == 2
    for i, text in enumerate(corpus):
        assert f"input_{i}" in text
        assert "bound method" not in text
        assert "Reason" not in text
