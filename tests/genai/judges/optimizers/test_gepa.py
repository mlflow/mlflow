import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace, TraceData, TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import Judge, JudgeField
from mlflow.genai.judges.optimizers.gepa import GePaAlignmentOptimizer


class MockJudge(Judge):
    """Mock Judge implementation for testing."""

    def __init__(self, name: str = "mock_judge", model: str = "openai:/gpt-4", **kwargs):
        super().__init__(name=name, **kwargs)
        self._model = model
        self._instructions = "Evaluate if {{ outputs }} is relevant to {{ inputs }}"

    @property
    def instructions(self) -> str:
        return self._instructions

    @property
    def model(self) -> str:
        return self._model

    def get_input_fields(self) -> list[JudgeField]:
        return [
            JudgeField(name="inputs", description="Input text"),
            JudgeField(name="outputs", description="Output text"),
        ]

    def __call__(self, **kwargs):
        return Feedback(name=self.name, value="yes", rationale="Mock evaluation")


@pytest.fixture
def mock_judge():
    return MockJudge(name="relevance", model="openai:/gpt-4o-mini")


@pytest.fixture
def sample_trace_with_human_feedback():
    """Create a trace with human feedback."""
    trace_info = TraceInfo(
        trace_id="trace_1",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1000,
        state=TraceState.OK,
        execution_duration=100,
        assessments=[
            Feedback(
                trace_id="trace_1",
                name="relevance",
                value="yes",
                rationale="The output is relevant",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN, source_id="user_1"
                ),
            )
        ],
    )
    trace_data = TraceData()
    return Trace(info=trace_info, data=trace_data)


@pytest.fixture
def sample_traces_with_assessments():
    """Create multiple traces with human feedback."""
    traces = []
    for i in range(15):
        trace_info = TraceInfo(
            trace_id=f"trace_{i}",
            trace_location=TraceLocation.from_experiment_id("0"),
            request_time=1000 + i,
            state=TraceState.OK,
            execution_duration=100,
            assessments=[
                Feedback(
                    trace_id=f"trace_{i}",
                    name="relevance",
                    value="yes" if i % 2 == 0 else "no",
                    rationale=f"Feedback {i}",
                    source=AssessmentSource(
                        source_type=AssessmentSourceType.HUMAN, source_id="user_1"
                    ),
                )
            ],
        )
        trace_data = TraceData()
        traces.append(Trace(info=trace_info, data=trace_data))
    return traces


@pytest.fixture
def sample_traces_mixed_assessments():
    """Create traces with both HUMAN and LLM_JUDGE assessments."""
    traces = []
    for i in range(12):
        assessments = [
            Feedback(
                trace_id=f"trace_{i}",
                name="relevance",
                value="yes" if i % 2 == 0 else "no",
                rationale=f"Human feedback {i}",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN, source_id="user_1"
                ),
            ),
            Feedback(
                trace_id=f"trace_{i}",
                name="relevance",
                value="maybe",
                rationale=f"LLM feedback {i}",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4"
                ),
            ),
        ]
        trace_info = TraceInfo(
            trace_id=f"trace_{i}",
            trace_location=TraceLocation.from_experiment_id("0"),
            request_time=1000 + i,
            state=TraceState.OK,
            execution_duration=100,
            assessments=assessments,
        )
        trace_data = TraceData()
        traces.append(Trace(info=trace_info, data=trace_data))
    return traces


def test_gepa_optimizer_initialization():
    optimizer = GePaAlignmentOptimizer(model="openai:/gpt-4o")
    assert optimizer._model == "openai:/gpt-4o"
    assert optimizer._max_metric_calls == 100
    assert optimizer._gepa_kwargs == {}


def test_gepa_optimizer_initialization_with_defaults():
    with patch(
        "mlflow.genai.judges.optimizers.gepa.get_default_model",
        return_value="default_model",
    ):
        optimizer = GePaAlignmentOptimizer()
        assert optimizer._model == "default_model"
        assert optimizer._max_metric_calls == 100


def test_gepa_optimizer_initialization_with_custom_params():
    optimizer = GePaAlignmentOptimizer(
        model="openai:/gpt-4o",
        max_metric_calls=50,
        gepa_kwargs={"custom_param": "value"},
    )
    assert optimizer._model == "openai:/gpt-4o"
    assert optimizer._max_metric_calls == 50
    assert optimizer._gepa_kwargs == {"custom_param": "value"}


def test_align_no_traces_raises_error(mock_judge):
    optimizer = GePaAlignmentOptimizer(model="openai:/gpt-4o")
    with pytest.raises(MlflowException, match="No traces provided"):
        optimizer.align(mock_judge, [])


def test_align_insufficient_traces_raises_error(mock_judge, sample_trace_with_human_feedback):
    optimizer = GePaAlignmentOptimizer(model="openai:/gpt-4o")
    # Only 1 trace, need at least 10
    with pytest.raises(MlflowException, match="At least 10 traces with human feedback"):
        optimizer.align(mock_judge, [sample_trace_with_human_feedback])


def test_align_no_human_feedback_raises_error(mock_judge):
    # Create traces without human feedback
    traces = []
    for i in range(12):
        trace_info = TraceInfo(
            trace_id=f"trace_{i}",
            trace_location=TraceLocation.from_experiment_id("0"),
            request_time=1000 + i,
            state=TraceState.OK,
            execution_duration=100,
            assessments=[],  # No assessments
        )
        trace_data = TraceData()
        traces.append(Trace(info=trace_info, data=trace_data))

    optimizer = GePaAlignmentOptimizer(model="openai:/gpt-4o")
    with pytest.raises(MlflowException, match="At least 10 traces with human feedback"):
        optimizer.align(mock_judge, traces)


def test_align_filters_traces_correctly(mock_judge, sample_traces_mixed_assessments):
    mock_gepa_module = MagicMock()
    mock_modules = {
        "gepa": mock_gepa_module,
        "gepa.core": MagicMock(),
        "gepa.core.adapter": MagicMock(),
    }
    mock_gepa_module.optimize.return_value = Mock(
        best_candidate={"instructions": "Optimized: Evaluate {{ outputs }} against {{ inputs }}"}
    )
    mock_gepa_module.EvaluationBatch = MagicMock()

    optimizer = GePaAlignmentOptimizer(model="openai:/gpt-4o")

    with patch.dict(sys.modules, mock_modules):
        result = optimizer.align(mock_judge, sample_traces_mixed_assessments)

    # Verify result
    assert result is not None
    assert isinstance(result, Judge)
    assert result.instructions == "Optimized: Evaluate {{ outputs }} against {{ inputs }}"

    # Verify GEPA was called
    mock_gepa_module.optimize.assert_called_once()
    call_kwargs = mock_gepa_module.optimize.call_args.kwargs

    # All 12 traces should be valid (they all have HUMAN feedback)
    assert len(call_kwargs["trainset"]) == 12


@pytest.mark.parametrize(
    ("model_uri", "expected_reflection_lm"),
    [
        ("openai:/gpt-4o", "openai/gpt-4o"),
        ("anthropic:/claude-3-5-sonnet-20241022", "anthropic/claude-3-5-sonnet-20241022"),
    ],
)
def test_align_with_different_model_providers(
    mock_judge, sample_traces_with_assessments, model_uri, expected_reflection_lm
):
    mock_gepa_module = MagicMock()
    mock_modules = {
        "gepa": mock_gepa_module,
        "gepa.core": MagicMock(),
        "gepa.core.adapter": MagicMock(),
    }
    optimized_instr = "Optimized: Check if {{ outputs }} directly addresses {{ inputs }}"
    mock_gepa_module.optimize.return_value = Mock(
        best_candidate={"instructions": optimized_instr}
    )
    mock_gepa_module.EvaluationBatch = MagicMock()

    optimizer = GePaAlignmentOptimizer(
        model=model_uri,
        max_metric_calls=50,
    )

    with patch.dict(sys.modules, mock_modules):
        result = optimizer.align(mock_judge, sample_traces_with_assessments)

    # Verify result
    assert result is not None
    assert result.model == mock_judge.model
    assert result.name == mock_judge.name
    assert result.instructions == optimized_instr

    # Verify GEPA was called with correct parameters
    mock_gepa_module.optimize.assert_called_once()
    call_kwargs = mock_gepa_module.optimize.call_args.kwargs

    assert call_kwargs["seed_candidate"] == {"instructions": mock_judge.instructions}
    assert len(call_kwargs["trainset"]) == 15
    assert call_kwargs["reflection_lm"] == expected_reflection_lm
    assert call_kwargs["max_metric_calls"] == 50
    assert call_kwargs["use_mlflow"] is True
    assert call_kwargs["adapter"] is not None


def test_align_with_custom_gepa_kwargs(mock_judge, sample_traces_with_assessments):
    mock_gepa_module = MagicMock()
    mock_modules = {
        "gepa": mock_gepa_module,
        "gepa.core": MagicMock(),
        "gepa.core.adapter": MagicMock(),
    }
    mock_gepa_module.optimize.return_value = Mock(
        best_candidate={"instructions": "Optimized: Check {{ outputs }} against {{ inputs }}"}
    )
    mock_gepa_module.EvaluationBatch = MagicMock()

    optimizer = GePaAlignmentOptimizer(
        model="openai:/gpt-4o",
        gepa_kwargs={"custom_param": "custom_value", "another_param": 123},
    )

    with patch.dict(sys.modules, mock_modules):
        optimizer.align(mock_judge, sample_traces_with_assessments)

    call_kwargs = mock_gepa_module.optimize.call_args.kwargs
    assert call_kwargs["custom_param"] == "custom_value"
    assert call_kwargs["another_param"] == 123


def test_align_version_compatibility(mock_judge, sample_traces_with_assessments):
    mock_gepa_module = MagicMock()
    mock_modules = {
        "gepa": mock_gepa_module,
        "gepa.core": MagicMock(),
        "gepa.core.adapter": MagicMock(),
    }
    mock_gepa_module.optimize.return_value = Mock(
        best_candidate={"instructions": "Optimized: Check {{ outputs }} against {{ inputs }}"}
    )
    mock_gepa_module.EvaluationBatch = MagicMock()

    optimizer = GePaAlignmentOptimizer(model="openai:/gpt-4o")

    with (
        patch.dict(sys.modules, mock_modules),
        patch("importlib.metadata.version", return_value="0.0.17"),
    ):
        optimizer.align(mock_judge, sample_traces_with_assessments)

    call_kwargs = mock_gepa_module.optimize.call_args.kwargs
    assert "use_mlflow" not in call_kwargs


def test_has_human_feedback_for_judge(mock_judge, sample_trace_with_human_feedback):
    optimizer = GePaAlignmentOptimizer(model="openai:/gpt-4o")

    # Should find human feedback for "relevance" judge
    assert optimizer._has_human_feedback_for_judge(
        sample_trace_with_human_feedback, "relevance"
    ) is True

    # Should not find for different judge name
    assert optimizer._has_human_feedback_for_judge(
        sample_trace_with_human_feedback, "correctness"
    ) is False


def test_has_human_feedback_case_insensitive():
    trace_info = TraceInfo(
        trace_id="trace_1",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1000,
        state=TraceState.OK,
        execution_duration=100,
        assessments=[
            Feedback(
                trace_id="trace_1",
                name="ReLeVaNcE",  # Mixed case
                value="yes",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN, source_id="user_1"
                ),
            )
        ],
    )
    trace = Trace(info=trace_info, data=TraceData())

    optimizer = GePaAlignmentOptimizer(model="openai:/gpt-4o")
    assert optimizer._has_human_feedback_for_judge(trace, "relevance") is True
    assert optimizer._has_human_feedback_for_judge(trace, "RELEVANCE") is True


def test_adapter_evaluate_basic(mock_judge):
    import gepa

    # Create trace with human feedback
    trace_info = TraceInfo(
        trace_id="trace_1",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1000,
        state=TraceState.OK,
        execution_duration=100,
        assessments=[
            Feedback(
                trace_id="trace_1",
                name="relevance",
                value="yes",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN, source_id="user_1"
                ),
            )
        ],
    )
    trace = Trace(info=trace_info, data=TraceData())

    # Mock gepa module
    mock_gepa_module = MagicMock()
    eval_batch_cls = (
        gepa.EvaluationBatch if hasattr(gepa, "EvaluationBatch") else MagicMock()
    )
    mock_gepa_module.EvaluationBatch = eval_batch_cls

    adapter = GePaAlignmentOptimizer._MlflowGEPAAdapter(
        base_judge=mock_judge,
        valid_traces=[trace],
    )

    # Mock make_judge to return a mock judge
    with (
        patch("mlflow.genai.judges.optimizers.gepa.make_judge", return_value=mock_judge),
        patch(
            "mlflow.genai.judges.optimizers.gepa.extract_request_from_trace",
            return_value="input text",
        ),
        patch(
            "mlflow.genai.judges.optimizers.gepa.extract_response_from_trace",
            return_value="output text",
        ),
    ):
        candidate = {"instructions": "New instructions"}
        result = adapter.evaluate([trace], candidate, capture_traces=False)

    assert result is not None
    assert len(result.outputs) == 1
    assert len(result.scores) == 1


@pytest.mark.parametrize(
    ("predicted", "expected", "score"),
    [
        # Exact matches
        ("yes", "yes", 1.0),
        ("no", "no", 1.0),
        # Case-insensitive matches
        ("YES", "yes", 1.0),
        ("Yes", "YES", 1.0),
        ("NO", "no", 1.0),
        # Whitespace handling
        (" yes ", "yes", 1.0),
        ("yes", " yes ", 1.0),
        ("  no  ", "no", 1.0),
        # Mismatches
        ("yes", "no", 0.0),
        ("true", "false", 0.0),
        ("pass", "fail", 0.0),
        # None handling
        (None, "yes", 0.0),
        ("yes", None, 0.0),
        (None, None, 0.0),
    ],
)
def test_adapter_agreement_score(predicted, expected, score):
    adapter = GePaAlignmentOptimizer._MlflowGEPAAdapter(
        base_judge=MockJudge(),
        valid_traces=[],
    )
    assert adapter._agreement_score(predicted, expected) == score


def test_adapter_extract_human_feedback():
    trace_info = TraceInfo(
        trace_id="trace_1",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1000,
        state=TraceState.OK,
        execution_duration=100,
        assessments=[
            Feedback(
                trace_id="trace_1",
                name="relevance",
                value="yes",
                rationale="Good answer",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN, source_id="user_1"
                ),
            )
        ],
    )
    trace = Trace(info=trace_info, data=TraceData())

    adapter = GePaAlignmentOptimizer._MlflowGEPAAdapter(
        base_judge=MockJudge(name="relevance"),
        valid_traces=[trace],
    )

    value, rationale = adapter._extract_human_feedback(trace)
    assert value == "yes"
    assert rationale == "Good answer"


def test_adapter_extract_human_feedback_most_recent():
    trace_info = TraceInfo(
        trace_id="trace_1",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1000,
        state=TraceState.OK,
        execution_duration=100,
        assessments=[
            Feedback(
                trace_id="trace_1",
                name="relevance",
                value="no",
                rationale="First assessment",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN, source_id="user_1"
                ),
                create_time_ms=1000,
            ),
            Feedback(
                trace_id="trace_1",
                name="relevance",
                value="yes",
                rationale="Updated assessment",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN, source_id="user_1"
                ),
                create_time_ms=2000,  # More recent
            ),
        ],
    )
    trace = Trace(info=trace_info, data=TraceData())

    adapter = GePaAlignmentOptimizer._MlflowGEPAAdapter(
        base_judge=MockJudge(name="relevance"),
        valid_traces=[trace],
    )

    value, rationale = adapter._extract_human_feedback(trace)
    assert value == "yes"
    assert rationale == "Updated assessment"


def test_adapter_ignores_llm_judge_feedback():
    trace_info = TraceInfo(
        trace_id="trace_1",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1000,
        state=TraceState.OK,
        execution_duration=100,
        assessments=[
            Feedback(
                trace_id="trace_1",
                name="relevance",
                value="maybe",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4"
                ),
                create_time_ms=1000,
            ),
            Feedback(
                trace_id="trace_1",
                name="relevance",
                value="yes",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN, source_id="user_1"
                ),
                create_time_ms=2000,
            ),
        ],
    )
    trace = Trace(info=trace_info, data=TraceData())

    adapter = GePaAlignmentOptimizer._MlflowGEPAAdapter(
        base_judge=MockJudge(name="relevance"),
        valid_traces=[trace],
    )

    value, _ = adapter._extract_human_feedback(trace)
    # Should extract HUMAN feedback, not LLM_JUDGE
    assert value == "yes"


@pytest.mark.parametrize(
    ("invalid_instructions", "description"),
    [
        ("Evaluate if the {{ outputs }} is good quality.", "missing {{ inputs }}"),
        ("", "empty instructions"),
        (
            "Evaluate if {{ inputs }} and {{ extra_var }} match.",
            "has extra variable {{ extra_var }}",
        ),
        ("Just plain text with no variables.", "no template variables"),
    ],
)
def test_align_raises_error_on_invalid_template_variables(
    mock_judge, sample_traces_with_assessments, invalid_instructions, description
):
    mock_gepa_module = MagicMock()
    mock_modules = {
        "gepa": mock_gepa_module,
        "gepa.core": MagicMock(),
        "gepa.core.adapter": MagicMock(),
    }
    mock_gepa_module.optimize.return_value = Mock(
        best_candidate={"instructions": invalid_instructions}
    )
    mock_gepa_module.EvaluationBatch = MagicMock()

    optimizer = GePaAlignmentOptimizer(model="openai:/gpt-4o")

    with patch.dict(sys.modules, mock_modules):
        with pytest.raises(
            MlflowException,
            match="Optimized instructions have different template variables"
        ):
            optimizer.align(mock_judge, sample_traces_with_assessments)


def test_adapter_handles_trace_with_missing_data():
    # Create trace without proper data
    trace_info = TraceInfo(
        trace_id="trace_1",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1000,
        state=TraceState.OK,
        execution_duration=100,
        assessments=[
            Feedback(
                trace_id="trace_1",
                name="relevance",
                value="yes",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN, source_id="user_1"
                ),
            )
        ],
    )
    trace = Trace(info=trace_info, data=TraceData())

    adapter = GePaAlignmentOptimizer._MlflowGEPAAdapter(
        base_judge=MockJudge(name="relevance"),
        valid_traces=[trace],
    )

    # Mock the judge evaluation to test error handling
    with patch("mlflow.genai.judges.optimizers.gepa.make_judge") as mock_make_judge:
        mock_judge_instance = Mock()
        mock_judge_instance.return_value = Feedback(
            name="relevance", value="yes", rationale="test"
        )
        mock_make_judge.return_value = mock_judge_instance

        with (
            patch(
                "mlflow.genai.judges.optimizers.gepa.extract_request_from_trace",
                return_value=None,
            ),
            patch(
                "mlflow.genai.judges.optimizers.gepa.extract_response_from_trace",
                return_value=None,
            ),
        ):
            candidate = {"instructions": "Test {{ inputs }} and {{ outputs }}"}
            result = adapter.evaluate([trace], candidate, capture_traces=False)

            # Should handle missing data gracefully
            assert len(result.outputs) == 1
            assert len(result.scores) == 1


def test_adapter_handles_judge_evaluation_exception():
    trace_info = TraceInfo(
        trace_id="trace_1",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1000,
        state=TraceState.OK,
        execution_duration=100,
        assessments=[
            Feedback(
                trace_id="trace_1",
                name="relevance",
                value="yes",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN, source_id="user_1"
                ),
            )
        ],
    )
    trace = Trace(info=trace_info, data=TraceData())

    adapter = GePaAlignmentOptimizer._MlflowGEPAAdapter(
        base_judge=MockJudge(name="relevance"),
        valid_traces=[trace],
    )

    # Mock make_judge to raise an exception
    with patch("mlflow.genai.judges.optimizers.gepa.make_judge") as mock_make_judge:
        mock_judge_instance = Mock()
        mock_judge_instance.side_effect = Exception("Judge evaluation failed")
        mock_make_judge.return_value = mock_judge_instance

        candidate = {"instructions": "Test {{ inputs }} and {{ outputs }}"}
        result = adapter.evaluate([trace], candidate, capture_traces=False)

        # Should handle exception gracefully with 0.0 score
        assert len(result.outputs) == 1
        assert result.outputs[0] is None
        assert result.scores[0] == 0.0


def test_align_raises_error_on_empty_optimized_instructions(
    mock_judge, sample_traces_with_assessments
):
    mock_gepa_module = MagicMock()
    mock_modules = {
        "gepa": mock_gepa_module,
        "gepa.core": MagicMock(),
        "gepa.core.adapter": MagicMock(),
    }
    # Return empty instructions
    mock_gepa_module.optimize.return_value = Mock(
        best_candidate={"instructions": ""}
    )
    mock_gepa_module.EvaluationBatch = MagicMock()

    optimizer = GePaAlignmentOptimizer(model="openai:/gpt-4o")

    with patch.dict(sys.modules, mock_modules):
        with pytest.raises(
            MlflowException,
            match="Optimized instructions have different template variables"
        ):
            optimizer.align(mock_judge, sample_traces_with_assessments)
