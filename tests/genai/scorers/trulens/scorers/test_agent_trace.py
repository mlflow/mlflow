from unittest.mock import Mock, patch

import pytest
import trulens  # noqa: F401

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.entities.span import SpanType
from mlflow.exceptions import MlflowException


@pytest.fixture
def sample_agent_trace():
    @mlflow.trace(name="agent_workflow", span_type=SpanType.AGENT)
    def run_agent(query):
        with mlflow.start_span(name="plan", span_type=SpanType.CHAIN) as plan_span:
            plan_span.set_inputs({"query": query})
            plan_span.set_outputs({"plan": "1. Search for info 2. Summarize"})

        with mlflow.start_span(name="search_tool", span_type=SpanType.TOOL) as tool_span:
            tool_span.set_inputs({"search_query": query})
            tool_span.set_outputs({"results": ["Result 1", "Result 2"]})

        return "Final answer based on search results"

    run_agent("What is MLflow?")
    return mlflow.get_trace(mlflow.get_last_active_trace_id())


@pytest.fixture
def mock_provider():
    return Mock()


@pytest.mark.parametrize(
    ("scorer_class", "metric_name", "method_name"),
    [
        ("LogicalConsistency", "logical_consistency", "logical_consistency_with_cot_reasons"),
        ("ExecutionEfficiency", "execution_efficiency", "execution_efficiency_with_cot_reasons"),
        ("PlanAdherence", "plan_adherence", "plan_adherence_with_cot_reasons"),
        ("PlanQuality", "plan_quality", "plan_quality_with_cot_reasons"),
        ("ToolSelection", "tool_selection", "tool_selection_with_cot_reasons"),
        ("ToolCalling", "tool_calling", "tool_calling_with_cot_reasons"),
    ],
)
def test_agent_trace_scorer(
    mock_provider, sample_agent_trace, scorer_class, metric_name, method_name
):
    expected_score = 0.87
    expected_reasons = {"reason": "Test rationale"}

    with patch(
        "mlflow.genai.scorers.trulens.scorers.agent_trace.create_trulens_provider",
        return_value=mock_provider,
    ):
        from mlflow.genai.scorers import trulens

        scorer_cls = getattr(trulens, scorer_class)
        scorer = scorer_cls(model="openai:/gpt-4")

    getattr(mock_provider, method_name).return_value = (expected_score, expected_reasons)
    result = scorer(trace=sample_agent_trace)

    assert isinstance(result, Feedback)
    assert result.name == metric_name
    assert result.value == expected_score
    assert result.rationale == "reason: Test rationale"
    assert result.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert result.source.source_id == "openai:/gpt-4"
    assert result.metadata == {"mlflow.scorer.framework": "trulens"}

    method = getattr(mock_provider, method_name)
    method.assert_called_once()
    call_kwargs = method.call_args[1]
    assert call_kwargs["trace"] == sample_agent_trace.to_json()


def test_scorer_requires_trace(mock_provider):
    with patch(
        "mlflow.genai.scorers.trulens.scorers.agent_trace.create_trulens_provider",
        return_value=mock_provider,
    ):
        from mlflow.genai.scorers.trulens import LogicalConsistency

        scorer = LogicalConsistency()

    with pytest.raises(MlflowException, match="Trace is required"):
        scorer(trace=None)


def test_scorer_accepts_string_trace(mock_provider):
    with patch(
        "mlflow.genai.scorers.trulens.scorers.agent_trace.create_trulens_provider",
        return_value=mock_provider,
    ):
        from mlflow.genai.scorers.trulens import LogicalConsistency

        scorer = LogicalConsistency()

    mock_provider.logical_consistency_with_cot_reasons.return_value = (0.85, None)
    trace_json = '{"info": {}, "data": {"spans": []}}'
    result = scorer(trace=trace_json)

    assert result.value == 0.85
    assert result.metadata == {"mlflow.scorer.framework": "trulens"}
    mock_provider.logical_consistency_with_cot_reasons.assert_called_once_with(trace=trace_json)


@pytest.mark.parametrize(
    ("reasons", "expected_rationale"),
    [
        (
            {"reason": "Main reason", "details": ["Detail 1", "Detail 2"]},
            "reason: Main reason | details: Detail 1; Detail 2",
        ),
        ({"single": "value"}, "single: value"),
        (None, None),
    ],
)
def test_scorer_rationale_formatting(
    mock_provider, sample_agent_trace, reasons, expected_rationale
):
    with patch(
        "mlflow.genai.scorers.trulens.scorers.agent_trace.create_trulens_provider",
        return_value=mock_provider,
    ):
        from mlflow.genai.scorers.trulens import LogicalConsistency

        scorer = LogicalConsistency()

    mock_provider.logical_consistency_with_cot_reasons.return_value = (0.8, reasons)
    result = scorer(trace=sample_agent_trace)

    assert result.rationale == expected_rationale


def test_scorer_error_handling(mock_provider, sample_agent_trace):
    with patch(
        "mlflow.genai.scorers.trulens.scorers.agent_trace.create_trulens_provider",
        return_value=mock_provider,
    ):
        from mlflow.genai.scorers.trulens import LogicalConsistency

        scorer = LogicalConsistency(model="openai:/gpt-4")

    mock_provider.logical_consistency_with_cot_reasons.side_effect = RuntimeError(
        "Evaluation failed"
    )
    result = scorer(trace=sample_agent_trace)

    assert isinstance(result, Feedback)
    assert result.error is not None
    assert "Evaluation failed" in str(result.error)
    assert result.metadata == {"mlflow.scorer.framework": "trulens"}
