import sys
from unittest.mock import MagicMock, patch

import pytest

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.span import SpanType
from mlflow.exceptions import MlflowException


@pytest.fixture
def mock_trulens_openai():
    mock_openai_class = MagicMock()
    provider_instance = MagicMock()
    mock_openai_class.return_value = provider_instance

    mock_trulens_module = MagicMock()
    mock_providers_module = MagicMock()
    mock_openai_module = MagicMock()
    mock_openai_module.OpenAI = mock_openai_class

    with patch.dict(
        sys.modules,
        {
            "trulens": mock_trulens_module,
            "trulens.providers": mock_providers_module,
            "trulens.providers.openai": mock_openai_module,
        },
    ):
        yield provider_instance


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


def test_trulens_groundedness_scorer(mock_trulens_openai):
    mock_trulens_openai.groundedness_measure_with_cot_reasons.return_value = (
        0.85,
        {"reason": "Output is grounded in context"},
    )

    from mlflow.genai.scorers import TruLensGroundednessScorer

    scorer = TruLensGroundednessScorer()
    result = scorer(
        outputs="Paris is the capital of France.",
        context="France is a country in Europe. Its capital is Paris.",
    )

    assert isinstance(result, Feedback)
    assert result.name == "trulens_groundedness"
    assert result.value == 0.85
    assert "reason" in result.rationale

    mock_trulens_openai.groundedness_measure_with_cot_reasons.assert_called_once()


def test_trulens_groundedness_scorer_with_list_context(mock_trulens_openai):
    mock_trulens_openai.groundedness_measure_with_cot_reasons.return_value = (
        0.9,
        {"reason": "Grounded"},
    )

    from mlflow.genai.scorers import TruLensGroundednessScorer

    scorer = TruLensGroundednessScorer()
    result = scorer(
        outputs="The answer is 42.",
        context=["Context chunk 1", "Context chunk 2", "Context chunk 3"],
    )

    assert result.value == 0.9
    call_args = mock_trulens_openai.groundedness_measure_with_cot_reasons.call_args
    assert "Context chunk 1\nContext chunk 2\nContext chunk 3" in call_args[1]["source"]


def test_trulens_context_relevance_scorer(mock_trulens_openai):
    mock_trulens_openai.context_relevance_with_cot_reasons.return_value = (
        0.95,
        {"reason": "Context is highly relevant to query"},
    )

    from mlflow.genai.scorers import TruLensContextRelevanceScorer

    scorer = TruLensContextRelevanceScorer()
    result = scorer(
        inputs={"query": "What is machine learning?"},
        context="Machine learning is a subset of AI that enables systems to learn from data.",
    )

    assert isinstance(result, Feedback)
    assert result.name == "trulens_context_relevance"
    assert result.value == 0.95

    mock_trulens_openai.context_relevance_with_cot_reasons.assert_called_once()


def test_trulens_answer_relevance_scorer(mock_trulens_openai):
    mock_trulens_openai.relevance_with_cot_reasons.return_value = (
        0.88,
        {"reason": "Answer directly addresses the question"},
    )

    from mlflow.genai.scorers import TruLensAnswerRelevanceScorer

    scorer = TruLensAnswerRelevanceScorer()
    result = scorer(
        inputs={"query": "What is Python?"},
        outputs="Python is a high-level programming language known for its simplicity.",
    )

    assert isinstance(result, Feedback)
    assert result.name == "trulens_answer_relevance"
    assert result.value == 0.88

    mock_trulens_openai.relevance_with_cot_reasons.assert_called_once()


def test_trulens_coherence_scorer(mock_trulens_openai):
    mock_trulens_openai.coherence_with_cot_reasons.return_value = (
        0.92,
        {"reason": "Text flows logically"},
    )

    from mlflow.genai.scorers import TruLensCoherenceScorer

    scorer = TruLensCoherenceScorer()
    result = scorer(
        outputs="Machine learning is a branch of AI. It enables systems to learn from data. "
        "This learning process improves performance over time.",
    )

    assert isinstance(result, Feedback)
    assert result.name == "trulens_coherence"
    assert result.value == 0.92

    mock_trulens_openai.coherence_with_cot_reasons.assert_called_once()


def test_trulens_logical_consistency_scorer(mock_trulens_openai, sample_agent_trace):
    mock_trulens_openai.logical_consistency_with_cot_reasons.return_value = (
        0.87,
        {"reason": "Agent reasoning is logically consistent"},
    )

    from mlflow.genai.scorers import TruLensLogicalConsistencyScorer

    scorer = TruLensLogicalConsistencyScorer()
    result = scorer(trace=sample_agent_trace)

    assert isinstance(result, Feedback)
    assert result.name == "trulens_logical_consistency"
    assert result.value == 0.87

    mock_trulens_openai.logical_consistency_with_cot_reasons.assert_called_once()
    call_args = mock_trulens_openai.logical_consistency_with_cot_reasons.call_args
    assert "trace" in call_args[1]


def test_trulens_execution_efficiency_scorer(mock_trulens_openai, sample_agent_trace):
    mock_trulens_openai.execution_efficiency_with_cot_reasons.return_value = (
        0.75,
        {"reason": "Agent execution was mostly efficient"},
    )

    from mlflow.genai.scorers import TruLensExecutionEfficiencyScorer

    scorer = TruLensExecutionEfficiencyScorer()
    result = scorer(trace=sample_agent_trace)

    assert isinstance(result, Feedback)
    assert result.name == "trulens_execution_efficiency"
    assert result.value == 0.75

    mock_trulens_openai.execution_efficiency_with_cot_reasons.assert_called_once()


def test_trulens_plan_adherence_scorer(mock_trulens_openai, sample_agent_trace):
    mock_trulens_openai.plan_adherence_with_cot_reasons.return_value = (
        0.93,
        {"reason": "Agent followed the plan closely"},
    )

    from mlflow.genai.scorers import TruLensPlanAdherenceScorer

    scorer = TruLensPlanAdherenceScorer()
    result = scorer(trace=sample_agent_trace)

    assert isinstance(result, Feedback)
    assert result.name == "trulens_plan_adherence"
    assert result.value == 0.93

    mock_trulens_openai.plan_adherence_with_cot_reasons.assert_called_once()


def test_trulens_plan_quality_scorer(mock_trulens_openai, sample_agent_trace):
    mock_trulens_openai.plan_quality_with_cot_reasons.return_value = (
        0.82,
        {"reason": "Plan was well-structured"},
    )

    from mlflow.genai.scorers import TruLensPlanQualityScorer

    scorer = TruLensPlanQualityScorer()
    result = scorer(trace=sample_agent_trace)

    assert isinstance(result, Feedback)
    assert result.name == "trulens_plan_quality"
    assert result.value == 0.82

    mock_trulens_openai.plan_quality_with_cot_reasons.assert_called_once()


def test_trulens_tool_selection_scorer(mock_trulens_openai, sample_agent_trace):
    mock_trulens_openai.tool_selection_with_cot_reasons.return_value = (
        0.91,
        {"reason": "Appropriate tools were selected"},
    )

    from mlflow.genai.scorers import TruLensToolSelectionScorer

    scorer = TruLensToolSelectionScorer()
    result = scorer(trace=sample_agent_trace)

    assert isinstance(result, Feedback)
    assert result.name == "trulens_tool_selection"
    assert result.value == 0.91

    mock_trulens_openai.tool_selection_with_cot_reasons.assert_called_once()


def test_trulens_tool_calling_scorer(mock_trulens_openai, sample_agent_trace):
    mock_trulens_openai.tool_calling_with_cot_reasons.return_value = (
        0.89,
        {"reason": "Tool calls were executed correctly"},
    )

    from mlflow.genai.scorers import TruLensToolCallingScorer

    scorer = TruLensToolCallingScorer()
    result = scorer(trace=sample_agent_trace)

    assert isinstance(result, Feedback)
    assert result.name == "trulens_tool_calling"
    assert result.value == 0.89

    mock_trulens_openai.tool_calling_with_cot_reasons.assert_called_once()


def test_trulens_scorer_with_custom_name_and_model(mock_trulens_openai):
    mock_trulens_openai.groundedness_measure_with_cot_reasons.return_value = (
        0.8,
        {"reason": "Grounded"},
    )

    from mlflow.genai.scorers import TruLensGroundednessScorer

    scorer = TruLensGroundednessScorer(
        name="custom_groundedness",
        model_name="gpt-4",
        model_provider="openai",
    )
    result = scorer(
        outputs="Test output",
        context="Test context",
    )

    assert result.name == "custom_groundedness"


def test_trulens_agent_trace_scorer_with_custom_params(mock_trulens_openai, sample_agent_trace):
    mock_trulens_openai.logical_consistency_with_cot_reasons.return_value = (
        0.9,
        {"reason": "Consistent"},
    )

    from mlflow.genai.scorers import TruLensLogicalConsistencyScorer

    scorer = TruLensLogicalConsistencyScorer(
        name="custom_logical_consistency",
        criteria="Custom evaluation criteria",
        custom_instructions="Additional instructions for evaluation",
        temperature=0.5,
    )
    result = scorer(trace=sample_agent_trace)

    assert result.name == "custom_logical_consistency"

    call_args = mock_trulens_openai.logical_consistency_with_cot_reasons.call_args
    assert call_args[1]["criteria"] == "Custom evaluation criteria"
    assert call_args[1]["custom_instructions"] == "Additional instructions for evaluation"
    assert call_args[1]["temperature"] == 0.5


def test_trulens_scorer_score_clamping(mock_trulens_openai):
    mock_trulens_openai.groundedness_measure_with_cot_reasons.return_value = (
        1.5,
        {"reason": "Over max"},
    )

    from mlflow.genai.scorers import TruLensGroundednessScorer

    scorer = TruLensGroundednessScorer()
    result = scorer(outputs="Test", context="Test")

    assert result.value == 1.0

    mock_trulens_openai.groundedness_measure_with_cot_reasons.return_value = (
        -0.5,
        {"reason": "Under min"},
    )

    result = scorer(outputs="Test", context="Test")
    assert result.value == 0.0


def test_trulens_agent_trace_scorer_requires_trace(mock_trulens_openai):
    from mlflow.genai.scorers import TruLensLogicalConsistencyScorer

    scorer = TruLensLogicalConsistencyScorer()

    with pytest.raises(MlflowException, match="Trace is required"):
        scorer(trace=None)


def test_trulens_agent_trace_scorer_accepts_string_trace(mock_trulens_openai):
    mock_trulens_openai.logical_consistency_with_cot_reasons.return_value = (
        0.85,
        {"reason": "Consistent"},
    )

    from mlflow.genai.scorers import TruLensLogicalConsistencyScorer

    scorer = TruLensLogicalConsistencyScorer()
    trace_json = '{"info": {}, "data": {"spans": []}}'
    result = scorer(trace=trace_json)

    assert result.value == 0.85
    call_args = mock_trulens_openai.logical_consistency_with_cot_reasons.call_args
    assert call_args[1]["trace"] == trace_json


def test_trulens_scorer_rationale_formatting(mock_trulens_openai):
    mock_trulens_openai.groundedness_measure_with_cot_reasons.return_value = (
        0.8,
        {
            "reason": "Main reason",
            "details": ["Detail 1", "Detail 2"],
            "score_breakdown": {"part1": 0.9, "part2": 0.7},
        },
    )

    from mlflow.genai.scorers import TruLensGroundednessScorer

    scorer = TruLensGroundednessScorer()
    result = scorer(outputs="Test", context="Test")

    assert "reason: Main reason" in result.rationale
    assert "details: Detail 1; Detail 2" in result.rationale


def test_trulens_scorer_empty_rationale(mock_trulens_openai):
    mock_trulens_openai.groundedness_measure_with_cot_reasons.return_value = (0.8, None)

    from mlflow.genai.scorers import TruLensGroundednessScorer

    scorer = TruLensGroundednessScorer()
    result = scorer(outputs="Test", context="Test")

    assert result.rationale == "No detailed reasoning available."


def test_trulens_scorers_available_in_module():
    from mlflow.genai.scorers import (
        TruLensAnswerRelevanceScorer,
        TruLensCoherenceScorer,
        TruLensContextRelevanceScorer,
        TruLensExecutionEfficiencyScorer,
        TruLensGroundednessScorer,
        TruLensLogicalConsistencyScorer,
        TruLensPlanAdherenceScorer,
        TruLensPlanQualityScorer,
        TruLensToolCallingScorer,
        TruLensToolSelectionScorer,
    )

    assert TruLensGroundednessScorer is not None
    assert TruLensContextRelevanceScorer is not None
    assert TruLensAnswerRelevanceScorer is not None
    assert TruLensCoherenceScorer is not None
    assert TruLensLogicalConsistencyScorer is not None
    assert TruLensExecutionEfficiencyScorer is not None
    assert TruLensPlanAdherenceScorer is not None
    assert TruLensPlanQualityScorer is not None
    assert TruLensToolSelectionScorer is not None
    assert TruLensToolCallingScorer is not None


@pytest.mark.parametrize(
    ("scorer_class", "scorer_name"),
    [
        ("TruLensGroundednessScorer", "trulens_groundedness"),
        ("TruLensContextRelevanceScorer", "trulens_context_relevance"),
        ("TruLensAnswerRelevanceScorer", "trulens_answer_relevance"),
        ("TruLensCoherenceScorer", "trulens_coherence"),
        ("TruLensLogicalConsistencyScorer", "trulens_logical_consistency"),
        ("TruLensExecutionEfficiencyScorer", "trulens_execution_efficiency"),
        ("TruLensPlanAdherenceScorer", "trulens_plan_adherence"),
        ("TruLensPlanQualityScorer", "trulens_plan_quality"),
        ("TruLensToolSelectionScorer", "trulens_tool_selection"),
        ("TruLensToolCallingScorer", "trulens_tool_calling"),
    ],
)
def test_trulens_scorer_default_names(scorer_class, scorer_name):
    import mlflow.genai.scorers as scorers_module

    scorer_cls = getattr(scorers_module, scorer_class)
    scorer = scorer_cls()
    assert scorer.name == scorer_name
