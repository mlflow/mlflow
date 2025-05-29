from unittest.mock import call, patch

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_error import AssessmentError
from mlflow.entities.assessment_source import AssessmentSource
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import (
    correctness,
    guideline_adherence,
    relevance_to_query,
    retrieval_groundedness,
    retrieval_relevance,
    retrieval_sufficiency,
    safety,
)


def test_builtin_scorer_block_mutations():
    """Test that the built-in scorers are immutable."""
    with pytest.raises(MlflowException, match=r"Built-in scorer fields are immutable"):
        retrieval_groundedness.name = "new_name"


@pytest.mark.parametrize(
    ("scorer", "updates"),
    [
        (retrieval_relevance, {"name": "custom_name"}),
        (retrieval_sufficiency, {"name": "custom_name"}),
        (retrieval_groundedness, {"name": "custom_name"}),
        (relevance_to_query, {"name": "custom_name"}),
        (safety, {"name": "custom_name"}),
        (correctness, {"name": "custom_name"}),
        (
            guideline_adherence,
            {"name": "custom_name", "global_guidelines": ["Be polite", "Be kind"]},
        ),
    ],
    ids=lambda x: x.__class__.__name__,
)
def test_configure_builtin_scorers(scorer, updates):
    updated_scorer = scorer.with_config(**updates)

    assert updated_scorer is not scorer  # with_config() should return a new instance
    assert isinstance(updated_scorer, scorer.__class__)
    for key, value in updates.items():
        assert getattr(updated_scorer, key) == value

    # Positional argument should not be allowed
    with pytest.raises(TypeError, match=rf"{scorer.__class__.__name__}.with_config\(\) takes"):
        scorer.with_config("custom_name")


def test_retrieval_groundedness(sample_rag_trace):
    with patch("databricks.agents.evals.judges.groundedness") as mock_groundedness:
        retrieval_groundedness(trace=sample_rag_trace)

    mock_groundedness.assert_called_once_with(
        request="query",
        response="answer",
        retrieved_context=[
            {"content": "content_1", "doc_uri": "url_1"},
            {"content": "content_2", "doc_uri": "url_2"},
            {"content": "content_3"},
        ],
        assessment_name="retrieval_groundedness",
    )


@pytest.mark.parametrize(
    ("chunk_relevance_values", "expected"),
    [
        (["yes", "yes", "yes"], 1.0),
        (["yes", "yes", "no"], 0.666),
        (["no", "no", "no"], 0.0),
        (["yes", "no", None], 0.5),
    ],
)
def test_retrieval_relevance(sample_rag_trace, chunk_relevance_values, expected):
    def mock_chunk_relevance(request, retrieved_context, assessment_name):
        value = chunk_relevance_values.pop(0)
        error = AssessmentError(error_code="test") if value is None else None
        return [
            Feedback(
                name=assessment_name,
                value=value,
                error=error,
                source=AssessmentSource(source_type="LLM_JUDGE", source_id="test"),
                trace_id="tr-123",
                span_id="span-123",
            )
        ]

    with patch(
        "databricks.agents.evals.judges.chunk_relevance", side_effect=mock_chunk_relevance
    ) as mock_chunk_relevance:
        feedback = retrieval_relevance(trace=sample_rag_trace)

    mock_chunk_relevance.assert_has_calls(
        [
            call(
                request="query",
                retrieved_context=[{"content": "content_1", "doc_uri": "url_1"}],
                assessment_name="retrieval_relevance",
            ),
            call(
                request="query",
                retrieved_context=[{"content": "content_2", "doc_uri": "url_2"}],
                assessment_name="retrieval_relevance",
            ),
            call(
                request="query",
                retrieved_context=[{"content": "content_3"}],
                assessment_name="retrieval_relevance",
            ),
        ],
    )

    assert feedback.name == "retrieval_relevance"
    assert abs(feedback.value - expected) < 0.01
    assert feedback.source == AssessmentSource(source_type="LLM_JUDGE", source_id="test")
    assert feedback.trace_id == "tr-123"
    assert feedback.span_id == "span-123"


def test_retrieval_sufficiency(sample_rag_trace):
    with patch("databricks.agents.evals.judges.context_sufficiency") as mock_context_sufficiency:
        retrieval_sufficiency(trace=sample_rag_trace)

    mock_context_sufficiency.assert_called_once_with(
        request="query",
        retrieved_context=[
            {"content": "content_1", "doc_uri": "url_1"},
            {"content": "content_2", "doc_uri": "url_2"},
            {"content": "content_3"},
        ],
        expected_response="expected answer",
        expected_facts=["fact1", "fact2"],
        assessment_name="retrieval_sufficiency",
    )


def test_guideline_adherence():
    # 1. Called with per-row guidelines
    with patch("databricks.agents.evals.judges.guideline_adherence") as mock_guideline_adherence:
        guideline_adherence(
            inputs={"question": "query"},
            outputs="answer",
            expectations={"guidelines": ["guideline1", "guideline2"]},
        )

    mock_guideline_adherence.assert_called_once_with(
        request="query",
        response="answer",
        guidelines=["guideline1", "guideline2"],
        assessment_name="guideline_adherence",
    )

    # 2. Called with global guidelines
    is_english = guideline_adherence.with_config(
        name="is_english",
        global_guidelines=["The response should be in English."],
    )

    with patch("databricks.agents.evals.judges.guideline_adherence") as mock_guideline_adherence:
        is_english(
            inputs={"question": "query"},
            outputs="answer",
        )

    mock_guideline_adherence.assert_called_once_with(
        request="query",
        response="answer",
        guidelines=["The response should be in English."],
        assessment_name="is_english",
    )


def test_relevance_to_query():
    with patch("databricks.agents.evals.judges.relevance_to_query") as mock_relevance_to_query:
        relevance_to_query(
            inputs={"question": "query"},
            outputs="answer",
        )

    mock_relevance_to_query.assert_called_once_with(
        request="query",
        response="answer",
        assessment_name="relevance_to_query",
    )


def test_safety():
    with patch("databricks.agents.evals.judges.safety") as mock_safety:
        safety(
            inputs={"question": "query"},
            outputs="answer",
        )

    mock_safety.assert_called_once_with(
        request="query",
        response="answer",
        assessment_name="safety",
    )


def test_correctness():
    with patch("databricks.agents.evals.judges.correctness") as mock_correctness:
        correctness(
            inputs={"question": "query"},
            outputs="answer",
            expectations={"expected_facts": ["fact1", "fact2"]},
        )

    mock_correctness.assert_called_once_with(
        request="query",
        response="answer",
        expected_facts=["fact1", "fact2"],
        expected_response=None,
        assessment_name="correctness",
    )
