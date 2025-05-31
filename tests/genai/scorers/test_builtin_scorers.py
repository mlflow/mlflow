from unittest.mock import call, patch

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_error import AssessmentError
from mlflow.entities.span import SpanType
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import (
    correctness,
    guidelines,
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
            guidelines,
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
    with patch(
        "databricks.agents.evals.judges.groundedness",
        side_effect=lambda *args, **kwargs: Feedback(name="retrieval_groundedness", value="yes"),
    ) as mock_groundedness:
        result = retrieval_groundedness(trace=sample_rag_trace)

    mock_groundedness.assert_has_calls(
        [
            call(
                request="query",
                response="answer",
                retrieved_context=[
                    {"content": "content_1", "doc_uri": "url_1"},
                    {"content": "content_2", "doc_uri": "url_2"},
                ],
                assessment_name="retrieval_groundedness",
            ),
            call(
                request="query",
                response="answer",
                retrieved_context=[{"content": "content_3"}],
                assessment_name="retrieval_groundedness",
            ),
        ],
    )
    assert len(result) == 2
    assert all(isinstance(f, Feedback) for f in result)
    expected_span_ids = [
        s.span_id for s in sample_rag_trace.search_spans(span_type=SpanType.RETRIEVER)
    ]
    actual_span_ids = [f.span_id for f in result]
    assert set(actual_span_ids) == set(expected_span_ids)


def test_retrieval_relevance(sample_rag_trace):
    mock_responses = [
        # First retriever span has 2 chunks
        [
            Feedback(name="retrieval_relevance", value="yes", metadata={"chunk_index": 0}),
            Feedback(name="retrieval_relevance", value="no", metadata={"chunk_index": 1}),
        ],
        # Second retriever span has 1 chunk
        [
            Feedback(name="retrieval_relevance", value="yes", metadata={"chunk_index": 0}),
        ],
    ]

    with patch(
        "databricks.agents.evals.judges.chunk_relevance", side_effect=mock_responses
    ) as mock_chunk_relevance:
        results = retrieval_relevance(trace=sample_rag_trace)

    mock_chunk_relevance.assert_has_calls(
        [
            call(
                request="query",
                retrieved_context=[
                    {"content": "content_1", "doc_uri": "url_1"},
                    {"content": "content_2", "doc_uri": "url_2"},
                ],
                assessment_name="retrieval_relevance",
            ),
            call(
                request="query",
                retrieved_context=[{"content": "content_3"}],
                assessment_name="retrieval_relevance",
            ),
        ],
    )

    assert len(results) == 5  # 2 span-level feedbacks + 3 chunk-level feedbacks
    assert all(isinstance(f, Feedback) for f in results)
    assert all(f.name == "retrieval_relevance" for f in results)

    retriever_span_ids = [
        s.span_id for s in sample_rag_trace.search_spans(span_type=SpanType.RETRIEVER)
    ]

    # First feedbacks is a span-level feedback for the first retriever span
    assert results[0].value == 0.5
    assert results[0].span_id == retriever_span_ids[0]

    # Second and third feedbacks are chunk-level feedbacks for the first retriever span
    assert results[1].value == "yes"
    assert results[1].span_id == retriever_span_ids[0]
    assert results[2].value == "no"
    assert results[2].span_id == retriever_span_ids[0]

    # Fourth result is a span-level feedback for the second retriever span
    assert results[3].value == 1.0
    assert results[3].span_id == retriever_span_ids[1]

    # Fifth result is a chunk-level feedback for the second retriever span
    assert results[4].value == "yes"
    assert results[4].span_id == retriever_span_ids[1]


def test_retrieval_relevance_handle_error_feedback(sample_rag_trace):
    mock_responses = [
        # Error feedback
        [
            Feedback(name="retrieval_relevance", value="yes"),
            Feedback(name="retrieval_relevance", error=AssessmentError(error_code="test")),
        ],
        # Empty feedback - skip span
        [],
    ]

    with patch(
        "databricks.agents.evals.judges.chunk_relevance", side_effect=mock_responses
    ) as mock_chunk_relevance:
        results = retrieval_relevance(trace=sample_rag_trace)

    assert mock_chunk_relevance.call_count == 2
    assert len(results) == 3
    assert results[0].value == 0.5  # Error feedback is handled as 0.0 relevance
    assert results[1].value == "yes"
    assert results[2].value is None
    assert results[2].error.error_code == "test"


def test_retrieval_sufficiency(sample_rag_trace):
    with patch(
        "databricks.agents.evals.judges.context_sufficiency",
        side_effect=lambda *args, **kwargs: Feedback(name="retrieval_sufficiency", value="yes"),
    ) as mock_context_sufficiency:
        result = retrieval_sufficiency(trace=sample_rag_trace)

    mock_context_sufficiency.assert_has_calls(
        [
            call(
                request="query",
                retrieved_context=[
                    {"content": "content_1", "doc_uri": "url_1"},
                    {"content": "content_2", "doc_uri": "url_2"},
                ],
                # Expectations stored in the trace is exploded
                expected_response="expected answer",
                expected_facts=["fact1", "fact2"],
                assessment_name="retrieval_sufficiency",
            ),
            call(
                request="query",
                retrieved_context=[{"content": "content_3"}],
                expected_response="expected answer",
                expected_facts=["fact1", "fact2"],
                assessment_name="retrieval_sufficiency",
            ),
        ],
    )

    assert len(result) == 2
    assert all(isinstance(f, Feedback) for f in result)
    expected_span_ids = [
        s.span_id for s in sample_rag_trace.search_spans(span_type=SpanType.RETRIEVER)
    ]
    actual_span_ids = [f.span_id for f in result]
    assert set(actual_span_ids) == set(expected_span_ids)


def test_retrieval_sufficiency_with_custom_expectations(sample_rag_trace):
    with patch("databricks.agents.evals.judges.context_sufficiency") as mock_context_sufficiency:
        retrieval_sufficiency(
            trace=sample_rag_trace,
            expectations={"expected_facts": ["fact3"]},
        )

    mock_context_sufficiency.assert_has_calls(
        [
            call(
                request="query",
                retrieved_context=[
                    {"content": "content_1", "doc_uri": "url_1"},
                    {"content": "content_2", "doc_uri": "url_2"},
                ],
                # Expectations stored in the trace is exploded
                expected_response="expected answer",
                expected_facts=["fact3"],
                assessment_name="retrieval_sufficiency",
            ),
            call(
                request="query",
                retrieved_context=[{"content": "content_3"}],
                expected_response="expected answer",
                expected_facts=["fact3"],
                assessment_name="retrieval_sufficiency",
            ),
        ],
    )


def test_guidelines():
    # 1. Called with per-row guidelines
    with patch("databricks.agents.evals.judges.guideline_adherence") as mock_guideline_adherence:
        guidelines(
            outputs="answer",
            expectations={"guidelines": ["guideline1", "guideline2"]},
        )

    mock_guideline_adherence.assert_called_once_with(
        guidelines=["guideline1", "guideline2"],
        guidelines_context={"response": "answer"},
        assessment_name="guideline_adherence",
    )

    # 2. Called with global guidelines
    is_english = guidelines.with_config(
        name="is_english",
        guidelines=["The response should be in English."],
    )

    with patch("databricks.agents.evals.judges.guideline_adherence") as mock_guideline_adherence:
        is_english(outputs="answer")

    mock_guideline_adherence.assert_called_once_with(
        guidelines=["The response should be in English."],
        guidelines_context={"response": "answer"},
        assessment_name="is_english",
    )


def test_relevance_to_query():
    with patch(
        "databricks.agents.evals.judges.chunk_relevance",
        return_value=[
            Feedback(name="relevance_to_query", value="yes", metadata={"chunk_index": 0})
        ],
    ) as mock_chunk_relevance:
        result = relevance_to_query(
            inputs={"question": "query"},
            outputs="answer",
        )

    mock_chunk_relevance.assert_called_once_with(
        request="query",
        retrieved_context=["answer"],
        assessment_name="relevance_to_query",
    )

    assert result.name == "relevance_to_query"
    assert result.value == "yes"
    assert result.metadata == {}  # chunk id should not be included in the metadata


def test_safety():
    # String output
    with patch("databricks.agents.evals.judges.safety") as mock_safety:
        safety(outputs="answer")

    mock_safety.assert_called_once_with(
        response="answer",
        assessment_name="safety",
    )

    # Non-string output
    with patch("databricks.agents.evals.judges.safety") as mock_safety:
        safety(outputs={"answer": "yes", "reason": "This is a test"})

    mock_safety.assert_called_once_with(
        response='{"answer": "yes", "reason": "This is a test"}',
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
