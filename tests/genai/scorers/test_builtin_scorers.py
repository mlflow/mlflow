from unittest.mock import call, patch

from databricks.rag_eval.evaluation.entities import CategoricalRating as DatabricksCategoricalRating

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_error import AssessmentError
from mlflow.entities.span import SpanType
from mlflow.genai.judges.builtin import CategoricalRating
from mlflow.genai.scorers import (
    Correctness,
    ExpectationsGuidelines,
    Guidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    RetrievalSufficiency,
    Safety,
)


def test_retrieval_groundedness(sample_rag_trace):
    with patch(
        "databricks.agents.evals.judges.groundedness",
        side_effect=lambda *args, **kwargs: Feedback(
            name="retrieval_groundedness", value=DatabricksCategoricalRating.YES
        ),
    ) as mock_groundedness:
        result = RetrievalGroundedness()(trace=sample_rag_trace)

    mock_groundedness.assert_has_calls(
        [
            call(
                request="{'question': 'query'}",
                response="answer",
                retrieved_context=[
                    {"content": "content_1", "doc_uri": "url_1"},
                    {"content": "content_2", "doc_uri": "url_2"},
                ],
                assessment_name="retrieval_groundedness",
            ),
            call(
                request="{'question': 'query'}",
                response="answer",
                retrieved_context=[{"content": "content_3"}],
                assessment_name="retrieval_groundedness",
            ),
        ],
    )
    assert len(result) == 2
    assert all(isinstance(f, Feedback) for f in result)
    assert result[0].value == CategoricalRating.YES
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
        results = RetrievalRelevance()(trace=sample_rag_trace)

    mock_chunk_relevance.assert_has_calls(
        [
            call(
                request="{'question': 'query'}",
                retrieved_context=[
                    {"content": "content_1", "doc_uri": "url_1"},
                    {"content": "content_2", "doc_uri": "url_2"},
                ],
                assessment_name="retrieval_relevance",
            ),
            call(
                request="{'question': 'query'}",
                retrieved_context=[{"content": "content_3"}],
                assessment_name="retrieval_relevance",
            ),
        ],
    )

    assert len(results) == 5  # 2 span-level feedbacks + 3 chunk-level feedbacks
    assert all(isinstance(f, Feedback) for f in results)

    retriever_span_ids = [
        s.span_id for s in sample_rag_trace.search_spans(span_type=SpanType.RETRIEVER)
    ]

    # First feedbacks is a span-level feedback for the first retriever span
    assert results[0].value == 0.5
    assert results[0].name == "retrieval_relevance/precision"
    assert results[0].span_id == retriever_span_ids[0]

    # Second and third feedbacks are chunk-level feedbacks for the first retriever span
    assert results[1].value == "yes"
    assert results[1].name == "retrieval_relevance"
    assert results[1].span_id == retriever_span_ids[0]
    assert results[2].value == "no"
    assert results[2].name == "retrieval_relevance"
    assert results[2].span_id == retriever_span_ids[0]

    # Fourth result is a span-level feedback for the second retriever span
    assert results[3].value == 1.0
    assert results[3].name == "retrieval_relevance/precision"
    assert results[3].span_id == retriever_span_ids[1]

    # Fifth result is a chunk-level feedback for the second retriever span
    assert results[4].value == "yes"
    assert results[4].name == "retrieval_relevance"
    assert results[4].span_id == retriever_span_ids[1]


def test_retrieval_relevance_handle_error_feedback(sample_rag_trace):
    mock_responses = [
        # Error feedback
        [
            Feedback(name="retrieval_relevance", value=DatabricksCategoricalRating.YES),
            Feedback(name="retrieval_relevance", error=AssessmentError(error_code="test")),
        ],
        # Empty feedback - skip span
        [],
    ]

    with patch(
        "databricks.agents.evals.judges.chunk_relevance", side_effect=mock_responses
    ) as mock_chunk_relevance:
        results = RetrievalRelevance()(trace=sample_rag_trace)

    assert mock_chunk_relevance.call_count == 2
    assert len(results) == 3
    assert results[0].value == 0.5  # Error feedback is handled as 0.0 relevance
    assert results[1].value == CategoricalRating.YES
    assert results[2].value is None
    assert results[2].error.error_code == "test"


def test_retrieval_sufficiency(sample_rag_trace):
    with patch(
        "databricks.agents.evals.judges.context_sufficiency",
        side_effect=lambda *args, **kwargs: Feedback(
            name="retrieval_sufficiency", value=DatabricksCategoricalRating.YES
        ),
    ) as mock_context_sufficiency:
        result = RetrievalSufficiency()(trace=sample_rag_trace)

    mock_context_sufficiency.assert_has_calls(
        [
            call(
                request="{'question': 'query'}",
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
                request="{'question': 'query'}",
                retrieved_context=[{"content": "content_3"}],
                expected_response="expected answer",
                expected_facts=["fact1", "fact2"],
                assessment_name="retrieval_sufficiency",
            ),
        ],
    )

    assert len(result) == 2
    assert all(isinstance(f, Feedback) for f in result)
    assert all(f.value == CategoricalRating.YES for f in result)
    expected_span_ids = [
        s.span_id for s in sample_rag_trace.search_spans(span_type=SpanType.RETRIEVER)
    ]
    actual_span_ids = [f.span_id for f in result]
    assert set(actual_span_ids) == set(expected_span_ids)


def test_retrieval_sufficiency_with_custom_expectations(sample_rag_trace):
    with patch(
        "databricks.agents.evals.judges.context_sufficiency",
        return_value=Feedback(name="retrieval_sufficiency", value=DatabricksCategoricalRating.YES),
    ) as mock_context_sufficiency:
        RetrievalSufficiency()(
            trace=sample_rag_trace,
            expectations={"expected_facts": ["fact3"]},
        )

    mock_context_sufficiency.assert_has_calls(
        [
            call(
                request="{'question': 'query'}",
                retrieved_context=[
                    {"content": "content_1", "doc_uri": "url_1"},
                    {"content": "content_2", "doc_uri": "url_2"},
                ],
                # Expectations stored in the trace is exploded
                expected_facts=["fact3"],
                expected_response="expected answer",
                assessment_name="retrieval_sufficiency",
            ),
            call(
                request="{'question': 'query'}",
                retrieved_context=[{"content": "content_3"}],
                expected_facts=["fact3"],
                expected_response="expected answer",
                assessment_name="retrieval_sufficiency",
            ),
        ],
    )


@patch("mlflow.genai.judges.meets_guidelines")
def test_guidelines(mock_guidelines):
    # 1. Called with per-row guidelines
    ExpectationsGuidelines()(
        inputs={"question": "query"},
        outputs="answer",
        expectations={"guidelines": ["guideline1", "guideline2"]},
    )

    mock_guidelines.assert_called_once_with(
        guidelines=["guideline1", "guideline2"],
        context={"request": "{'question': 'query'}", "response": "answer"},
        name="expectations_guidelines",
        model=None,
    )
    mock_guidelines.reset_mock()

    # 2. Called with global guidelines
    is_english = Guidelines(
        name="is_english",
        guidelines=["The response should be in English."],
        model="openai:/gpt-4.1-mini",
    )
    is_english(inputs={"question": "query"}, outputs="answer")

    mock_guidelines.assert_called_once_with(
        guidelines=["The response should be in English."],
        context={"request": "{'question': 'query'}", "response": "answer"},
        name="is_english",
        model="openai:/gpt-4.1-mini",
    )
    mock_guidelines.reset_mock()

    # 3. Test with string input (should wrap in list)
    is_polite = Guidelines(
        name="is_polite",
        guidelines="Be polite and respectful.",
        model="openai:/gpt-4.1-mini",
    )
    is_polite(inputs={"question": "query"}, outputs="answer")

    mock_guidelines.assert_called_once_with(
        guidelines="Be polite and respectful.",
        context={"request": "{'question': 'query'}", "response": "answer"},
        name="is_polite",
        model="openai:/gpt-4.1-mini",
    )


@patch("mlflow.genai.judges.is_context_relevant")
def test_relevance_to_query(mock_is_context_relevant):
    # 1. Test with default scorer
    RelevanceToQuery()(
        inputs={"question": "query"},
        outputs="answer",
    )

    mock_is_context_relevant.assert_called_once_with(
        request="{'question': 'query'}",
        context="answer",
        name="relevance_to_query",
        model=None,
    )
    mock_is_context_relevant.reset_mock()

    # 2. Test with custom model parameter
    relevance_custom = RelevanceToQuery(
        name="custom_relevance",
        model="openai:/gpt-4.1-mini",
    )
    relevance_custom(inputs={"question": "query"}, outputs="answer")

    mock_is_context_relevant.assert_called_once_with(
        request="{'question': 'query'}",
        context="answer",
        name="custom_relevance",
        model="openai:/gpt-4.1-mini",
    )


def test_safety():
    # String output
    with patch("databricks.agents.evals.judges.safety") as mock_safety:
        Safety()(outputs="answer")

    mock_safety.assert_called_once_with(
        response="answer",
        assessment_name="safety",
    )

    # Non-string output
    with patch("databricks.agents.evals.judges.safety") as mock_safety:
        Safety()(outputs={"answer": "yes", "reason": "This is a test"})

    mock_safety.assert_called_once_with(
        response="{'answer': 'yes', 'reason': 'This is a test'}",
        assessment_name="safety",
    )


@patch("mlflow.genai.judges.is_correct")
def test_correctness(mock_is_correct):
    # 1. Test with expected_facts
    Correctness()(
        inputs={"question": "query"},
        outputs="answer",
        expectations={"expected_facts": ["fact1", "fact2"]},
    )

    mock_is_correct.assert_called_once_with(
        request="{'question': 'query'}",
        response="answer",
        expected_facts=["fact1", "fact2"],
        expected_response=None,
        name="correctness",
        model=None,
    )
    mock_is_correct.reset_mock()

    # 2. Test with custom model parameter
    correctness_custom = Correctness(
        name="custom_correctness",
        model="openai:/gpt-4.1-mini",
    )
    correctness_custom(
        inputs={"question": "query"},
        outputs="answer",
        expectations={"expected_response": "expected answer"},
    )

    mock_is_correct.assert_called_once_with(
        request="{'question': 'query'}",
        response="answer",
        expected_facts=None,
        expected_response="expected answer",
        name="custom_correctness",
        model="openai:/gpt-4.1-mini",
    )
