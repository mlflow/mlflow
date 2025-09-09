from unittest.mock import call, patch

import pytest

import mlflow
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
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.builtin_scorers import get_all_scorers
from mlflow.utils.uri import is_databricks_uri

from tests.genai.conftest import databricks_only


@patch("mlflow.genai.judges.is_grounded")
def test_retrieval_groundedness(mock_is_grounded, sample_rag_trace):
    mock_is_grounded.side_effect = lambda *args, **kwargs: Feedback(
        name="retrieval_groundedness", value=CategoricalRating.YES
    )

    # 1. Test with default scorer
    result = RetrievalGroundedness()(trace=sample_rag_trace)

    mock_is_grounded.assert_has_calls(
        [
            call(
                request="{'question': 'query'}",
                response="answer",
                context=[
                    {"content": "content_1", "doc_uri": "url_1"},
                    {"content": "content_2", "doc_uri": "url_2"},
                ],
                name="retrieval_groundedness",
                model=None,
            ),
            call(
                request="{'question': 'query'}",
                response="answer",
                context=[{"content": "content_3"}],
                name="retrieval_groundedness",
                model=None,
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


@databricks_only
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


@databricks_only
def test_retrieval_relevance_handle_error_feedback(sample_rag_trace):
    from databricks.rag_eval.evaluation.entities import CategoricalRating as DatabricksRating

    mock_responses = [
        # Error feedback
        [
            Feedback(name="retrieval_relevance", value=DatabricksRating.YES),
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


def test_retrieval_relevance_with_custom_model(sample_rag_trace, monkeypatch: pytest.MonkeyPatch):
    # Set a dummy OpenAI key to avoid validation errors
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch(
        "mlflow.genai.scorers.builtin_scorers.invoke_judge_model",
        return_value=Feedback(
            name="retrieval_relevance", value="yes", rationale="Relevant content"
        ),
    ) as mock_invoke_judge:
        custom_model = "openai:/gpt-4"
        scorer = RetrievalRelevance(model=custom_model)
        results = scorer(trace=sample_rag_trace)

        # Should be called for each chunk (3 total chunks)
        assert mock_invoke_judge.call_count == 3

        # Verify model was passed correctly
        for call_args in mock_invoke_judge.call_args_list:
            args, kwargs = call_args
            assert args[0] == custom_model  # First positional arg is model
            assert kwargs["assessment_name"] == "retrieval_relevance"

        # 2 span-level + 3 chunk-level feedbacks
        assert len(results) == 5
        # Span-level feedbacks should be 100% relevance
        assert results[0].value == 1.0
        assert results[3].value == 1.0


@patch("mlflow.genai.judges.is_context_sufficient")
def test_retrieval_sufficiency(mock_is_context_sufficient, sample_rag_trace):
    mock_is_context_sufficient.side_effect = lambda *args, **kwargs: Feedback(
        name="retrieval_sufficiency", value=CategoricalRating.YES
    )

    # 1. Test with default scorer
    result = RetrievalSufficiency()(trace=sample_rag_trace)

    mock_is_context_sufficient.assert_has_calls(
        [
            call(
                request="{'question': 'query'}",
                context=[
                    {"content": "content_1", "doc_uri": "url_1"},
                    {"content": "content_2", "doc_uri": "url_2"},
                ],
                # Expectations stored in the trace is exploded
                expected_response="expected answer",
                expected_facts=["fact1", "fact2"],
                name="retrieval_sufficiency",
                model=None,
            ),
            call(
                request="{'question': 'query'}",
                context=[{"content": "content_3"}],
                expected_response="expected answer",
                expected_facts=["fact1", "fact2"],
                name="retrieval_sufficiency",
                model=None,
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

    mock_is_context_sufficient.reset_mock()

    # 2. Test with custom model parameter
    mock_is_context_sufficient.side_effect = lambda *args, **kwargs: Feedback(
        name="custom_sufficiency", value=CategoricalRating.YES
    )

    custom_scorer = RetrievalSufficiency(
        name="custom_sufficiency",
        model="openai:/gpt-4.1-mini",
    )
    result = custom_scorer(trace=sample_rag_trace)

    mock_is_context_sufficient.assert_has_calls(
        [
            call(
                request="{'question': 'query'}",
                context=[
                    {"content": "content_1", "doc_uri": "url_1"},
                    {"content": "content_2", "doc_uri": "url_2"},
                ],
                expected_response="expected answer",
                expected_facts=["fact1", "fact2"],
                name="custom_sufficiency",
                model="openai:/gpt-4.1-mini",
            ),
            call(
                request="{'question': 'query'}",
                context=[{"content": "content_3"}],
                expected_response="expected answer",
                expected_facts=["fact1", "fact2"],
                name="custom_sufficiency",
                model="openai:/gpt-4.1-mini",
            ),
        ],
    )
    assert len(result) == 2


@patch("mlflow.genai.judges.is_context_sufficient")
def test_retrieval_sufficiency_with_custom_expectations(
    mock_is_context_sufficient, sample_rag_trace
):
    mock_is_context_sufficient.return_value = Feedback(
        name="retrieval_sufficiency", value=CategoricalRating.YES
    )

    RetrievalSufficiency()(
        trace=sample_rag_trace,
        expectations={"expected_facts": ["fact3"]},
    )

    mock_is_context_sufficient.assert_has_calls(
        [
            call(
                request="{'question': 'query'}",
                context=[
                    {"content": "content_1", "doc_uri": "url_1"},
                    {"content": "content_2", "doc_uri": "url_2"},
                ],
                # Expectations stored in the trace is exploded
                expected_facts=["fact3"],
                expected_response="expected answer",
                name="retrieval_sufficiency",
                model=None,
            ),
            call(
                request="{'question': 'query'}",
                context=[{"content": "content_3"}],
                expected_facts=["fact3"],
                expected_response="expected answer",
                name="retrieval_sufficiency",
                model=None,
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


@databricks_only
def test_safety_databricks():
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
        response='{"answer": "yes", "reason": "This is a test"}',
        assessment_name="safety",
    )


def test_safety_non_databricks():
    mlflow.set_tracking_uri("file://")

    # Safety scorer should now work with non-Databricks tracking URIs
    safety_scorer = Safety()
    assert safety_scorer.name == "safety"


def test_safety_with_custom_model(monkeypatch: pytest.MonkeyPatch):
    # Set a dummy OpenAI key to avoid validation errors
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch(
        "mlflow.genai.judges.builtin.invoke_judge_model",
        return_value=Feedback(name="safety", value="yes", rationale="Safe content"),
    ) as mock_invoke_judge:
        custom_model = "anthropic:/claude-3-opus"
        scorer = Safety(model=custom_model)
        result = scorer(outputs="This is a safe response")

        mock_invoke_judge.assert_called_once()
        args, kwargs = mock_invoke_judge.call_args
        assert args[0] == custom_model  # First positional arg is model
        assert kwargs["assessment_name"] == "safety"

        assert result.name == "safety"
        assert result.value == "yes"
        assert result.rationale == "Safe content"


def test_safety_with_custom_model_and_name(monkeypatch: pytest.MonkeyPatch):
    # Set a dummy OpenAI key to avoid validation errors
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch(
        "mlflow.genai.judges.builtin.invoke_judge_model",
        return_value=Feedback(name="custom_safety", value="no", rationale="Unsafe content"),
    ) as mock_invoke_judge:
        custom_model = "openai:/gpt-4"
        scorer = Safety(name="custom_safety", model=custom_model)
        result = scorer(outputs={"response": "test content"})

        mock_invoke_judge.assert_called_once()
        args, kwargs = mock_invoke_judge.call_args
        assert args[0] == custom_model
        assert kwargs["assessment_name"] == "custom_safety"

        assert result.name == "custom_safety"
        assert result.value == "no"


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


@pytest.mark.parametrize("tracking_uri", ["file://test", "databricks"])
def test_get_all_scorers_oss(tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)

    scorers = get_all_scorers()

    # Safety and RetrievalRelevance are only available in Databricks
    assert len(scorers) == (7 if tracking_uri == "databricks" else 5)
    assert all(isinstance(scorer, Scorer) for scorer in scorers)


def test_retrieval_relevance_get_input_fields():
    """Test that RetrievalRelevance get_input_fields method returns expected field names."""
    if is_databricks_uri(mlflow.get_tracking_uri()):
        relevance = RetrievalRelevance(name="test")
        field_names = [field.name for field in relevance.get_input_fields()]
        assert field_names == ["trace"]


def test_retrieval_sufficiency_get_input_fields():
    """Test that RetrievalSufficiency get_input_fields method returns expected field names."""
    if is_databricks_uri(mlflow.get_tracking_uri()):
        sufficiency = RetrievalSufficiency(name="test")
        field_names = [field.name for field in sufficiency.get_input_fields()]
        assert field_names == ["trace", "expectations"]


def test_retrieval_groundedness_get_input_fields():
    """Test that RetrievalGroundedness get_input_fields method returns expected field names."""
    if is_databricks_uri(mlflow.get_tracking_uri()):
        groundedness = RetrievalGroundedness(name="test")
        field_names = [field.name for field in groundedness.get_input_fields()]
        assert field_names == ["trace"]


def test_guidelines_get_input_fields():
    """Test that Guidelines get_input_fields method returns expected field names."""
    guidelines = Guidelines(name="test", guidelines=["Be helpful"])
    field_names = [field.name for field in guidelines.get_input_fields()]
    assert field_names == ["inputs", "outputs"]


def test_expectations_guidelines_get_input_fields():
    """Test that ExpectationsGuidelines get_input_fields method returns expected field names."""
    exp_guidelines = ExpectationsGuidelines(name="test")
    field_names = [field.name for field in exp_guidelines.get_input_fields()]
    assert field_names == ["inputs", "outputs", "expectations"]


def test_relevance_to_query_get_input_fields():
    """Test that RelevanceToQuery get_input_fields method returns expected field names."""
    relevance_query = RelevanceToQuery(name="test")
    field_names = [field.name for field in relevance_query.get_input_fields()]
    assert field_names == ["inputs", "outputs"]


def test_safety_get_input_fields():
    """Test that Safety get_input_fields method returns expected field names."""
    if is_databricks_uri(mlflow.get_tracking_uri()):
        safety = Safety(name="test")
        field_names = [field.name for field in safety.get_input_fields()]
        assert field_names == ["outputs"]


def test_correctness_get_input_fields():
    """Test that Correctness get_input_fields method returns expected field names."""
    correctness = Correctness(name="test")
    field_names = [field.name for field in correctness.get_input_fields()]
    assert field_names == ["inputs", "outputs", "expectations"]
