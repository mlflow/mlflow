from unittest.mock import MagicMock, call, patch

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
from mlflow.genai.scorers.builtin_scorers import (
    _extract_fields_from_trace,
    get_all_scorers,
)
from mlflow.utils.uri import is_databricks_uri

from tests.genai.conftest import databricks_only


@pytest.fixture
def trace_without_inputs_outputs():
    with mlflow.start_span(name="empty_span") as span:
        pass
    return mlflow.get_trace(span.trace_id)


@pytest.fixture
def trace_with_only_inputs():
    with mlflow.start_span(name="inputs_only_span") as span:
        span.set_inputs({"question": "Test question"})
    return mlflow.get_trace(span.trace_id)


@pytest.fixture
def trace_with_only_outputs():
    with mlflow.start_span(name="outputs_only_span") as span:
        span.set_outputs({"response": "Test response"})
    return mlflow.get_trace(span.trace_id)


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
    if is_databricks_uri(mlflow.get_tracking_uri()):
        relevance = RetrievalRelevance(name="test")
        field_names = [field.name for field in relevance.get_input_fields()]
        assert field_names == ["trace"]


def test_retrieval_sufficiency_get_input_fields():
    if is_databricks_uri(mlflow.get_tracking_uri()):
        sufficiency = RetrievalSufficiency(name="test")
        field_names = [field.name for field in sufficiency.get_input_fields()]
        assert field_names == ["trace", "expectations"]


def test_retrieval_groundedness_get_input_fields():
    if is_databricks_uri(mlflow.get_tracking_uri()):
        groundedness = RetrievalGroundedness(name="test")
        field_names = [field.name for field in groundedness.get_input_fields()]
        assert field_names == ["trace"]


def test_guidelines_get_input_fields():
    guidelines = Guidelines(name="test", guidelines=["Be helpful"])
    field_names = [field.name for field in guidelines.get_input_fields()]
    assert field_names == ["inputs", "outputs"]


def test_expectations_guidelines_get_input_fields():
    exp_guidelines = ExpectationsGuidelines(name="test")
    field_names = [field.name for field in exp_guidelines.get_input_fields()]
    assert field_names == ["inputs", "outputs", "expectations"]


def test_relevance_to_query_get_input_fields():
    relevance_query = RelevanceToQuery(name="test")
    field_names = [field.name for field in relevance_query.get_input_fields()]
    assert field_names == ["inputs", "outputs"]


def test_safety_get_input_fields():
    if is_databricks_uri(mlflow.get_tracking_uri()):
        safety = Safety(name="test")
        field_names = [field.name for field in safety.get_input_fields()]
        assert field_names == ["outputs"]


def test_correctness_get_input_fields():
    correctness = Correctness(name="test")
    field_names = [field.name for field in correctness.get_input_fields()]
    assert field_names == ["inputs", "outputs", "expectations"]


def create_simple_trace(inputs=None, outputs=None):
    @mlflow.trace(name="test_span", span_type=SpanType.CHAIN)
    def _create(question):
        return outputs or "MLflow is an open-source platform for ML lifecycle."

    input_dict = inputs or {"question": "What is MLflow?"}
    _create(**input_dict)
    return mlflow.get_trace(mlflow.get_last_active_trace_id())


def test_correctness_with_trace():
    with patch("mlflow.genai.judges.is_correct") as mock_is_correct:
        mock_is_correct.return_value = Feedback(
            name="correctness", value=True, rationale="Correct answer"
        )

        trace = create_simple_trace()
        scorer = Correctness()
        result = scorer(trace=trace, expectations={"expected_response": "Expected answer"})

        assert result.name == "correctness"
        assert result.value is True
        mock_is_correct.assert_called_once()


def test_guidelines_with_trace():
    with patch("mlflow.genai.judges.meets_guidelines") as mock_meets_guidelines:
        mock_meets_guidelines.return_value = Feedback(
            name="guidelines", value=True, rationale="Follows guidelines"
        )

        trace = create_simple_trace()
        scorer = Guidelines(guidelines=["Be helpful"])
        result = scorer(trace=trace)

        assert result.name == "guidelines"
        assert result.value is True
        mock_meets_guidelines.assert_called_once()


def test_relevance_to_query_with_trace():
    with patch("mlflow.genai.judges.is_context_relevant") as mock_is_context_relevant:
        mock_is_context_relevant.return_value = Feedback(
            name="relevance_to_query", value="yes", rationale="Relevant"
        )

        trace = create_simple_trace()
        scorer = RelevanceToQuery()
        result = scorer(trace=trace)

        assert result.name == "relevance_to_query"
        assert result.value == "yes"
        mock_is_context_relevant.assert_called_once()


@databricks_only
def test_safety_with_trace():
    with patch("databricks.agents.evals.judges.safety") as mock_safety:
        mock_safety.return_value = Feedback(name="safety", value="yes", rationale="Safe content")

        try:
            trace = create_simple_trace()
            if trace is None:
                pytest.skip("Could not create trace in test environment")
        except Exception:
            pytest.skip("Could not create trace in test environment")

        scorer = Safety()
        result = scorer(trace=trace)

        assert result.name == "safety"
        assert result.value == "yes"
        mock_safety.assert_called_once()


def test_correctness_fallback_with_expectations(trace_without_inputs_outputs):
    with patch("mlflow.genai.scorers.builtin_scorers.make_judge") as mock_make_judge:
        extraction_judge = MagicMock()
        extraction_judge.return_value = Feedback(
            name="field_extractor",
            value='{"inputs": {"question": "extracted question"}, "outputs": "extracted answer"}',
            rationale="Extracted fields",
        )

        with patch("mlflow.genai.judges.is_correct") as mock_is_correct:
            mock_is_correct.return_value = Feedback(
                name="correctness", value=True, rationale="Correct answer"
            )

            mock_make_judge.return_value = extraction_judge

            mlflow.log_expectation(
                trace_id=trace_without_inputs_outputs.info.trace_id,
                name="expected_response",
                value="Expected answer",
            )

            trace_with_expectations = mlflow.get_trace(trace_without_inputs_outputs.info.trace_id)

            scorer = Correctness()
            result = scorer(trace=trace_with_expectations)

            assert result.name == "correctness"
            assert result.value is True

            mock_make_judge.assert_called_once()
            call_args = mock_make_judge.call_args
            instructions = call_args[1]["instructions"]
            assert "{{ trace }}" in instructions

            mock_is_correct.assert_called_once()
            is_correct_args = mock_is_correct.call_args
            assert is_correct_args[1]["expected_response"] == "Expected answer"


def test_scorer_fallback_to_make_judge(trace_without_inputs_outputs):
    with patch("mlflow.genai.scorers.builtin_scorers.make_judge") as mock_make_judge:
        extraction_judge = MagicMock()
        extraction_judge.return_value = Feedback(
            name="field_extractor",
            value='{"inputs": {"question": "test"}, "outputs": "test response"}',
            rationale="Extracted fields",
        )
        mock_make_judge.return_value = extraction_judge

        with patch("mlflow.genai.judges.meets_guidelines") as mock_meets_guidelines:
            mock_meets_guidelines.return_value = Feedback(
                name="guidelines", value=True, rationale="Follows guidelines"
            )

            scorer = Guidelines(guidelines=["Be helpful"])
            result = scorer(trace=trace_without_inputs_outputs)

            assert result.name == "guidelines"
            assert result.value is True

            mock_make_judge.assert_called_once()
            call_args = mock_make_judge.call_args
            instructions = call_args[1]["instructions"]
            assert "{{ trace }}" in instructions

            mock_meets_guidelines.assert_called_once()
            guidelines_args = mock_meets_guidelines.call_args
            assert guidelines_args[1]["guidelines"] == ["Be helpful"]


@pytest.mark.parametrize(
    ("scorer_factory", "expected_name", "judge_to_mock"),
    [
        (lambda: Guidelines(guidelines=["Be concise"]), "guidelines", "meets_guidelines"),
        (lambda: RelevanceToQuery(), "relevance_to_context", "is_context_relevant"),
        (lambda: Safety(), "safety", "is_safe"),
    ],
)
def test_trace_not_formatted_into_prompt_for_fallback(
    scorer_factory, expected_name, judge_to_mock, trace_without_inputs_outputs
):
    with patch("mlflow.genai.scorers.builtin_scorers.make_judge") as mock_make_judge:
        extraction_judge = MagicMock()
        extraction_judge.return_value = Feedback(
            name="field_extractor",
            value='{"inputs": {"question": "test"}, "outputs": "test response"}',
            rationale="Extracted fields",
        )
        mock_make_judge.return_value = extraction_judge

        with patch(f"mlflow.genai.judges.{judge_to_mock}") as mock_judge:
            mock_judge.return_value = Feedback(
                name=expected_name, value=True, rationale="Test passed"
            )

            scorer = scorer_factory()
            result = scorer(trace=trace_without_inputs_outputs)
            assert result.name == expected_name

            call_args = mock_make_judge.call_args
            instructions = call_args[1]["instructions"]

            trace_json = trace_without_inputs_outputs.to_json()
            assert trace_json not in instructions
            assert "{{ trace }}" in instructions

            mock_judge.assert_called_once()


def test_correctness_with_override_outputs():
    with patch("mlflow.genai.judges.is_correct") as mock_is_correct:
        mock_is_correct.return_value = Feedback(name="correctness", value=True, rationale="Correct")

        trace = create_simple_trace()
        scorer = Correctness()

        result = scorer(
            trace=trace,
            inputs={"question": "Custom question"},
            outputs="Custom answer",
            expectations={"expected_response": "Custom expected"},
        )

        assert result.name == "correctness"
        assert result.value is True
        mock_is_correct.assert_called_once()
        call_args = mock_is_correct.call_args
        assert call_args[1]["request"] == "{'question': 'Custom question'}"
        assert call_args[1]["response"] == "Custom answer"
        assert call_args[1]["expected_response"] == "Custom expected"


def test_relevance_mixed_override():
    with patch("mlflow.genai.judges.is_context_relevant") as mock_is_context_relevant:
        mock_is_context_relevant.return_value = Feedback(
            name="relevance_to_query", value="yes", rationale="Relevant"
        )

        trace = create_simple_trace()
        scorer = RelevanceToQuery()
        result = scorer(trace=trace, inputs={"question": "New question"})

        assert result.name == "relevance_to_query"
        assert result.value == "yes"
        mock_is_context_relevant.assert_called_once()
        call_args = mock_is_context_relevant.call_args
        assert call_args[1]["request"] == "{'question': 'New question'}"
        assert call_args[1]["context"] == "MLflow is an open-source platform for ML lifecycle."


def test_trace_agent_mode_with_extra_fields(trace_with_only_inputs):
    with patch("mlflow.genai.scorers.builtin_scorers.make_judge") as mock_make_judge:
        extraction_judge = MagicMock()
        extraction_judge.return_value = Feedback(
            name="field_extractor",
            value='{"outputs": "extracted outputs"}',
            rationale="Extracted missing outputs",
        )
        mock_make_judge.return_value = extraction_judge

        with patch("mlflow.genai.judges.is_safe") as mock_is_safe:
            mock_is_safe.return_value = Feedback(
                name="safety", value="yes", rationale="Safe via trace"
            )

            scorer = Safety()
            result = scorer(trace=trace_with_only_inputs)

            assert result.name == "safety"
            assert result.value == "yes"

            mock_make_judge.assert_called_once()
            call_args = mock_make_judge.call_args
            instructions = call_args[1]["instructions"]
            assert "{{ trace }}" in instructions


def test_pure_trace_mode_with_expectations(trace_with_only_outputs):
    with patch("mlflow.genai.scorers.builtin_scorers.make_judge") as mock_make_judge:
        extraction_judge = MagicMock()
        extraction_judge.return_value = Feedback(
            name="field_extractor",
            value='{"inputs": {"question": "extracted question"}}',
            rationale="Extracted missing inputs",
        )
        mock_make_judge.return_value = extraction_judge

        with patch("mlflow.genai.judges.is_correct") as mock_is_correct:
            mock_is_correct.return_value = Feedback(
                name="correctness", value=True, rationale="Pure trace mode"
            )

            scorer = Correctness()
            result = scorer(
                trace=trace_with_only_outputs, expectations={"expected_response": "Expected answer"}
            )

            assert result.name == "correctness"
            assert result.value is True

            mock_make_judge.assert_called_once()
            call_args = mock_make_judge.call_args
            instructions = call_args[1]["instructions"]
            assert "{{ trace }}" in instructions


def test_correctness_default_extracts_from_trace():
    trace = create_simple_trace()

    mlflow.log_expectation(
        trace_id=trace.info.trace_id, name="expected_response", value="MLflow is a tool"
    )

    trace_with_expectations = mlflow.get_trace(trace.info.trace_id)

    with patch("mlflow.genai.judges.is_correct") as mock_is_correct:
        mock_is_correct.return_value = Feedback(
            name="correctness", value=True, rationale="Extracted from trace"
        )

        scorer = Correctness()
        result = scorer(trace=trace_with_expectations)

        assert result.name == "correctness"
        assert result.value is True
        mock_is_correct.assert_called_once()


def test_backwards_compatibility():
    with patch("mlflow.genai.judges.meets_guidelines") as mock_meets_guidelines:
        mock_meets_guidelines.return_value = Feedback(
            name="guidelines", value=True, rationale="Compatible"
        )

        trace = create_simple_trace()
        scorer = Guidelines(guidelines=["Be helpful"])
        result = scorer(
            inputs={"question": "What is MLflow?"}, outputs="MLflow is a platform", trace=trace
        )

        assert result.name == "guidelines"
        assert result.value is True
        mock_meets_guidelines.assert_called_once()


def test_expectations_guidelines_with_trace():
    with patch("mlflow.genai.judges.meets_guidelines") as mock_meets_guidelines:
        mock_meets_guidelines.return_value = Feedback(
            name="expectations_guidelines", value=True, rationale="Follows guidelines"
        )

        trace = create_simple_trace()
        scorer = ExpectationsGuidelines()
        result = scorer(trace=trace, expectations={"guidelines": ["Be helpful and concise"]})

        assert result.name == "expectations_guidelines"
        assert result.value is True
        mock_meets_guidelines.assert_called_once()


def test_expectations_guidelines_extraction_from_trace():
    trace = create_simple_trace()

    mlflow.log_expectation(
        trace_id=trace.info.trace_id, name="guidelines", value=["Be helpful", "Be concise"]
    )

    trace_with_expectations = mlflow.get_trace(trace.info.trace_id)

    with patch("mlflow.genai.judges.meets_guidelines") as mock_meets_guidelines:
        mock_meets_guidelines.return_value = Feedback(
            name="expectations_guidelines", value=True, rationale="Follows guidelines"
        )

        scorer = ExpectationsGuidelines()
        result = scorer(trace=trace_with_expectations)

        assert result.name == "expectations_guidelines"
        assert result.value is True
        mock_meets_guidelines.assert_called_once()
        call_args = mock_meets_guidelines.call_args
        assert call_args[1]["guidelines"] == ["Be helpful", "Be concise"]


def test_expectations_guidelines_fallback_with_trace(trace_without_inputs_outputs):
    with patch("mlflow.genai.scorers.builtin_scorers.make_judge") as mock_make_judge:
        extraction_judge = MagicMock()
        extraction_judge.return_value = Feedback(
            name="field_extractor",
            value='{"inputs": {"question": "test"}, "outputs": "test response"}',
            rationale="Extracted fields",
        )
        mock_make_judge.return_value = extraction_judge

        with patch("mlflow.genai.judges.meets_guidelines") as mock_meets_guidelines:
            mock_meets_guidelines.return_value = Feedback(
                name="expectations_guidelines", value=True, rationale="Follows guidelines"
            )

            scorer = ExpectationsGuidelines()
            result = scorer(
                trace=trace_without_inputs_outputs, expectations={"guidelines": ["Be helpful"]}
            )

            assert result.name == "expectations_guidelines"
            assert result.value is True

            mock_make_judge.assert_called_once()
            call_args = mock_make_judge.call_args
            instructions = call_args[1]["instructions"]
            assert "{{ trace }}" in instructions


def test_expectations_guidelines_mixed_override():
    with patch("mlflow.genai.judges.meets_guidelines") as mock_meets_guidelines:
        mock_meets_guidelines.return_value = Feedback(
            name="expectations_guidelines", value=True, rationale="Follows guidelines"
        )

        trace = create_simple_trace()
        scorer = ExpectationsGuidelines()
        result = scorer(
            trace=trace,
            inputs={"question": "New question"},
            expectations={"guidelines": ["Be helpful"]},
        )

        assert result.name == "expectations_guidelines"
        assert result.value is True
        mock_meets_guidelines.assert_called_once()
        call_args = mock_meets_guidelines.call_args
        assert call_args[1]["context"]["request"] == "{'question': 'New question'}"
        expected_response = "MLflow is an open-source platform for ML lifecycle."
        assert call_args[1]["context"]["response"] == expected_response


def test_extract_fields_from_trace_with_trace_extraction():
    trace = create_simple_trace()

    fields = _extract_fields_from_trace(trace=trace)
    assert fields.inputs == {"question": "What is MLflow?"}
    assert fields.outputs == "MLflow is an open-source platform for ML lifecycle."
    assert fields.expectations is None

    fields = _extract_fields_from_trace(
        trace=trace, inputs="override inputs", outputs="override outputs"
    )
    assert fields.inputs == "override inputs"
    assert fields.outputs == "override outputs"


def test_extract_fields_from_trace_with_expectations():
    trace = create_simple_trace()
    mlflow.log_expectation(
        trace_id=trace.info.trace_id, name="expected_response", value="MLflow is a tool"
    )
    trace_with_expectations = mlflow.get_trace(trace.info.trace_id)

    fields = _extract_fields_from_trace(trace=trace_with_expectations, extract_expectations=True)
    assert fields.inputs == {"question": "What is MLflow?"}
    assert fields.outputs == "MLflow is an open-source platform for ML lifecycle."
    assert fields.expectations == {"expected_response": "MLflow is a tool"}


def test_extract_fields_from_trace_llm_fallback(trace_without_inputs_outputs):
    with patch("mlflow.genai.scorers.builtin_scorers.make_judge") as mock_make_judge:
        mock_judge = MagicMock()
        mock_judge.return_value = Feedback(
            name="field_extractor",
            value='{"inputs": "extracted input", "outputs": "extracted output"}',
            rationale="Extracted fields",
        )
        mock_make_judge.return_value = mock_judge

        fields = _extract_fields_from_trace(
            trace=trace_without_inputs_outputs, model="openai:/gpt-4"
        )

        assert fields.inputs == "extracted input"
        assert fields.outputs == "extracted output"
        assert fields.expectations is None

        mock_make_judge.assert_called_once()
        call_args = mock_make_judge.call_args
        assert call_args[1]["name"] == "field_extractor"
        assert "{{ trace }}" in call_args[1]["instructions"]
        assert call_args[1]["model"] == "openai:/gpt-4"


def test_extract_fields_from_trace_llm_fallback_with_invalid_json(trace_without_inputs_outputs):
    with patch("mlflow.genai.scorers.builtin_scorers.make_judge") as mock_make_judge:
        mock_judge = MagicMock()
        mock_judge.return_value = Feedback(
            name="field_extractor", value="not valid json", rationale="Invalid extraction"
        )
        mock_make_judge.return_value = mock_judge

        fields = _extract_fields_from_trace(trace=trace_without_inputs_outputs)

        assert fields.inputs is None
        assert fields.outputs is None
        assert fields.expectations is None


def test_extract_fields_from_trace_partial_extraction():
    with mlflow.start_span(name="partial_span") as span:
        span.set_inputs({"question": "test"})
    trace_with_partial = mlflow.get_trace(span.trace_id)

    with patch("mlflow.genai.scorers.builtin_scorers.make_judge") as mock_make_judge:
        mock_judge = MagicMock()
        mock_judge.return_value = Feedback(
            name="field_extractor",
            value='{"outputs": "llm extracted output"}',
            rationale="Partial extraction",
        )
        mock_make_judge.return_value = mock_judge

        fields = _extract_fields_from_trace(trace=trace_with_partial)

        assert fields.inputs == {"question": "test"}
        assert fields.outputs == "llm extracted output"
        assert fields.expectations is None
