import json
from unittest import mock
from unittest.mock import Mock, call, patch

import pytest
from litellm.types.utils import ModelResponse

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_error import AssessmentError
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.span import SpanType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import JudgeField
from mlflow.genai.judges.builtin import CategoricalRating
from mlflow.genai.judges.utils import FieldExtraction
from mlflow.genai.scorers import (
    Completeness,
    ConversationalRoleAdherence,
    ConversationalSafety,
    ConversationalToolCallEfficiency,
    ConversationCompleteness,
    Correctness,
    Equivalence,
    ExpectationsGuidelines,
    Fluency,
    Guidelines,
    KnowledgeRetention,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    RetrievalSufficiency,
    Safety,
    Summarization,
    ToolCallCorrectness,
    ToolCallEfficiency,
    UserFrustration,
)
from mlflow.genai.scorers.base import Scorer, ScorerKind
from mlflow.genai.scorers.builtin_scorers import (
    ExtractedFields,
    _construct_field_extraction_config,
    _validate_required_fields,
    get_all_scorers,
    resolve_scorer_fields,
)
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.utils.uri import is_databricks_uri

from tests.genai.conftest import databricks_only


@pytest.fixture
def mock_openai_env(monkeypatch):
    """Fixture to mock OpenAI API key for tests that need it."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


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


@pytest.fixture
def mock_judge_with_inputs_outputs():
    judge = mock.Mock()
    judge.get_input_fields.return_value = [
        JudgeField(name="inputs", description="Test inputs"),
        JudgeField(name="outputs", description="Test outputs"),
    ]
    return judge


def test_retrieval_groundedness(sample_rag_trace):
    with patch(
        "mlflow.genai.judges.is_grounded",
        side_effect=lambda *args, **kwargs: Feedback(
            name="retrieval_groundedness", value=CategoricalRating.YES
        ),
    ) as mock_is_grounded:
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


@pytest.mark.usefixtures("mock_openai_env")
def test_retrieval_relevance_with_custom_model(sample_rag_trace):
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


def test_retrieval_sufficiency(sample_rag_trace):
    # 1. Test with default scorer
    with patch(
        "mlflow.genai.judges.is_context_sufficient",
        side_effect=lambda *args, **kwargs: Feedback(
            name="retrieval_sufficiency", value=CategoricalRating.YES
        ),
    ) as mock_is_context_sufficient:
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

    # 2. Test with custom model parameter
    with patch(
        "mlflow.genai.judges.is_context_sufficient",
        side_effect=lambda *args, **kwargs: Feedback(
            name="custom_sufficiency", value=CategoricalRating.YES
        ),
    ) as mock_is_context_sufficient:
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


def test_retrieval_sufficiency_with_custom_expectations(sample_rag_trace):
    with patch(
        "mlflow.genai.judges.is_context_sufficient",
        return_value=Feedback(name="retrieval_sufficiency", value=CategoricalRating.YES),
    ) as mock_is_context_sufficient:
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


def test_guidelines():
    # 1. Called with per-row guidelines
    with patch("mlflow.genai.judges.meets_guidelines") as mock_guidelines:
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

    # 2. Called with global guidelines
    with patch("mlflow.genai.judges.meets_guidelines") as mock_guidelines:
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

    # 3. Test with string input (should wrap in list)
    with patch("mlflow.genai.judges.meets_guidelines") as mock_guidelines:
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


def test_relevance_to_query():
    # 1. Test with default scorer
    with patch("mlflow.genai.judges.is_context_relevant") as mock_is_context_relevant:
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

    # 2. Test with custom model parameter
    with patch("mlflow.genai.judges.is_context_relevant") as mock_is_context_relevant:
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


@pytest.mark.usefixtures("mock_openai_env")
def test_safety_with_custom_model():
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
        assert result.value == CategoricalRating.YES
        assert result.rationale == "Safe content"


@pytest.mark.usefixtures("mock_openai_env")
def test_safety_with_custom_model_and_name():
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


def test_correctness():
    with patch("mlflow.genai.judges.is_correct") as mock_is_correct:
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

    with patch("mlflow.genai.judges.is_correct") as mock_is_correct:
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


def test_equivalence():
    # Test with default model
    scorer = Equivalence()

    # Test exact numerical match
    result = scorer(outputs=42, expectations={"expected_response": 42})
    assert result.name == "equivalence"
    assert result.value == CategoricalRating.YES
    assert "Exact numerical match" in result.rationale

    # Test exact string match
    result = scorer(outputs="Paris", expectations={"expected_response": "Paris"})
    assert result.value == CategoricalRating.YES
    assert "Exact string match" in result.rationale

    # Test numerical mismatch
    result = scorer(outputs=42, expectations={"expected_response": 43})
    assert result.value == CategoricalRating.NO
    assert "Values do not match" in result.rationale

    # Test with custom name and model
    scorer_custom = Equivalence(name="custom_output_equiv", model="openai:/gpt-4o-mini")
    result = scorer_custom(outputs=100, expectations={"expected_response": 100})
    assert result.name == "custom_output_equiv"
    assert result.value == CategoricalRating.YES


def test_get_all_scorers():
    scorers = get_all_scorers()
    scorer_class_names = {type(s).__name__ for s in scorers}

    expected_scorers = {
        "RetrievalRelevance",
        "RetrievalSufficiency",
        "RetrievalGroundedness",
        "ExpectationsGuidelines",
        "RelevanceToQuery",
        "Safety",
        "Correctness",
        "Fluency",
        "Equivalence",
        "Completeness",
        "Summarization",
        "UserFrustration",
        "ConversationCompleteness",
        "ConversationalSafety",
        "ConversationalToolCallEfficiency",
        "ConversationalRoleAdherence",
        "KnowledgeRetention",
        "ToolCallEfficiency",
        "ToolCallCorrectness",
    }

    assert scorer_class_names == expected_scorers
    assert all(isinstance(scorer, Scorer) for scorer in scorers)
    assert len({s.name for s in scorers}) == len(scorers)


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


def test_fluency_get_input_fields():
    fluency = Fluency(name="test")
    field_names = [field.name for field in fluency.get_input_fields()]
    assert field_names == ["outputs"]


@pytest.mark.usefixtures("mock_openai_env")
def test_fluency_default_name():
    mock_content = json.dumps(
        {
            "result": "yes",
            "rationale": "The text is fluent.",
        }
    )
    mock_response = ModelResponse(choices=[{"message": {"content": mock_content}}])

    with patch("litellm.completion", return_value=mock_response):
        scorer = Fluency()
        result = scorer(outputs="The cat sat on the mat.")

        assert result.name == "fluency"
        assert result.value == CategoricalRating.YES


@pytest.mark.usefixtures("mock_openai_env")
def test_fluency_with_custom_model():
    mock_content = json.dumps(
        {
            "result": "yes",
            "rationale": "The text is fluent.",
        }
    )
    mock_response = ModelResponse(choices=[{"message": {"content": mock_content}}])

    with patch("litellm.completion", return_value=mock_response):
        custom_model = "anthropic:/claude-3-opus"
        scorer = Fluency(model=custom_model)
        result = scorer(outputs="This is a fluent response")

        assert result.name == "fluency"
        assert result.value == CategoricalRating.YES


@pytest.mark.usefixtures("mock_openai_env")
def test_fluency_with_custom_name():
    mock_content = json.dumps(
        {
            "result": "no",
            "rationale": "The text has issues.",
        }
    )
    mock_response = ModelResponse(choices=[{"message": {"content": mock_content}}])

    with patch("litellm.completion", return_value=mock_response):
        scorer = Fluency(name="my_fluency_check")
        result = scorer(outputs="Bad text")

        assert result.name == "my_fluency_check"
        assert result.value == CategoricalRating.NO


def test_correctness_get_input_fields():
    correctness = Correctness(name="test")
    field_names = [field.name for field in correctness.get_input_fields()]
    assert field_names == ["inputs", "outputs", "expectations"]


def test_equivalence_get_input_fields():
    output_equiv = Equivalence(name="test")
    field_names = [field.name for field in output_equiv.get_input_fields()]
    assert field_names == ["outputs", "expectations"]


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


def test_equivalence_with_trace():
    trace = create_simple_trace(outputs="Paris")

    mlflow.log_expectation(
        trace_id=trace.info.trace_id,
        name="expected_response",
        value="Paris",
    )

    trace_with_expectations = mlflow.get_trace(trace.info.trace_id)

    scorer = Equivalence()
    result = scorer(trace=trace_with_expectations)

    assert result.name == "equivalence"
    assert result.value == CategoricalRating.YES
    assert "Exact string match" in result.rationale


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


def test_builtin_scorers_kind_property():
    # Test scorers returned by get_all_scorers() - they should all be BUILTIN
    for scorer in get_all_scorers():
        assert scorer.kind == ScorerKind.BUILTIN, (
            f"{scorer.__class__.__name__} should have kind BUILTIN, got {scorer.kind}"
        )

    # Test Guidelines explicitly - it has its own kind (not included in get_all_scorers)
    guidelines_scorer = Guidelines(
        name="test_guidelines",
        guidelines=["Be polite", "Be helpful"],
    )
    assert guidelines_scorer.kind == ScorerKind.GUIDELINES


def test_relevance_to_query_with_trace():
    with patch("mlflow.genai.judges.is_context_relevant") as mock_is_context_relevant:
        mock_is_context_relevant.return_value = Feedback(
            name="relevance_to_query", value="yes", rationale="Relevant"
        )

        trace = create_simple_trace()
        scorer = RelevanceToQuery()
        result = scorer(trace=trace)

        assert result.name == "relevance_to_query"
        assert result.value == CategoricalRating.YES
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
        assert result.value == CategoricalRating.YES
        mock_safety.assert_called_once()


def test_correctness_fallback_with_expectations(trace_without_inputs_outputs):
    patch_path = "mlflow.genai.scorers.builtin_scorers.get_chat_completions_with_structured_output"
    with patch(patch_path) as mock_extract:
        mock_extract.return_value = FieldExtraction(
            inputs='{"question": "extracted question"}', outputs="extracted answer"
        )

        with patch("mlflow.genai.judges.is_correct") as mock_is_correct:
            mock_is_correct.return_value = Feedback(
                name="correctness", value=True, rationale="Correct answer"
            )

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

            mock_extract.assert_called_once()
            call_args = mock_extract.call_args
            # Verify we're calling with the right trace
            assert call_args[1]["trace"] == trace_with_expectations

            mock_is_correct.assert_called_once()
            is_correct_args = mock_is_correct.call_args
            assert is_correct_args[1]["expected_response"] == "Expected answer"


def test_scorer_fallback_to_make_judge(trace_without_inputs_outputs):
    patch_path = "mlflow.genai.scorers.builtin_scorers.get_chat_completions_with_structured_output"

    with patch(patch_path) as mock_extract:
        mock_extract.return_value = FieldExtraction(
            inputs='{"question": "test"}', outputs="test response"
        )

        with patch("mlflow.genai.judges.meets_guidelines") as mock_meets_guidelines:
            mock_meets_guidelines.return_value = Feedback(
                name="guidelines", value=True, rationale="Follows guidelines"
            )

            scorer = Guidelines(guidelines=["Be helpful"])
            result = scorer(trace=trace_without_inputs_outputs)

            assert result.name == "guidelines"
            assert result.value is True

            mock_extract.assert_called_once()
            call_args = mock_extract.call_args
            # Verify we're calling with the right trace
            assert "trace" in call_args[1]

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
    patch_path = "mlflow.genai.scorers.builtin_scorers.get_chat_completions_with_structured_output"

    with patch(patch_path) as mock_extract:
        mock_extract.return_value = FieldExtraction(
            inputs='{"question": "test"}', outputs="test response"
        )

        with patch(f"mlflow.genai.judges.{judge_to_mock}") as mock_judge:
            mock_judge.return_value = Feedback(
                name=expected_name, value=True, rationale="Test passed"
            )

            scorer = scorer_factory()
            result = scorer(trace=trace_without_inputs_outputs)
            assert result.name == expected_name

            mock_extract.assert_called_once()
            call_args = mock_extract.call_args
            assert "trace" in call_args[1]
            assert call_args[1]["trace"] == trace_without_inputs_outputs

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


def test_equivalence_with_override_outputs():
    trace = create_simple_trace(outputs="Original output")
    scorer = Equivalence()

    result = scorer(
        trace=trace,
        outputs="Custom output",
        expectations={"expected_response": "Custom output"},
    )

    assert result.name == "equivalence"
    assert result.value == CategoricalRating.YES
    assert "Exact string match" in result.rationale


def test_relevance_mixed_override():
    with patch("mlflow.genai.judges.is_context_relevant") as mock_is_context_relevant:
        mock_is_context_relevant.return_value = Feedback(
            name="relevance_to_query", value="yes", rationale="Relevant"
        )

        trace = create_simple_trace()
        scorer = RelevanceToQuery()
        result = scorer(trace=trace, inputs={"question": "New question"})

        assert result.name == "relevance_to_query"
        assert result.value == CategoricalRating.YES
        mock_is_context_relevant.assert_called_once()
        call_args = mock_is_context_relevant.call_args
        assert call_args[1]["request"] == "{'question': 'New question'}"
        assert call_args[1]["context"] == "MLflow is an open-source platform for ML lifecycle."


def test_trace_agent_mode_with_extra_fields(trace_with_only_inputs):
    patch_path = "mlflow.genai.scorers.builtin_scorers.get_chat_completions_with_structured_output"

    with patch(patch_path) as mock_extract:
        mock_extract.return_value = FieldExtraction(
            inputs='{"question": "Test question"}', outputs="extracted outputs"
        )

        with patch("mlflow.genai.judges.is_safe") as mock_is_safe:
            mock_is_safe.return_value = Feedback(
                name="safety", value="yes", rationale="Safe via trace"
            )

            scorer = Safety()
            result = scorer(trace=trace_with_only_inputs)

            assert result.name == "safety"
            assert result.value == CategoricalRating.YES

            mock_extract.assert_called_once()
            call_args = mock_extract.call_args
            assert "trace" in call_args[1]
            assert call_args[1]["trace"] == trace_with_only_inputs


def test_pure_trace_mode_with_expectations(trace_with_only_outputs):
    patch_path = "mlflow.genai.scorers.builtin_scorers.get_chat_completions_with_structured_output"

    with patch(patch_path) as mock_extract:
        mock_extract.return_value = FieldExtraction(
            inputs='{"question": "extracted question"}', outputs="Test response"
        )

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

            mock_extract.assert_called_once()
            call_args = mock_extract.call_args
            assert "trace" in call_args[1]
            assert call_args[1]["trace"] == trace_with_only_outputs


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


def test_equivalence_default_extracts_from_trace():
    trace = create_simple_trace(outputs="MLflow is a platform")

    mlflow.log_expectation(
        trace_id=trace.info.trace_id, name="expected_response", value="MLflow is a platform"
    )

    trace_with_expectations = mlflow.get_trace(trace.info.trace_id)

    scorer = Equivalence()
    result = scorer(trace=trace_with_expectations)

    assert result.name == "equivalence"
    assert result.value == CategoricalRating.YES
    assert "Exact string match" in result.rationale


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
    patch_path = "mlflow.genai.scorers.builtin_scorers.get_chat_completions_with_structured_output"

    with patch(patch_path) as mock_extract:
        mock_extract.return_value = FieldExtraction(
            inputs='{"question": "test"}', outputs="test response"
        )

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

            mock_extract.assert_called_once()
            call_args = mock_extract.call_args
            assert "trace" in call_args[1]
            assert call_args[1]["trace"] == trace_without_inputs_outputs


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


def test_resolve_scorer_fields_with_trace_extraction(mock_judge_with_inputs_outputs):
    trace = create_simple_trace()

    fields = resolve_scorer_fields(trace=trace, judge=mock_judge_with_inputs_outputs)
    assert fields.inputs == {"question": "What is MLflow?"}
    assert fields.outputs == "MLflow is an open-source platform for ML lifecycle."
    assert fields.expectations is None

    fields = resolve_scorer_fields(
        trace=trace,
        judge=mock_judge_with_inputs_outputs,
        inputs="override inputs",
        outputs="override outputs",
    )
    assert fields.inputs == "override inputs"
    assert fields.outputs == "override outputs"


def test_resolve_scorer_fields_with_expectations(mock_judge_with_inputs_outputs):
    trace = create_simple_trace()
    mlflow.log_expectation(
        trace_id=trace.info.trace_id, name="expected_response", value="MLflow is a tool"
    )
    trace_with_expectations = mlflow.get_trace(trace.info.trace_id)

    fields = resolve_scorer_fields(
        trace=trace_with_expectations,
        judge=mock_judge_with_inputs_outputs,
        extract_expectations=True,
    )
    assert fields.inputs == {"question": "What is MLflow?"}
    assert fields.outputs == "MLflow is an open-source platform for ML lifecycle."
    assert fields.expectations == {"expected_response": "MLflow is a tool"}


def test_resolve_scorer_fields_llm_fallback(
    trace_without_inputs_outputs, mock_judge_with_inputs_outputs
):
    patch_path = "mlflow.genai.scorers.builtin_scorers.get_chat_completions_with_structured_output"

    with patch(patch_path) as mock_extract:
        mock_result = mock.Mock()
        mock_result.inputs = "extracted input"
        mock_result.outputs = "extracted output"
        mock_extract.return_value = mock_result

        fields = resolve_scorer_fields(
            trace=trace_without_inputs_outputs,
            judge=mock_judge_with_inputs_outputs,
            model="openai:/gpt-4",
        )

        assert fields.inputs == "extracted input"
        assert fields.outputs == "extracted output"
        assert fields.expectations is None

        mock_extract.assert_called_once()
        call_args = mock_extract.call_args
        assert call_args[1]["model_uri"] == "openai:/gpt-4"
        assert call_args[1]["trace"] == trace_without_inputs_outputs


def test_resolve_scorer_fields_llm_fallback_with_invalid_json(
    trace_without_inputs_outputs, mock_judge_with_inputs_outputs
):
    patch_path = "mlflow.genai.scorers.builtin_scorers.get_chat_completions_with_structured_output"

    with patch(patch_path) as mock_extract:
        mock_extract.side_effect = Exception("Failed to extract")

        fields = resolve_scorer_fields(
            trace=trace_without_inputs_outputs, judge=mock_judge_with_inputs_outputs
        )

        assert fields.inputs is None
        assert fields.outputs is None
        assert fields.expectations is None


def test_resolve_scorer_fields_partial_extraction(mock_judge_with_inputs_outputs):
    with mlflow.start_span(name="partial_span") as span:
        span.set_inputs({"question": "test"})
    trace_with_partial = mlflow.get_trace(span.trace_id)

    patch_path = "mlflow.genai.scorers.builtin_scorers.get_chat_completions_with_structured_output"

    with patch(patch_path) as mock_extract:
        mock_result = mock.Mock()
        mock_result.inputs = '{"question": "test"}'
        mock_result.outputs = "llm extracted output"
        mock_extract.return_value = mock_result

        fields = resolve_scorer_fields(
            trace=trace_with_partial, judge=mock_judge_with_inputs_outputs
        )

        assert fields.inputs == {"question": "test"}
        assert fields.outputs == "llm extracted output"
        assert fields.expectations is None


@pytest.mark.parametrize(
    ("needs_inputs", "needs_outputs", "expected_in_content", "not_expected_in_content"),
    [
        (True, False, ['"inputs": The initial user request/question'], ["outputs"]),
        (False, True, ['"outputs": The final system response'], ["inputs"]),
        (
            True,
            True,
            ['"inputs": The initial user request/question', '"outputs": The final system response'],
            [],
        ),
    ],
)
def test_construct_field_extraction_config_messages(
    needs_inputs, needs_outputs, expected_in_content, not_expected_in_content
):
    config = _construct_field_extraction_config(
        needs_inputs=needs_inputs, needs_outputs=needs_outputs
    )

    assert len(config.messages) == 2
    assert config.messages[0].role == "system"
    assert config.messages[1].role == "user"

    for expected in expected_in_content:
        assert expected in config.messages[0].content

    for not_expected in not_expected_in_content:
        assert not_expected not in config.messages[0].content


@pytest.mark.parametrize(
    ("needs_inputs", "needs_outputs", "expected_fields", "expected_descriptions"),
    [
        (
            True,
            False,
            {"inputs": True, "outputs": False},
            {"inputs": 'The user\'s original request (field name must be exactly "inputs")'},
        ),
        (
            False,
            True,
            {"inputs": False, "outputs": True},
            {"outputs": 'The system\'s final response (field name must be exactly "outputs")'},
        ),
        (
            True,
            True,
            {"inputs": True, "outputs": True},
            {
                "inputs": 'The user\'s original request (field name must be exactly "inputs")',
                "outputs": 'The system\'s final response (field name must be exactly "outputs")',
            },
        ),
    ],
)
def test_construct_field_extraction_config_schema(
    needs_inputs, needs_outputs, expected_fields, expected_descriptions
):
    config = _construct_field_extraction_config(
        needs_inputs=needs_inputs, needs_outputs=needs_outputs
    )

    schema_fields = config.schema.model_fields

    for field_name, should_exist in expected_fields.items():
        if should_exist:
            assert field_name in schema_fields
        else:
            assert field_name not in schema_fields

    for field_name, description in expected_descriptions.items():
        assert schema_fields[field_name].description == description


def test_construct_field_extraction_config_structure():
    config = _construct_field_extraction_config(needs_inputs=True, needs_outputs=True)

    assert config.messages[0].content.startswith("Extract the following fields from the trace.")
    assert "Use the provided tools to examine the trace's spans" in config.messages[0].content
    assert (
        "IMPORTANT: Return the result as JSON with the EXACT field names"
        in config.messages[0].content
    )
    expected_user_message = (
        "Use the tools to find the required fields, then return them as JSON "
        "with the exact field names specified."
    )
    assert config.messages[1].content == expected_user_message


def test_validate_required_fields_single_missing():
    from mlflow.exceptions import MlflowException

    judge = mock.Mock()
    judge.get_input_fields.return_value = [JudgeField(name="inputs", description="Test inputs")]

    fields = ExtractedFields(inputs=None, outputs="some output")

    with pytest.raises(MlflowException, match=r"TestScorer requires the following fields: inputs"):
        _validate_required_fields(fields, judge, "TestScorer")


def test_validate_required_fields_multiple_missing():
    from mlflow.exceptions import MlflowException

    judge = mock.Mock()
    judge.get_input_fields.return_value = [
        JudgeField(name="inputs", description="Test inputs"),
        JudgeField(name="outputs", description="Test outputs"),
        JudgeField(name="expectations", description="Test expectations"),
    ]

    fields = ExtractedFields(inputs=None, outputs=None, expectations=None)

    with pytest.raises(
        MlflowException,
        match=r"TestScorer requires the following fields: inputs, outputs, expectations",
    ):
        _validate_required_fields(fields, judge, "TestScorer")


def test_validate_required_fields_all_present():
    judge = mock.Mock()
    judge.get_input_fields.return_value = [
        JudgeField(name="inputs", description="Test inputs"),
        JudgeField(name="outputs", description="Test outputs"),
    ]

    fields = ExtractedFields(inputs="some input", outputs="some output")

    _validate_required_fields(fields, judge, "TestScorer")


def test_user_frustration_with_session():
    session_id = "test_session_123"
    traces = []
    for i in range(3):
        with mlflow.start_span(name=f"conversation_turn_{i}") as span:
            span.set_inputs({"question": f"User question {i}"})
            span.set_outputs(f"AI response {i}")
            mlflow.update_current_trace(metadata={TraceMetadataKey.TRACE_SESSION: session_id})
        traces.append(mlflow.get_trace(span.trace_id))

    with patch(
        "mlflow.genai.judges.instructions_judge.invoke_judge_model",
        return_value=Feedback(name="user_frustration", value="none", rationale="User is satisfied"),
    ) as mock_invoke_judge:
        scorer = UserFrustration()
        result = scorer(session=traces)

        assert result.name == "user_frustration"
        assert result.value == "none"
        assert result.rationale == "User is satisfied"
        mock_invoke_judge.assert_called_once()


def test_user_frustration_with_custom_name_and_model(monkeypatch: pytest.MonkeyPatch):
    session_id = "test_session_456"
    traces = []
    with mlflow.start_span(name="test_turn") as span:
        span.set_inputs({"question": "Test question"})
        span.set_outputs("Test response")
        mlflow.update_current_trace(metadata={TraceMetadataKey.TRACE_SESSION: session_id})
    traces.append(mlflow.get_trace(span.trace_id))

    with patch(
        "mlflow.genai.judges.instructions_judge.invoke_judge_model",
        return_value=Feedback(
            name="custom_frustration_check",
            value="resolved",
            rationale="User was initially frustrated but satisfied by the end",
        ),
    ) as mock_invoke_judge:
        scorer = UserFrustration(name="custom_frustration_check", model="openai:/gpt-4")
        result = scorer(session=traces)

        assert result.name == "custom_frustration_check"
        assert result.value == "resolved"
        mock_invoke_judge.assert_called_once()


@pytest.mark.parametrize(
    ("name", "model", "expected_name"),
    [
        (None, None, "conversation_completeness"),
        ("custom_completion_check", "openai:/gpt-4", "custom_completion_check"),
    ],
)
def test_conversation_completeness_with_session(name, model, expected_name):
    session_id = "test_session_789"
    traces = []
    for i in range(3):
        with mlflow.start_span(name=f"conversation_turn_{i}") as span:
            span.set_inputs({"question": f"User question {i}"})
            span.set_outputs(f"AI response {i}")
            mlflow.update_current_trace(metadata={TraceMetadataKey.TRACE_SESSION: session_id})
        traces.append(mlflow.get_trace(span.trace_id))

    with patch(
        "mlflow.genai.judges.instructions_judge.invoke_judge_model",
        return_value=Feedback(name=expected_name, value="yes", rationale="All needs addressed"),
    ) as mock_invoke_judge:
        kwargs = {}
        if name:
            kwargs["name"] = name
        if model:
            kwargs["model"] = model
        scorer = ConversationCompleteness(**kwargs)
        result = scorer(session=traces)

        assert result.name == expected_name
        assert result.value == "yes"
        assert result.rationale == "All needs addressed"
        mock_invoke_judge.assert_called_once()


@pytest.mark.parametrize(
    ("name", "model", "expected_name", "rationale"),
    [
        (None, None, "completeness", "All questions answered"),
        ("custom_completeness_check", "openai:/gpt-4", "custom_completeness_check", "All good"),
    ],
)
def test_completeness_with_inputs_outputs(name, model, expected_name, rationale):
    with patch(
        "mlflow.genai.judges.instructions_judge.invoke_judge_model",
        return_value=Feedback(name=expected_name, value="yes", rationale=rationale),
    ) as mock_invoke_judge:
        kwargs = {}
        if name:
            kwargs["name"] = name
        if model:
            kwargs["model"] = model
        scorer = Completeness(**kwargs)
        result = scorer(
            inputs={"question": "What is MLflow?"},
            outputs="MLflow is an open-source platform for managing the ML lifecycle.",
        )

        assert result.name == expected_name
        assert result.value == "yes"
        assert result.rationale == rationale
        mock_invoke_judge.assert_called_once()


def test_completeness_with_trace():
    with patch(
        "mlflow.genai.judges.instructions_judge.invoke_judge_model",
        return_value=Feedback(name="completeness", value="yes", rationale="Fully addressed"),
    ) as mock_invoke_judge:
        trace = create_simple_trace()
        scorer = Completeness()
        result = scorer(trace=trace)

        assert result.name == "completeness"
        assert result.value == "yes"
        assert result.rationale == "Fully addressed"
        mock_invoke_judge.assert_called_once()


def test_conversational_safety_with_session():
    session_id = "test_session_safety"
    traces = []
    for i, (q, a) in enumerate(
        [
            ("What is Python?", "Python is a programming language."),
            ("How do I install it?", "You can download it from python.org."),
        ]
    ):
        with mlflow.start_span(name=f"turn_{i}") as span:
            span.set_inputs({"question": q})
            span.set_outputs(a)
            mlflow.update_current_trace(metadata={TraceMetadataKey.TRACE_SESSION: session_id})
        traces.append(mlflow.get_trace(span.trace_id))

    with patch(
        "mlflow.genai.judges.instructions_judge.invoke_judge_model",
        return_value=Feedback(
            name="conversational_safety",
            value="yes",
            rationale="Safe conversation across session",
        ),
    ) as mock_invoke_judge:
        scorer = ConversationalSafety()
        result = scorer(session=traces)

        assert result.name == "conversational_safety"
        assert result.value == "yes"
        assert result.rationale == "Safe conversation across session"
        mock_invoke_judge.assert_called_once()


@pytest.mark.parametrize(
    ("name", "model", "expected_name"),
    [
        (None, None, "conversational_safety"),
        ("custom_safety_check", "openai:/gpt-4", "custom_safety_check"),
    ],
)
def test_conversational_safety_with_custom_name_and_model(name, model, expected_name):
    session_id = "test_session_safety_custom"
    traces = []
    with mlflow.start_span(name="test_turn") as span:
        span.set_inputs({"question": "Test question"})
        span.set_outputs("Test response")
        mlflow.update_current_trace(metadata={TraceMetadataKey.TRACE_SESSION: session_id})
    traces.append(mlflow.get_trace(span.trace_id))

    with patch(
        "mlflow.genai.judges.instructions_judge.invoke_judge_model",
        return_value=Feedback(name=expected_name, value="yes", rationale="Conversation is safe"),
    ) as mock_invoke_judge:
        kwargs = {}
        if name:
            kwargs["name"] = name
        if model:
            kwargs["model"] = model
        scorer = ConversationalSafety(**kwargs)
        result = scorer(session=traces)

        assert result.name == expected_name
        assert result.value == "yes"
        assert result.rationale == "Conversation is safe"
        mock_invoke_judge.assert_called_once()


def test_conversational_safety_get_input_fields():
    scorer = ConversationalSafety()
    fields = scorer.get_input_fields()
    field_names = [field.name for field in fields]
    assert field_names == ["session"]


def test_conversational_safety_instructions():
    scorer = ConversationalSafety()
    instructions = scorer.instructions
    assert "conversation" in instructions.lower()
    assert "assistant" in instructions.lower()
    assert "safety" in instructions.lower()


def test_conversational_tool_call_efficiency_with_session():
    session_id = "test_session_efficiency"
    traces = []
    for i, (question, stock, stock_price) in enumerate(
        [
            ("What is the price of AAPL?", "AAPL", "150"),
            ("How about MSFT?", "MSFT", "300"),
        ]
    ):
        answer = f"{stock} is ${stock_price}."
        with mlflow.start_span(name=f"turn_{i}") as span:
            span.set_inputs({"question": question})
            span.set_outputs(answer)
            mlflow.update_current_trace(metadata={TraceMetadataKey.TRACE_SESSION: session_id})

            with mlflow.start_span(name="get_stock_price", span_type=SpanType.TOOL) as tool_span:
                tool_span.set_inputs({"symbol": stock})
                tool_span.set_outputs(f"${stock_price}")

        traces.append(mlflow.get_trace(span.trace_id))

    mock_feedback = Feedback(
        name="conversational_tool_call_efficiency",
        value=CategoricalRating.YES,
        rationale="Efficient tool usage across session",
    )

    with patch(
        "mlflow.genai.judges.instructions_judge.invoke_judge_model",
        return_value=mock_feedback,
    ) as mock_invoke:
        scorer = ConversationalToolCallEfficiency()
        result = scorer(session=traces)

        assert result.name == "conversational_tool_call_efficiency"
        assert result.value == CategoricalRating.YES
        mock_invoke.assert_called_once()


def test_conversational_tool_call_efficiency_get_input_fields():
    scorer = ConversationalToolCallEfficiency()
    fields = scorer.get_input_fields()
    field_names = [field.name for field in fields]
    assert field_names == ["session"]


def test_conversational_tool_call_efficiency_instructions():
    scorer = ConversationalToolCallEfficiency()
    instructions = scorer.instructions
    assert "tool" in instructions.lower()
    assert "efficient" in instructions.lower() or "redundant" in instructions.lower()


def test_tool_call_efficiency():
    with mlflow.start_span(name="agent") as span:
        span.set_inputs({"question": "What is the weather in Paris?"})
        with mlflow.start_span(name="get_weather", span_type=SpanType.TOOL) as tool_span:
            tool_span.set_inputs({"city": "Paris"})
            tool_span.set_outputs("Sunny, 22°C")
        span.set_outputs("The weather in Paris is sunny and 22°C")

    efficient_trace = mlflow.get_trace(span.trace_id)

    with mlflow.start_span(name="agent") as span:
        span.set_inputs({"question": "Get weather for InvalidCity"})
        with mlflow.start_span(name="get_weather", span_type=SpanType.TOOL) as tool_span:
            tool_span.set_inputs({"city": "InvalidCity123"})
            tool_span.record_exception(ValueError("City not found"))
        span.set_outputs("Sorry, I couldn't find that city")

    exception_trace = mlflow.get_trace(span.trace_id)

    with patch("mlflow.genai.judges.builtin.invoke_judge_model") as mock_invoke:
        mock_invoke.return_value = Feedback(
            name="tool_call_efficiency",
            value=CategoricalRating.YES,
            rationale="Tool usage is efficient",
        )

        scorer = ToolCallEfficiency()
        result = scorer(trace=efficient_trace)

        assert result.name == "tool_call_efficiency"
        assert result.value == CategoricalRating.YES
        mock_invoke.assert_called()

        result = scorer(trace=exception_trace)
        assert result.name == "tool_call_efficiency"
        mock_invoke.assert_called()


def test_tool_call_correctness_with_correct_tool_call():
    with mlflow.start_span(name="agent") as span:
        span.set_inputs({"question": "What is the weather in San Francisco?"})
        with mlflow.start_span(name="get_weather", span_type=SpanType.TOOL) as tool_span:
            tool_span.set_inputs({"location": "San Francisco", "unit": "celsius"})
            tool_span.set_outputs("15°C, partly cloudy")
        span.set_outputs("The weather in San Francisco is 15°C and partly cloudy")

    correct_trace = mlflow.get_trace(span.trace_id)

    with patch("mlflow.genai.judges.builtin.invoke_judge_model") as mock_invoke:
        mock_invoke.return_value = Feedback(
            name="tool_call_correctness",
            value=CategoricalRating.YES,
            rationale="Tool calls are correct and appropriate",
        )

        scorer = ToolCallCorrectness()
        result = scorer(trace=correct_trace)

        assert result.name == "tool_call_correctness"
        assert result.value == CategoricalRating.YES
        mock_invoke.assert_called()


def test_tool_call_correctness_with_incorrect_tool_call():
    with mlflow.start_span(name="agent") as span:
        span.set_inputs({"question": "What is the weather in San Francisco?"})
        with mlflow.start_span(name="calculator", span_type=SpanType.TOOL) as tool_span:
            tool_span.set_inputs({"expression": "San Francisco"})
            tool_span.set_outputs("Error: invalid expression")
        span.set_outputs("I'm sorry, I couldn't get the weather")

    incorrect_trace = mlflow.get_trace(span.trace_id)

    with patch("mlflow.genai.judges.builtin.invoke_judge_model") as mock_invoke:
        mock_invoke.return_value = Feedback(
            name="tool_call_correctness",
            value=CategoricalRating.NO,
            rationale="Wrong tool used for weather query",
        )

        scorer = ToolCallCorrectness()
        result = scorer(trace=incorrect_trace)

        assert result.name == "tool_call_correctness"
        assert result.value == CategoricalRating.NO
        mock_invoke.assert_called()


def test_conversational_role_adherence_with_session():
    session_id = "test_session_role"
    traces = []
    for i, (question, answer) in enumerate(
        [
            ("What can you cook?", "I can help you make many dishes!"),
            ("How do I make soup?", "Start by boiling vegetables..."),
        ]
    ):
        with mlflow.start_span(name=f"turn_{i}") as span:
            span.set_inputs({"question": question})
            span.set_outputs(answer)
            mlflow.update_current_trace(metadata={TraceMetadataKey.TRACE_SESSION: session_id})
        traces.append(mlflow.get_trace(span.trace_id))

    with patch(
        "mlflow.genai.judges.instructions_judge.invoke_judge_model",
        return_value=Feedback(
            name="conversational_role_adherence",
            value=CategoricalRating.YES,
            rationale="Role maintained across session.",
        ),
    ) as mock_invoke_judge:
        scorer = ConversationalRoleAdherence()
        result = scorer(session=traces)

        assert result.name == "conversational_role_adherence"
        assert result.value == CategoricalRating.YES
        mock_invoke_judge.assert_called_once()


def test_conversational_role_adherence_get_input_fields():
    scorer = ConversationalRoleAdherence()
    fields = scorer.get_input_fields()
    field_names = [field.name for field in fields]
    assert field_names == ["session"]


def test_conversational_role_adherence_instructions():
    scorer = ConversationalRoleAdherence()
    instructions = scorer.instructions
    assert "role" in instructions.lower()
    assert "persona" in instructions.lower() or "boundaries" in instructions.lower()


def _create_test_trace_with_session(
    turn_name: str,
    session_id: str,
    input_data: dict[str, str],
    output_data: str,
):
    """Helper to create a trace for KnowledgeRetention tests."""
    with mlflow.start_span(name=turn_name) as span:
        span.set_inputs(input_data)
        span.set_outputs(output_data)
        mlflow.update_current_trace(metadata={TraceMetadataKey.TRACE_SESSION: session_id})
    return mlflow.get_trace(span.trace_id)


def test_knowledge_retention_uses_default_last_turn_scorer():
    session_id = "test_session_default_scorer"
    session = []

    session.append(
        _create_test_trace_with_session(
            turn_name="turn_0",
            session_id=session_id,
            input_data={"question": "My name is Alice and I love Python"},
            output_data="Nice to meet you Alice! Python is great.",
        )
    )

    session.append(
        _create_test_trace_with_session(
            turn_name="turn_1",
            session_id=session_id,
            input_data={"question": "What programming language do I like?"},
            output_data="You mentioned you love Python!",
        )
    )

    with patch(
        "mlflow.genai.judges.instructions_judge.invoke_judge_model",
        return_value=Feedback(
            name="last_turn_knowledge_retention",
            value="yes",
            rationale="AI correctly recalled the user loves Python",
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id="test-model",
            ),
        ),
    ) as mock_invoke_judge:
        scorer = KnowledgeRetention()
        result = scorer(session=session)

        assert isinstance(result, Feedback)
        assert result.value == "yes"
        assert "successful" in result.rationale.lower()

        assert mock_invoke_judge.call_count == 2

        first_call_kwargs = mock_invoke_judge.call_args_list[0].kwargs
        assert "prompt" in first_call_kwargs
        conversation_content = first_call_kwargs["prompt"][1].content
        assert "My name is Alice and I love Python" in conversation_content


def test_knowledge_retention_success():
    session_id = "test_session_kr_success"
    traces = []

    traces.append(
        _create_test_trace_with_session(
            turn_name="turn_0",
            session_id=session_id,
            input_data={"question": "My name is Alice and I love Python"},
            output_data="Nice to meet you Alice! Python is great.",
        )
    )

    traces.append(
        _create_test_trace_with_session(
            turn_name="turn_1",
            session_id=session_id,
            input_data={"question": "What programming language do I like?"},
            output_data="You mentioned you love Python!",
        )
    )

    fake_scorer = Mock(spec=Scorer)
    fake_scorer.return_value = Feedback(
        name="last_turn_knowledge_retention",
        value="yes",
        rationale="AI correctly recalled the user loves Python",
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id="test-model",
        ),
    )

    scorer = KnowledgeRetention(last_turn_scorer=fake_scorer)
    result = scorer(session=traces)

    assert isinstance(result, Feedback)
    assert result.value == "yes"
    assert "successful" in result.rationale.lower()

    assert fake_scorer.call_count == 2

    first_call_args = fake_scorer.call_args_list[0]
    assert len(first_call_args.kwargs["session"]) == 1

    second_call_args = fake_scorer.call_args_list[1]
    assert len(second_call_args.kwargs["session"]) == 2


def test_knowledge_retention_failure():
    session_id = "test_session_kr_failure"
    traces = []

    traces.append(
        _create_test_trace_with_session(
            turn_name="turn_0",
            session_id=session_id,
            input_data={"question": "My name is Alice"},
            output_data="Nice to meet you Alice!",
        )
    )

    traces.append(
        _create_test_trace_with_session(
            turn_name="turn_1",
            session_id=session_id,
            input_data={"question": "What's my name?"},
            output_data="Your name is Bob!",
        )
    )

    fake_scorer = Mock(spec=Scorer)
    fake_scorer.return_value = Feedback(
        name="last_turn_knowledge_retention",
        value="no",
        rationale="AI incorrectly recalled name as Bob instead of Alice",
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id="test-model",
        ),
    )

    scorer = KnowledgeRetention(last_turn_scorer=fake_scorer)
    result = scorer(session=traces)

    assert isinstance(result, Feedback)
    assert result.value == "no"
    assert "failed" in result.rationale.lower() or "no" in result.rationale.lower()

    assert "Turn 1" in result.rationale

    assert "AI incorrectly recalled name as Bob instead of Alice" in result.rationale


def test_knowledge_retention_single_turn():
    session_id = "test_session_single"
    traces = []

    traces.append(
        _create_test_trace_with_session(
            turn_name="turn_0",
            session_id=session_id,
            input_data={"question": "Hello"},
            output_data="Hi there!",
        )
    )

    fake_scorer = Mock(spec=Scorer)
    fake_scorer.return_value = Feedback(
        name="last_turn_knowledge_retention",
        value="yes",
        rationale="Single turn evaluated - no prior context to contradict",
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id="test-model",
        ),
    )

    scorer = KnowledgeRetention(last_turn_scorer=fake_scorer)
    result = scorer(session=traces)

    assert isinstance(result, Feedback)
    assert result.value == "yes"
    assert "successful" in result.rationale.lower()

    fake_scorer.assert_called_once()
    call_args = fake_scorer.call_args
    assert len(call_args.kwargs["session"]) == 1


def test_knowledge_retention_empty_session():
    scorer = KnowledgeRetention()
    with pytest.raises(
        MlflowException, match="cannot evaluate knowledge retention on empty session"
    ):
        scorer(session=[])


def test_session_level_scorer_with_invalid_kwargs():
    scorer = UserFrustration()

    with pytest.raises(
        TypeError,
        match=r"Session level scorers can only accept the `session` and `expectations` "
        r"parameters\. Got unexpected keyword argument\(s\): 'trace'",
    ):
        scorer(trace=create_simple_trace())

    with pytest.raises(
        TypeError,
        match=r"Session level scorers can only accept the `session` and `expectations` "
        r"parameters\. Got unexpected keyword argument\(s\): 'outputs'",
    ):
        scorer(outputs="some output")


@pytest.mark.parametrize(
    ("name", "model", "expected_name", "rationale"),
    [
        (None, None, "summarization", "Good summary"),
        ("custom_summarization_check", "openai:/gpt-4", "custom_summarization_check", "Excellent"),
    ],
)
def test_summarization_with_inputs_outputs(name, model, expected_name, rationale):
    with patch(
        "mlflow.genai.judges.instructions_judge.invoke_judge_model",
        return_value=Feedback(name=expected_name, value="yes", rationale=rationale),
    ) as mock_invoke_judge:
        kwargs = {}
        if name:
            kwargs["name"] = name
        if model:
            kwargs["model"] = model
        scorer = Summarization(**kwargs)
        result = scorer(
            inputs={
                "text": (
                    "MLflow is an open-source platform for managing the end-to-end machine "
                    "learning lifecycle. It provides tools for experiment tracking, model "
                    "packaging, and deployment."
                )
            },
            outputs="MLflow is an ML lifecycle management platform.",
        )

        assert result.name == expected_name
        assert result.value == "yes"
        assert result.rationale == rationale
        mock_invoke_judge.assert_called_once()


def test_summarization_with_trace():
    with patch(
        "mlflow.genai.judges.instructions_judge.invoke_judge_model",
        return_value=Feedback(name="summarization", value="yes", rationale="Accurate summary"),
    ) as mock_invoke_judge:
        trace = create_simple_trace(
            inputs={"question": "Summarize MLflow"},
            outputs="MLflow is an open-source platform for managing ML workflows.",
        )
        scorer = Summarization()
        result = scorer(trace=trace)

        assert result.name == "summarization"
        assert result.value == "yes"
        assert result.rationale == "Accurate summary"
        mock_invoke_judge.assert_called_once()
