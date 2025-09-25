import json
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

import mlflow
from mlflow.entities.assessment import Expectation, Feedback
from mlflow.entities.assessment_error import AssessmentError
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.span import Span, SpanType
from mlflow.entities.trace import Trace, TraceInfo
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
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
from mlflow.genai.utils.trace_utils import (
    extract_expectations_from_trace,
    extract_inputs_from_trace,
    extract_outputs_from_trace,
)
from mlflow.tracing.utils import build_otel_context
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
    assert field_names == ["trace", "inputs", "outputs"]


def test_expectations_guidelines_get_input_fields():
    """Test that ExpectationsGuidelines get_input_fields method returns expected field names."""
    exp_guidelines = ExpectationsGuidelines(name="test")
    field_names = [field.name for field in exp_guidelines.get_input_fields()]
    assert field_names == ["inputs", "outputs", "expectations"]


def test_relevance_to_query_get_input_fields():
    """Test that RelevanceToQuery get_input_fields method returns expected field names."""
    relevance_query = RelevanceToQuery(name="test")
    field_names = [field.name for field in relevance_query.get_input_fields()]
    assert field_names == ["trace", "inputs", "outputs"]


def test_safety_get_input_fields():
    """Test that Safety get_input_fields method returns expected field names."""
    if is_databricks_uri(mlflow.get_tracking_uri()):
        safety = Safety(name="test")
        field_names = [field.name for field in safety.get_input_fields()]
        assert field_names == ["outputs"]


def test_correctness_get_input_fields():
    correctness = Correctness(name="test")
    field_names = [field.name for field in correctness.get_input_fields()]
    assert field_names == ["trace", "inputs", "outputs", "expectations"]


from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

from mlflow.exceptions import MlflowException
from mlflow.tracing.utils import build_otel_context


@pytest.fixture
def mock_trace():
    root_span = Span(
        OTelReadableSpan(
            name="root",
            context=build_otel_context(trace_id=12345, span_id=67890),
            parent=None,
            start_time=0,
            end_time=100,
            attributes={
                "mlflow.spanInputs": json.dumps({"question": "What is MLflow?"}),
                "mlflow.spanOutputs": json.dumps(
                    {"answer": "MLflow is an open source platform for ML lifecycle"}
                ),
            },
        )
    )

    trace_info = TraceInfo(
        trace_id="test-trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=0,
        execution_duration=100,
        state=TraceState.OK,
    )

    trace_data = TraceData(spans=[root_span])
    return Trace(info=trace_info, data=trace_data)


@pytest.fixture
def mock_trace_with_expectations(mock_trace):
    from mlflow.entities.assessment import Expectation

    trace = mock_trace
    # Add expectation assessment
    expectation = Expectation(
        name="expected_response",
        value="MLflow is a platform for ML lifecycle",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
    )
    # Mock the search_assessments method to return expectations
    trace.search_assessments = MagicMock(return_value=[expectation])
    return trace


def test_correctness_missing_requirements_error():
    scorer = Correctness()

    # Should raise error when neither trace nor required fields are provided
    with pytest.raises(MlflowException, match="Correctness scorer requires either"):
        scorer()

    # Should raise error when trace is None and fields are incomplete
    with pytest.raises(MlflowException, match="Correctness scorer requires either"):
        scorer(inputs={"q": "test"})  # Missing outputs and expectations


# ============================================================================
# Real Trace Tests for Field Extraction and Override Behavior
# ============================================================================


@pytest.fixture
def simple_qa_trace() -> Trace:
    @mlflow.trace(name="qa_chain", span_type=SpanType.CHAIN)
    def qa_function(question: str) -> str:
        return f"The answer to '{question}' is 42"

    qa_function("What is the meaning of life?")
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    trace.info.assessments = [
        Expectation(
            name="expected_response",
            value="The answer is 42",
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
        ),
        Expectation(
            name="expected_facts",
            value=["42 is the answer", "from Hitchhiker's Guide"],
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
        ),
    ]
    return trace


@pytest.fixture
def chat_trace() -> Trace:
    @mlflow.trace(name="chat_bot", span_type=SpanType.CHAT_MODEL)
    def chat_function(messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "Paris is the capital of France."}}
            ]
        }

    messages = [{"role": "user", "content": "What is the capital of France?"}]
    chat_function(messages)
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    trace.info.assessments = [
        Expectation(
            name="expected_response",
            value="The capital of France is Paris",
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
        ),
    ]
    return trace


@pytest.fixture
def nested_trace() -> Trace:
    @mlflow.trace(name="main_chain", span_type=SpanType.CHAIN)
    def main_function(query: str) -> str:
        preprocessed = preprocess(query)
        result = process(preprocessed)
        return postprocess(result)

    @mlflow.trace(name="preprocess", span_type=SpanType.CHAIN)
    def preprocess(text: str) -> str:
        return text.lower().strip()

    @mlflow.trace(name="process", span_type=SpanType.LLM)
    def process(text: str) -> str:
        return f"Processed: {text}"

    @mlflow.trace(name="postprocess", span_type=SpanType.CHAIN)
    def postprocess(text: str) -> str:
        return text.upper()

    main_function("Tell me about MLflow")
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    trace.info.assessments = [
        Expectation(
            name="expected_facts",
            value=["MLflow is a platform", "manages ML lifecycle"],
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
        ),
        Expectation(
            name="guidelines",
            value=["Be concise", "Use technical terms appropriately"],
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
        ),
    ]
    return trace


@pytest.fixture
def trace_without_expectations() -> Trace:
    @mlflow.trace(name="simple", span_type=SpanType.CHAIN)
    def simple_function(input_text: str) -> str:
        return f"Output: {input_text}"

    simple_function("test input")
    return mlflow.get_trace(mlflow.get_last_active_trace_id())


@pytest.fixture
def complex_io_trace() -> Trace:
    @mlflow.trace(name="complex", span_type=SpanType.CHAIN)
    def complex_function(data: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "success",
            "results": [
                {"id": 1, "value": data.get("query")},
                {"id": 2, "value": "additional data"},
            ],
            "metadata": {"processed": True},
        }

    complex_function({"query": "complex query", "filters": ["a", "b"]})
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    trace.info.assessments = [
        Expectation(
            name="expected_response",
            value=json.dumps({"status": "success", "results": ["some", "data"]}),
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
        ),
    ]
    return trace


def test_correctness_extraction_from_simple_trace(simple_qa_trace):
    scorer = Correctness()

    with patch("mlflow.genai.judges.is_correct") as mock_is_correct:
        mock_is_correct.return_value = Feedback(
            name="correctness", value=True, rationale="Correct based on trace"
        )

        result = scorer(trace=simple_qa_trace)

        mock_is_correct.assert_called_once()
        call_args = mock_is_correct.call_args[1]

        assert "What is the meaning of life?" in call_args["request"]
        assert "The answer to 'What is the meaning of life?' is 42" in call_args["response"]
        assert call_args["expected_response"] == "The answer is 42"
        assert call_args["expected_facts"] == ["42 is the answer", "from Hitchhiker's Guide"]
        assert result.value is True


def test_correctness_with_override_outputs(simple_qa_trace):
    scorer = Correctness()

    with patch("mlflow.genai.judges.is_correct") as mock_is_correct:
        mock_is_correct.return_value = Feedback(
            name="correctness", value=False, rationale="Override output incorrect"
        )

        custom_output = "This is a custom wrong answer"
        result = scorer(trace=simple_qa_trace, outputs=custom_output)

        call_args = mock_is_correct.call_args[1]
        assert call_args["response"] == custom_output
        assert "What is the meaning of life?" in call_args["request"]
        assert call_args["expected_response"] == "The answer is 42"
        assert result.value is False


def test_correctness_with_override_expectations(simple_qa_trace):
    scorer = Correctness()

    with patch("mlflow.genai.judges.is_correct") as mock_is_correct:
        mock_is_correct.return_value = Feedback(
            name="correctness", value=True, rationale="Custom expectations"
        )

        custom_expectations = {
            "expected_response": "Custom expected answer",
            "expected_facts": ["custom", "facts"],
        }
        scorer(trace=simple_qa_trace, expectations=custom_expectations)

        call_args = mock_is_correct.call_args[1]
        assert call_args["expected_response"] == "Custom expected answer"
        assert call_args["expected_facts"] == ["custom", "facts"]
        assert "What is the meaning of life?" in call_args["request"]
        assert "The answer to 'What is the meaning of life?' is 42" in call_args["response"]


def test_correctness_llm_fallback_for_missing_expectations(trace_without_expectations):
    scorer = Correctness()

    with patch.object(scorer, "_extract_missing_fields_with_llm") as mock_extract:
        mock_extract.return_value = {
            "expected_response": "LLM extracted response",
            "expected_facts": ["LLM", "extracted", "facts"],
        }

        with patch("mlflow.genai.judges.is_correct") as mock_is_correct:
            mock_is_correct.return_value = Feedback(
                name="correctness", value=True, rationale="LLM extraction worked"
            )

            scorer(trace=trace_without_expectations)
            mock_extract.assert_called_once()

            call_args = mock_is_correct.call_args[1]
            assert call_args["expected_response"] == "LLM extracted response"
            assert call_args["expected_facts"] == ["LLM", "extracted", "facts"]


def test_correctness_with_chat_trace(chat_trace):
    scorer = Correctness()

    with patch("mlflow.genai.judges.is_correct") as mock_is_correct:
        mock_is_correct.return_value = Feedback(
            name="correctness", value=True, rationale="Chat format handled"
        )

        scorer(trace=chat_trace)
        call_args = mock_is_correct.call_args[1]

        assert "What is the capital of France?" in call_args["request"]
        assert "Paris is the capital of France" in call_args["response"]
        assert call_args["expected_response"] == "The capital of France is Paris"


def test_guidelines_extraction_from_trace(simple_qa_trace):
    scorer = Guidelines(guidelines=["Be helpful", "Be accurate"])

    with patch("mlflow.genai.judges.meets_guidelines") as mock_meets:
        mock_meets.return_value = Feedback(name="guidelines", value=True)

        result = scorer(trace=simple_qa_trace)
        call_args = mock_meets.call_args[1]

        assert "What is the meaning of life?" in call_args["context"]["request"]
        assert (
            "The answer to 'What is the meaning of life?' is 42" in call_args["context"]["response"]
        )
        assert call_args["guidelines"] == ["Be helpful", "Be accurate"]
        assert result.value is True


def test_guidelines_with_override_inputs(simple_qa_trace):
    scorer = Guidelines(guidelines=["Test guideline"])

    with patch("mlflow.genai.judges.meets_guidelines") as mock_meets:
        mock_meets.return_value = Feedback(name="guidelines", value=False)

        custom_inputs = {"question": "Custom question"}
        scorer(trace=simple_qa_trace, inputs=custom_inputs)

        call_args = mock_meets.call_args[1]
        assert "Custom question" in call_args["context"]["request"]
        assert (
            "The answer to 'What is the meaning of life?' is 42" in call_args["context"]["response"]
        )


def test_guidelines_with_nested_trace(nested_trace):
    scorer = Guidelines(guidelines=["Be technical"])

    with patch("mlflow.genai.judges.meets_guidelines") as mock_meets:
        mock_meets.return_value = Feedback(name="guidelines", value=True)

        scorer(trace=nested_trace)
        call_args = mock_meets.call_args[1]

        assert "Tell me about MLflow" in call_args["context"]["request"]
        assert "PROCESSED: TELL ME ABOUT MLFLOW" in call_args["context"]["response"]


def test_relevance_extraction_from_trace(simple_qa_trace):
    scorer = RelevanceToQuery()

    with patch("mlflow.genai.judges.is_context_relevant") as mock_relevant:
        mock_relevant.return_value = Feedback(name="relevance_to_query", value=True)

        result = scorer(trace=simple_qa_trace)
        call_args = mock_relevant.call_args[1]

        assert "What is the meaning of life?" in call_args["request"]
        assert "The answer to 'What is the meaning of life?' is 42" == call_args["context"]
        assert result.value is True


def test_relevance_with_complex_io(complex_io_trace):
    scorer = RelevanceToQuery()

    with patch("mlflow.genai.judges.is_context_relevant") as mock_relevant:
        mock_relevant.return_value = Feedback(name="relevance_to_query", value=True)

        scorer(trace=complex_io_trace)
        call_args = mock_relevant.call_args[1]

        assert "complex query" in call_args["request"]
        assert call_args["context"]["status"] == "success"
        assert "results" in call_args["context"]


def test_relevance_mixed_override(simple_qa_trace):
    scorer = RelevanceToQuery()

    with patch("mlflow.genai.judges.is_context_relevant") as mock_relevant:
        mock_relevant.return_value = Feedback(name="relevance_to_query", value=False)

        custom_outputs = "Irrelevant response about weather"
        result = scorer(trace=simple_qa_trace, outputs=custom_outputs)

        call_args = mock_relevant.call_args[1]
        assert call_args["context"] == custom_outputs
        assert "What is the meaning of life?" in call_args["request"]
        assert result.value is False


def test_safety_extraction_from_trace(simple_qa_trace):
    if not mlflow.get_tracking_uri().startswith("databricks"):
        pytest.skip("Safety scorer requires Databricks environment")

    scorer = Safety()

    with patch("databricks.agents.evals.judges.safety") as mock_safety:
        mock_safety.return_value = Feedback(name="safety", value=True)

        result = scorer(trace=simple_qa_trace)
        call_args = mock_safety.call_args[1]

        assert "The answer to 'What is the meaning of life?' is 42" in call_args["response"]
        assert result.value is True


def test_safety_with_override(simple_qa_trace):
    if not mlflow.get_tracking_uri().startswith("databricks"):
        pytest.skip("Safety scorer requires Databricks environment")

    scorer = Safety()

    with patch("databricks.agents.evals.judges.safety") as mock_safety:
        mock_safety.return_value = Feedback(name="safety", value=False)

        unsafe_output = "This contains unsafe content"
        result = scorer(trace=simple_qa_trace, outputs=unsafe_output)

        call_args = mock_safety.call_args[1]
        assert call_args["response"] == unsafe_output
        assert result.value is False


def test_expectations_guidelines_extraction(nested_trace):
    scorer = ExpectationsGuidelines()

    inputs = extract_inputs_from_trace(nested_trace)
    outputs = extract_outputs_from_trace(nested_trace)
    expectations = extract_expectations_from_trace(nested_trace)

    with patch("mlflow.genai.judges.meets_guidelines") as mock_meets:
        mock_meets.return_value = Feedback(name="expectations_guidelines", value=True)

        result = scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        call_args = mock_meets.call_args[1]

        assert call_args["guidelines"] == ["Be concise", "Use technical terms appropriately"]
        assert result.value is True


def test_scorer_with_malformed_trace():
    @mlflow.trace(name="malformed", span_type=SpanType.CHAIN)
    def malformed_function():
        pass

    malformed_function()
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    scorer = Guidelines(guidelines=["Some guideline"])

    with patch("mlflow.genai.judges.meets_guidelines") as mock_meets:
        mock_meets.return_value = Feedback(name="guidelines", value=True)

        scorer(trace=trace)
        call_args = mock_meets.call_args[1]

        assert "context" in call_args
        assert "request" in call_args["context"]
        assert "response" in call_args["context"]
        assert isinstance(call_args["context"]["request"], str)
        assert isinstance(call_args["context"]["response"], str)


def test_all_fields_provided_ignores_trace(simple_qa_trace):
    scorer = Correctness()

    direct_inputs = {"q": "Direct question"}
    direct_outputs = "Direct answer"
    direct_expectations = {"expected_response": "Direct expected"}

    with patch("mlflow.genai.judges.is_correct") as mock_is_correct:
        mock_is_correct.return_value = Feedback(name="correctness", value=True)

        scorer(
            trace=simple_qa_trace,
            inputs=direct_inputs,
            outputs=direct_outputs,
            expectations=direct_expectations,
        )

        call_args = mock_is_correct.call_args[1]

        assert "Direct question" in call_args["request"]
        assert call_args["response"] == "Direct answer"
        assert call_args["expected_response"] == "Direct expected"
        assert "What is the meaning of life?" not in call_args["request"]
        assert "42" not in call_args["response"]


def test_correctness_with_different_expectation_types():
    scorer = Correctness()

    with patch("mlflow.genai.judges.is_correct") as mock_is_correct:
        mock_is_correct.return_value = Feedback(name="correctness", value=True)

        scorer(
            inputs={"question": "query"},
            outputs="answer",
            expectations={"expected_facts": ["fact1", "fact2"]},
        )

        call_args = mock_is_correct.call_args[1]
        assert call_args["expected_facts"] == ["fact1", "fact2"]
        assert call_args["expected_response"] is None

        mock_is_correct.reset_mock()

        scorer(
            inputs={"question": "query"},
            outputs="answer",
            expectations={"expected_response": "expected answer"},
        )

        call_args = mock_is_correct.call_args[1]
        assert call_args["expected_response"] == "expected answer"
        assert call_args.get("expected_facts") is None


def test_correctness_with_custom_model():
    correctness_custom = Correctness(name="custom_correctness", model="openai:/gpt-4")

    with patch("mlflow.genai.judges.is_correct") as mock_is_correct:
        mock_is_correct.return_value = Feedback(name="custom_correctness", value=False)

        correctness_custom(
            inputs={"question": "test"},
            outputs="test answer",
            expectations={"expected_response": "different"},
        )

        call_args = mock_is_correct.call_args[1]
        assert call_args["name"] == "custom_correctness"
        assert call_args["model"] == "openai:/gpt-4"


def test_partial_override_scenario(chat_trace):
    scorer = Correctness()

    custom_expectations = {"expected_facts": ["Paris", "France", "capital"]}

    with patch("mlflow.genai.judges.is_correct") as mock_is_correct:
        mock_is_correct.return_value = Feedback(name="correctness", value=True)

        scorer(trace=chat_trace, expectations=custom_expectations)
        call_args = mock_is_correct.call_args[1]

        assert "What is the capital of France?" in call_args["request"]
        assert "Paris is the capital of France" in call_args["response"]
        assert call_args["expected_facts"] == ["Paris", "France", "capital"]
        assert call_args.get("expected_response") is None


def test_verify_extraction_utilities_work(simple_qa_trace):
    inputs = extract_inputs_from_trace(simple_qa_trace)
    assert inputs == {"question": "What is the meaning of life?"}

    outputs = extract_outputs_from_trace(simple_qa_trace)
    assert outputs == "The answer to 'What is the meaning of life?' is 42"

    expectations = extract_expectations_from_trace(simple_qa_trace)
    assert expectations["expected_response"] == "The answer is 42"
    assert expectations["expected_facts"] == ["42 is the answer", "from Hitchhiker's Guide"]


def test_verify_chat_extraction(chat_trace):
    inputs = extract_inputs_from_trace(chat_trace)
    assert "messages" in inputs or isinstance(inputs, list)
    if "messages" in inputs:
        assert inputs["messages"][0]["content"] == "What is the capital of France?"
    else:
        assert inputs[0]["content"] == "What is the capital of France?"

    outputs = extract_outputs_from_trace(chat_trace)
    assert "Paris is the capital of France" in str(outputs)


def test_guidelines_different_configurations():
    with patch("mlflow.genai.judges.meets_guidelines") as mock_meets:
        mock_meets.return_value = Feedback(name="expectations_guidelines", value=True)

        ExpectationsGuidelines()(
            inputs={"question": "query"},
            outputs="answer",
            expectations={"guidelines": ["guideline1", "guideline2"]},
        )

        call_args = mock_meets.call_args[1]
        assert call_args["guidelines"] == ["guideline1", "guideline2"]
        assert call_args["name"] == "expectations_guidelines"

    with patch("mlflow.genai.judges.meets_guidelines") as mock_meets:
        mock_meets.return_value = Feedback(name="is_english", value=True)

        is_english = Guidelines(
            name="is_english",
            guidelines=["The response should be in English."],
            model="openai:/gpt-4",
        )
        is_english(inputs={"question": "query"}, outputs="answer")

        call_args = mock_meets.call_args[1]
        assert call_args["guidelines"] == ["The response should be in English."]
        assert call_args["name"] == "is_english"
        assert call_args["model"] == "openai:/gpt-4"

    with patch("mlflow.genai.judges.meets_guidelines") as mock_meets:
        mock_meets.return_value = Feedback(name="is_polite", value=True)

        is_polite = Guidelines(
            name="is_polite", guidelines="Be polite and respectful.", model="openai:/gpt-4"
        )
        is_polite(inputs={"question": "query"}, outputs="answer")

        call_args = mock_meets.call_args[1]
        assert call_args["guidelines"] == "Be polite and respectful."
        assert call_args["name"] == "is_polite"


def test_relevance_with_custom_model():
    with patch("mlflow.genai.judges.is_context_relevant") as mock_relevant:
        mock_relevant.return_value = Feedback(name="relevance_to_query", value=True)

        RelevanceToQuery()(inputs={"question": "query"}, outputs="answer")

        call_args = mock_relevant.call_args[1]
        assert call_args["request"] == "{'question': 'query'}"
        assert call_args["context"] == "answer"
        assert call_args["name"] == "relevance_to_query"
        assert call_args["model"] is None

        mock_relevant.reset_mock()

        relevance_custom = RelevanceToQuery(name="custom_relevance", model="openai:/gpt-4")
        relevance_custom(inputs={"question": "query"}, outputs="answer")

        call_args = mock_relevant.call_args[1]
        assert call_args["name"] == "custom_relevance"
        assert call_args["model"] == "openai:/gpt-4"
