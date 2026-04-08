import json
from unittest.mock import patch

import pytest

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.span import SpanType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers import HallucinationDetection
from mlflow.genai.utils.trace_utils import (
    extract_generation_context,
    extract_messages_from_span_inputs,
)


def _make_trace(*, span_type, messages, response):
    @mlflow.trace(name="root", span_type=SpanType.CHAIN)
    def _predict():
        return _llm()

    @mlflow.trace(name="llm_call", span_type=span_type)
    def _llm():
        span = mlflow.get_current_active_span()
        span.set_inputs({"messages": messages})
        return response

    _predict()
    return mlflow.get_trace(mlflow.get_last_active_trace_id())


@pytest.fixture
def trace_with_llm_span():
    return _make_trace(
        span_type=SpanType.LLM,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "assistant",
                "content": "Based on the document: Paris is the capital of France.",
            },
            {"role": "user", "content": "What is the capital of France?"},
        ],
        response="The capital of France is Paris, located on the Seine river.",
    )


@pytest.fixture
def trace_no_llm_span():
    @mlflow.trace(name="root", span_type=SpanType.CHAIN)
    def _predict():
        return "Some answer"

    _predict()
    return mlflow.get_trace(mlflow.get_last_active_trace_id())


@pytest.fixture
def trace_with_only_user_messages():
    return _make_trace(
        span_type=SpanType.LLM,
        messages=[{"role": "user", "content": "Hello"}],
        response="Some response.",
    )


# -- extract_messages_from_span_inputs tests --


@pytest.mark.parametrize(
    ("inputs", "expected_count"),
    [
        ({"messages": [{"role": "user", "content": "hi"}]}, 1),
        (json.dumps([{"role": "system", "content": "ctx"}, {"role": "user", "content": "q"}]), 2),
        (json.dumps({"messages": [{"role": "user", "content": "q"}]}), 1),
        (json.dumps([[{"type": "human", "content": "q"}, {"type": "ai", "content": "a"}]]), 2),
    ],
    ids=["dict", "json-list", "json-dict", "nested-langchain"],
)
def test_extract_messages_from_span_inputs(inputs, expected_count):
    result = extract_messages_from_span_inputs(inputs)
    assert result is not None
    assert len(result) == expected_count


@pytest.mark.parametrize(
    "inputs",
    [None, "", "not-json", json.dumps(42)],
    ids=["none", "empty", "invalid-json", "json-number"],
)
def test_extract_messages_from_span_inputs_returns_none(inputs):
    assert extract_messages_from_span_inputs(inputs) is None


# -- extract_generation_context tests --


def test_extract_generation_context(trace_with_llm_span):
    context = extract_generation_context(trace_with_llm_span)

    assert "You are a helpful assistant." in context
    assert "Based on the document: Paris is the capital of France." in context
    assert "What is the capital of France?" not in context


@pytest.mark.parametrize(
    ("fixture_name", "error_match"),
    [
        ("trace_no_llm_span", "No LLM or CHAT_MODEL span found"),
        ("trace_with_only_user_messages", "No non-user context messages found"),
    ],
)
def test_extract_generation_context_errors(fixture_name, error_match, request):
    trace = request.getfixturevalue(fixture_name)
    with pytest.raises(MlflowException, match=error_match):
        extract_generation_context(trace)


# -- HallucinationDetection scorer tests --


@pytest.mark.parametrize(
    ("span_type", "system_content", "response", "expected_context_substr"),
    [
        (
            SpanType.LLM,
            "You are a helpful assistant.",
            "The capital of France is Paris, located on the Seine river.",
            "You are a helpful assistant.",
        ),
        (
            SpanType.CHAT_MODEL,
            "Reference doc: MLflow is an ML platform.",
            "MLflow is an open-source ML platform.",
            "Reference doc: MLflow is an ML platform.",
        ),
    ],
    ids=["llm-span", "chat-model-span"],
)
def test_hallucination_detection(span_type, system_content, response, expected_context_substr):
    trace = _make_trace(
        span_type=span_type,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": "question"},
        ],
        response=response,
    )

    with patch(
        "mlflow.genai.scorers.builtin_scorers.invoke_judge_model",
        return_value=Feedback(name="hallucination_detection", value="yes", rationale="Faithful."),
    ) as mock_invoke:
        result = HallucinationDetection()(trace=trace)

        assert result.name == "hallucination_detection"
        assert result.value == CategoricalRating.YES
        mock_invoke.assert_called_once()

        prompt_arg = mock_invoke.call_args[0][1]
        assert expected_context_substr in prompt_arg
        assert response in prompt_arg


@pytest.mark.parametrize(
    ("fixture_name", "error_match"),
    [
        ("trace_no_llm_span", "No LLM or CHAT_MODEL span found"),
        ("trace_with_only_user_messages", "No non-user context messages found"),
    ],
)
def test_hallucination_detection_errors(fixture_name, error_match, request):
    trace = request.getfixturevalue(fixture_name)
    with pytest.raises(MlflowException, match=error_match):
        HallucinationDetection()(trace=trace)


def test_hallucination_detection_custom_name(trace_with_llm_span):
    with patch(
        "mlflow.genai.scorers.builtin_scorers.invoke_judge_model",
        return_value=Feedback(name="my_hallucination", value="yes", rationale="OK."),
    ) as mock_invoke:
        result = HallucinationDetection(name="my_hallucination")(trace=trace_with_llm_span)

        assert result.name == "my_hallucination"
        mock_invoke.assert_called_once()
        assert mock_invoke.call_args[1]["assessment_name"] == "my_hallucination"


def test_hallucination_detection_hallucinated(trace_with_llm_span):
    with patch(
        "mlflow.genai.scorers.builtin_scorers.invoke_judge_model",
        return_value=Feedback(
            name="hallucination_detection", value="no", rationale="Contains unsupported claims."
        ),
    ) as mock_invoke:
        result = HallucinationDetection()(trace=trace_with_llm_span)

        assert result.value == CategoricalRating.NO
        mock_invoke.assert_called_once()
