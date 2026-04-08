from unittest.mock import patch

import pytest

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.span import SpanType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers import HallucinationDetection

_MOCK_INVOKE = "mlflow.genai.judges.instructions_judge.invoke_judge_model"


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


_RAG_SYSTEM_PROMPT = "You are a helpful assistant. Answer based on the provided context."
_RAG_CONTEXT = (
    "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. "
    "It was created by Databricks in 2018. MLflow has four main components: Tracking, Projects, "
    "Models, and Model Registry."
)
_RAG_USER_QUERY = "What is MLflow and when was it created?"


def _make_rag_messages(system_prompt=_RAG_SYSTEM_PROMPT, context=_RAG_CONTEXT):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _RAG_USER_QUERY},
        {"role": "assistant", "content": f"Context:\n{context}"},
    ]


@pytest.fixture
def rag_trace():
    return _make_trace(
        span_type=SpanType.LLM,
        messages=_make_rag_messages(),
        response="MLflow is an open-source ML platform created by Databricks in 2018.",
    )


def _get_prompt_text(mock_invoke):
    """Extract the full prompt text from the invoke_judge_model mock call."""
    messages = mock_invoke.call_args.kwargs["prompt"]
    return "\n".join(msg.content for msg in messages)


@pytest.mark.parametrize(
    ("span_type", "response"),
    [
        (SpanType.LLM, "MLflow is an open-source ML platform created by Databricks in 2018."),
        (SpanType.CHAT_MODEL, "MLflow was created in 2018 and has four main components."),
    ],
    ids=["llm-span", "chat-model-span"],
)
def test_hallucination_detection(span_type, response):
    trace = _make_trace(
        span_type=span_type,
        messages=_make_rag_messages(),
        response=response,
    )

    with patch(
        _MOCK_INVOKE,
        return_value=Feedback(name="hallucination_detection", value="yes", rationale="Faithful."),
    ) as mock_invoke:
        result = HallucinationDetection()(trace=trace)

        assert result.name == "hallucination_detection"
        assert result.value == CategoricalRating.YES
        mock_invoke.assert_called_once()

        prompt_text = _get_prompt_text(mock_invoke)
        assert _RAG_CONTEXT in prompt_text
        assert response in prompt_text


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


def test_hallucination_detection_custom_name(rag_trace):
    with patch(
        _MOCK_INVOKE,
        return_value=Feedback(name="my_hallucination", value="yes", rationale="OK."),
    ) as mock_invoke:
        result = HallucinationDetection(name="my_hallucination")(trace=rag_trace)

        assert result.name == "my_hallucination"
        mock_invoke.assert_called_once()
        assert mock_invoke.call_args.kwargs["assessment_name"] == "my_hallucination"


def test_hallucination_detection_hallucinated(rag_trace):
    with patch(
        _MOCK_INVOKE,
        return_value=Feedback(
            name="hallucination_detection", value="no", rationale="Contains unsupported claims."
        ),
    ) as mock_invoke:
        result = HallucinationDetection()(trace=rag_trace)

        assert result.value == CategoricalRating.NO
        mock_invoke.assert_called_once()
