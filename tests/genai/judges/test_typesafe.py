import json
from typing import Any, Literal
from unittest import mock

import pydantic
import pytest
import requests

from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.typesafe import (
    _RETRY_CODES,
    _build_question,
    _gateway_typesafe_negative_cache,
    _invoke_typesafe_judge,
    _is_gateway_model,
    _is_typesafe_model,
    _try_invoke_gateway_typesafe_judge,
)
from mlflow.genai.scorers import Safety
from mlflow.tracing.constant import AssessmentMetadataKey

_REQUEST_TARGET = "mlflow.genai.judges.typesafe._get_http_response_with_retries"
_DEFAULT_STATE = object()


@pytest.fixture(autouse=True)
def clear_gateway_typesafe_negative_cache():
    _gateway_typesafe_negative_cache.clear()
    yield
    _gateway_typesafe_negative_cache.clear()


def _response(answer, usage=None):
    response = mock.Mock(status_code=200)
    body = {"model": "jev-1.13.0", "answers": {"evaluation": answer}}
    if usage is not None:
        body["usage"] = usage
    response.json.return_value = body
    return response


def _invoke(
    *,
    model_uri="typesafe:/jev-latest",
    instructions="Does {{ outputs }} answer {{ inputs }}?",
    state: Any = _DEFAULT_STATE,
    feedback_value_type=bool,
    **kwargs,
):
    if state is _DEFAULT_STATE:
        state = {"inputs": "Question", "outputs": "Answer"}
    return _invoke_typesafe_judge(
        model_uri,
        instructions=instructions,
        state=state,
        feedback_value_type=feedback_value_type,
        assessment_name="quality",
        **kwargs,
    )


@pytest.mark.parametrize(
    ("model_uri", "expected"),
    [
        ("typesafe:/jev-latest", True),
        ("typesafe://jev-latest", True),
        ("typesafe", False),
    ],
)
def test_is_typesafe_model(model_uri, expected):
    assert _is_typesafe_model(model_uri) is expected


def test_gateway_model_is_not_classified_as_direct_typesafe():
    assert _is_typesafe_model("gateway:/jev-evaluator") is False
    assert _is_gateway_model("gateway:/jev-evaluator") is True
    assert _is_gateway_model("typesafe:/jev-evaluator") is False


def test_gateway_chat_endpoint_falls_back_during_runtime_invocation():
    response = mock.Mock(status_code=422)
    response.json.return_value = {"detail": "Gateway endpoint does not use the TypeSafe provider."}
    with (
        mock.patch(
            "mlflow.genai.judges.typesafe._resolve_gateway_uri", return_value="https://mlflow"
        ),
        mock.patch("mlflow.genai.judges.typesafe.get_default_host_creds"),
        mock.patch("mlflow.genai.judges.typesafe.http_request", return_value=response),
    ):
        assert (
            _try_invoke_gateway_typesafe_judge(
                "gateway:/chat-endpoint",
                instructions="Does {{ outputs }} answer {{ inputs }}?",
                state={"inputs": "Question", "outputs": "Answer"},
                feedback_value_type=bool,
                assessment_name="quality",
            )
            is None
        )


def test_gateway_chat_endpoint_negative_result_is_cached():
    response = mock.Mock(status_code=422)
    response.json.return_value = {"detail": "Gateway endpoint does not use the TypeSafe provider."}
    kwargs = {
        "instructions": "Does {{ outputs }} answer {{ inputs }}?",
        "state": {"inputs": "Question", "outputs": "Answer"},
        "feedback_value_type": bool,
        "assessment_name": "quality",
    }
    with (
        mock.patch(
            "mlflow.genai.judges.typesafe._resolve_gateway_uri", return_value="https://mlflow"
        ) as resolve_gateway_uri,
        mock.patch("mlflow.genai.judges.typesafe.get_default_host_creds"),
        mock.patch("mlflow.genai.judges.typesafe.http_request", return_value=response) as request,
    ):
        assert _try_invoke_gateway_typesafe_judge("gateway:/chat-endpoint", **kwargs) is None
        assert _try_invoke_gateway_typesafe_judge("gateway:/chat-endpoint", **kwargs) is None

    assert resolve_gateway_uri.call_count == 3
    request.assert_called_once()


def test_gateway_404_preserves_chat_fallback_for_older_servers():
    response = mock.Mock(status_code=404)
    response.json.return_value = {"detail": "Not Found"}
    with (
        mock.patch(
            "mlflow.genai.judges.typesafe._resolve_gateway_uri", return_value="https://mlflow"
        ),
        mock.patch("mlflow.genai.judges.typesafe.get_default_host_creds"),
        mock.patch("mlflow.genai.judges.typesafe.http_request", return_value=response),
    ):
        assert (
            _try_invoke_gateway_typesafe_judge(
                "gateway:/chat-endpoint",
                instructions="Does {{ outputs }} answer {{ inputs }}?",
                state={"inputs": "Question", "outputs": "Answer"},
                feedback_value_type=bool,
                assessment_name="quality",
            )
            is None
        )


def test_gateway_typesafe_success_does_not_cache_negative_result():
    response = _response({"type": "noul", "noul": 0.8})
    with (
        mock.patch(
            "mlflow.genai.judges.typesafe._resolve_gateway_uri", return_value="https://mlflow"
        ),
        mock.patch("mlflow.genai.judges.typesafe.get_default_host_creds"),
        mock.patch("mlflow.genai.judges.typesafe.http_request", return_value=response) as request,
    ):
        feedback = _try_invoke_gateway_typesafe_judge(
            "gateway:/typesafe-endpoint",
            instructions="Does {{ outputs }} answer {{ inputs }}?",
            state={"inputs": "Question", "outputs": "Answer"},
            feedback_value_type=bool,
            assessment_name="quality",
        )

    assert feedback.value is True
    request.assert_called_once()
    assert _gateway_typesafe_negative_cache == {}


def test_gateway_typesafe_endpoint_404_does_not_fall_back_to_chat():
    response = mock.Mock(status_code=404)
    response.json.return_value = {"detail": "Endpoint not found"}
    with (
        mock.patch(
            "mlflow.genai.judges.typesafe._resolve_gateway_uri", return_value="https://mlflow"
        ),
        mock.patch("mlflow.genai.judges.typesafe.get_default_host_creds"),
        mock.patch("mlflow.genai.judges.typesafe.http_request", return_value=response),
        pytest.raises(MlflowException, match="TypeSafe evaluation failed with HTTP 404"),
    ):
        _try_invoke_gateway_typesafe_judge(
            "gateway:/typesafe-endpoint",
            instructions="Does {{ outputs }} answer {{ inputs }}?",
            state={"inputs": "Question", "outputs": "Answer"},
            feedback_value_type=bool,
            assessment_name="quality",
        )
    assert _gateway_typesafe_negative_cache == {}


def test_gateway_chat_judge_with_unsupported_typesafe_options_skips_native_attempt():
    with mock.patch("mlflow.genai.judges.typesafe.http_request") as request:
        assert (
            _try_invoke_gateway_typesafe_judge(
                "gateway:/chat-endpoint",
                instructions="Explain {{ outputs }}",
                state={"outputs": "Answer"},
                feedback_value_type=str,
                assessment_name="quality",
            )
            is None
        )
    request.assert_not_called()


def test_gateway_mixed_provider_runtime_invocation_is_rejected():
    response = mock.Mock(status_code=400)
    response.json.return_value = {
        "detail": "Gateway judge endpoints cannot mix TypeSafe and chat model providers."
    }
    with (
        mock.patch(
            "mlflow.genai.judges.typesafe._resolve_gateway_uri", return_value="https://mlflow"
        ),
        mock.patch("mlflow.genai.judges.typesafe.get_default_host_creds"),
        mock.patch("mlflow.genai.judges.typesafe.http_request", return_value=response),
        pytest.raises(MlflowException, match="TypeSafe evaluation failed with HTTP 400"),
    ):
        _try_invoke_gateway_typesafe_judge(
            "gateway:/mixed-endpoint",
            instructions="Does {{ outputs }} answer {{ inputs }}?",
            state={"inputs": "Question", "outputs": "Answer"},
            feedback_value_type=bool,
            assessment_name="quality",
        )


def test_gateway_invocation_uses_system_one_route_without_typesafe_api_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    response = _response({"type": "noul", "noul": 0.8})
    with (
        mock.patch(
            "mlflow.genai.judges.typesafe._resolve_gateway_uri", return_value="https://mlflow"
        ),
        mock.patch("mlflow.genai.judges.typesafe.get_default_host_creds") as get_creds,
        mock.patch("mlflow.genai.judges.typesafe.http_request", return_value=response) as request,
    ):
        feedback = _invoke(model_uri="gateway:/jev-evaluator")

    assert feedback.source.source_id == "gateway:/jev-evaluator"
    request.assert_called_once_with(
        host_creds=get_creds.return_value,
        endpoint="/gateway/typesafe/v1/systemone",
        method="POST",
        max_retries=10,
        backoff_factor=1,
        backoff_jitter=0.1,
        retry_codes=_RETRY_CODES,
        timeout=mock.ANY,
        raise_on_status=False,
        json={
            "model": "jev-evaluator",
            "state": {"inputs": "Question", "outputs": "Answer"},
            "questions": {
                "evaluation": {
                    "type": "noul",
                    "instructions": "Does state.outputs answer state.inputs?",
                }
            },
        },
    )


def test_make_judge_invokes_gateway_typesafe_through_public_api():
    evaluation_response = _response({"type": "noul", "noul": 0.9})
    with (
        mock.patch(
            "mlflow.genai.judges.typesafe._resolve_gateway_uri", return_value="https://mlflow"
        ),
        mock.patch("mlflow.genai.judges.typesafe.get_default_host_creds"),
        mock.patch(
            "mlflow.genai.judges.typesafe.http_request",
            return_value=evaluation_response,
        ) as request,
    ):
        feedback = make_judge(
            name="quality",
            instructions="Does {{ outputs }} answer {{ inputs }}?",
            model="gateway:/jev-evaluator",
            feedback_value_type=bool,
        )(inputs={"question": "Why?"}, outputs={"answer": "Because."})

    assert feedback.value is True
    request.assert_called_once()
    assert request.call_args.kwargs["endpoint"] == "/gateway/typesafe/v1/systemone"


def test_direct_bool_invocation_uses_native_evaluation(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret")
    response = _response(
        {"type": "noul", "noul": 0.8},
        usage={"input_tokens": 100, "output_tokens": 20},
    )
    with mock.patch(_REQUEST_TARGET, return_value=response) as request:
        feedback = _invoke(num_retries=3)

    assert feedback.name == "quality"
    assert feedback.value is True
    assert feedback.rationale is None
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "typesafe:/jev-latest"
    assert feedback.metadata == {
        "typesafe.model": "jev-1.13.0",
        "typesafe.probability": "0.8",
        AssessmentMetadataKey.JUDGE_INPUT_TOKENS: "100",
        AssessmentMetadataKey.JUDGE_OUTPUT_TOKENS: "20",
    }
    request.assert_called_once()
    call = request.call_args.kwargs
    assert call["url"] == "https://api.typesafe.ai/v1/systemone"
    assert call["headers"] == {"Authorization": "Bearer typesafe-secret"}
    assert call["max_retries"] == 3
    assert call["allow_redirects"] is False
    assert call["json"] == {
        "model": "jev-latest",
        "state": {"inputs": "Question", "outputs": "Answer"},
        "questions": {
            "evaluation": {
                "type": "noul",
                "instructions": "Does state.outputs answer state.inputs?",
            }
        },
    }


def test_bool_noul_probability_below_half_is_false(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret")
    with mock.patch(
        _REQUEST_TARGET,
        return_value=_response({"type": "noul", "noul": 0.49}),
    ):
        feedback = _invoke()

    assert feedback.value is False
    assert feedback.metadata["typesafe.probability"] == "0.49"


def test_literal_maps_choice_and_metadata(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret")
    answer = {
        "type": "choice",
        "choice": "resolved",
        "probabilities": {"none": 0.1, "resolved": 0.8, "unresolved": 0.1},
        "confidence": 0.7,
        "legend": {
            "none": "No issue",
            "resolved": "Resolved issue",
            "unresolved": "Unresolved issue",
        },
    }
    with mock.patch(_REQUEST_TARGET, return_value=_response(answer)) as request:
        feedback = _invoke(feedback_value_type=Literal["none", "resolved", "unresolved"])

    assert feedback.value == "resolved"
    assert json.loads(feedback.metadata["typesafe.probabilities"]) == {
        "none": 0.1,
        "resolved": 0.8,
        "unresolved": 0.1,
    }
    assert feedback.metadata["typesafe.confidence"] == "0.7"
    assert json.loads(feedback.metadata["typesafe.legend"]) == answer["legend"]
    question = request.call_args.kwargs["json"]["questions"]["evaluation"]
    assert question == {
        "type": "choice",
        "criteria": {
            "none": "The evaluation result is none.",
            "resolved": "The evaluation result is resolved.",
            "unresolved": "The evaluation result is unresolved.",
        },
        "instructions": "Does state.outputs answer state.inputs?",
    }


def test_state_reference_rewriting_uses_available_fields_only(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret")
    with mock.patch(
        _REQUEST_TARGET,
        return_value=_response({"type": "noul", "noul": 0.8}),
    ) as request:
        _invoke(
            instructions="Evaluate {{ request }}; keep {{ unknown }} literal.",
            state={"request": "Question"},
        )

    instructions = request.call_args.kwargs["json"]["questions"]["evaluation"]["instructions"]
    assert instructions == "Evaluate state.request; keep {{ unknown }} literal."


def test_state_uses_trace_json_encoding_and_rejects_nan(monkeypatch):
    class StateValue(pydantic.BaseModel):
        answer: str

    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret")
    with mock.patch(
        _REQUEST_TARGET,
        return_value=_response({"type": "noul", "noul": 0.8}),
    ) as request:
        _invoke(state={"outputs": StateValue(answer="yes")})

    assert request.call_args.kwargs["json"]["state"] == {"outputs": {"answer": "yes"}}

    with (
        mock.patch(_REQUEST_TARGET) as request,
        pytest.raises(MlflowException, match="without NaN or infinity"),
    ):
        _invoke(state={"outputs": float("nan")})
    request.assert_not_called()


@pytest.mark.parametrize("feedback_value_type", [str, int, float, list[str], dict[str, str]])
def test_unsupported_feedback_value_type(feedback_value_type):
    with pytest.raises(MlflowException, match="bool or finite Literal"):
        _build_question(feedback_value_type)


@pytest.mark.parametrize(
    ("feedback_value_type", "message"),
    [
        (Literal[1, "1"], "distinct string representations"),
        (Literal[" ", "valid"], "non-empty choice labels"),
        (Literal[float("nan"), "valid"], "NaN or infinity"),
        (Literal[tuple(range(256))], "between 1 and 255"),
    ],
)
def test_invalid_literal_feedback_value_type(feedback_value_type, message):
    with pytest.raises(MlflowException, match=message):
        _build_question(feedback_value_type)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_uri": "openai:/gpt-4"}, "Expected a typesafe:/ or gateway:/ model URI"),
        ({"instructions": " "}, "instructions must be a non-empty string"),
        ({"instructions": "Evaluate {{ trace }}"}, "trace-based tool calling"),
        ({"state": []}, "state must be a dictionary"),
    ],
)
def test_rejects_invalid_invocation_inputs(monkeypatch, kwargs, message):
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret")
    with (
        mock.patch(_REQUEST_TARGET) as request,
        pytest.raises(MlflowException, match=message),
    ):
        _invoke(**kwargs)
    request.assert_not_called()


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with (
        mock.patch(_REQUEST_TARGET) as request,
        pytest.raises(MlflowException, match="Set TYPESAFE_API_KEY"),
    ):
        _invoke()
    request.assert_not_called()


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("inference_params", {"temperature": 0}),
        ("base_url", "https://example.com"),
        ("extra_headers", {"X-Test": "value"}),
    ],
)
def test_rejects_unsupported_options(monkeypatch, option, value):
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret")
    with (
        mock.patch(_REQUEST_TARGET) as request,
        pytest.raises(MlflowException, match=option),
    ):
        _invoke(**{option: value})
    request.assert_not_called()


@pytest.mark.parametrize("status_code", [302, 401, 403, 422, 429, 529])
def test_http_errors_do_not_expose_response_or_credentials(monkeypatch, status_code):
    monkeypatch.setenv("TYPESAFE_API_KEY", "private-key")
    response = mock.Mock(status_code=status_code, text="provider echoed private-key")
    with (
        mock.patch(_REQUEST_TARGET, return_value=response) as request,
        pytest.raises(MlflowException, match=f"HTTP {status_code}") as exc,
    ):
        _invoke()
    request.assert_called_once()
    assert "private-key" not in str(exc.value)


def test_connection_error_is_sanitized(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "private-key")
    with (
        mock.patch(
            _REQUEST_TARGET,
            side_effect=requests.ConnectionError("private-key"),
        ) as request,
        pytest.raises(MlflowException, match="Failed to connect") as exc,
    ):
        _invoke()
    request.assert_called_once()
    assert "private-key" not in str(exc.value)


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "noul", "noul": True},
        {"type": "noul", "noul": float("nan")},
        {"type": "noul", "noul": 10**10000},
        {"type": "choice", "choice": "unknown", "probabilities": {}, "confidence": 0.5},
        {
            "type": "choice",
            "choice": "a",
            "probabilities": {"a": 0.9, "b": 0.9},
            "confidence": 0.5,
        },
    ],
)
def test_invalid_answer(monkeypatch, answer):
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret")
    feedback_value_type = bool if answer["type"] == "noul" else Literal["a", "b"]
    with (
        mock.patch(_REQUEST_TARGET, return_value=_response(answer)) as request,
        pytest.raises(MlflowException, match="invalid .* answer"),
    ):
        _invoke(feedback_value_type=feedback_value_type)
    request.assert_called_once()


def test_make_judge_invokes_typesafe_through_public_api(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret")
    response = _response({
        "type": "choice",
        "choice": "pass",
        "probabilities": {"pass": 0.9, "fail": 0.1},
        "confidence": 0.9,
    })
    with mock.patch(_REQUEST_TARGET, return_value=response) as request:
        feedback = make_judge(
            name="quality",
            instructions="Does {{ outputs }} answer {{ inputs }}?",
            model="typesafe:/jev-latest",
            feedback_value_type=Literal["pass", "fail"],
        )(inputs={"question": "Why?"}, outputs={"answer": "Because."})

    assert feedback.value == "pass"
    request.assert_called_once()
    assert request.call_args.kwargs["json"]["state"] == {
        "inputs": {"question": "Why?"},
        "outputs": {"answer": "Because."},
    }


def test_builtin_scorer_invokes_typesafe_through_public_api(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret")
    response = _response({
        "type": "choice",
        "choice": "yes",
        "probabilities": {"yes": 0.99, "no": 0.01},
        "confidence": 0.99,
    })
    with mock.patch(_REQUEST_TARGET, return_value=response) as request:
        feedback = Safety(model="typesafe:/jev-latest")(outputs="A harmless response")

    assert feedback.value == "yes"
    request.assert_called_once()
    payload = request.call_args.kwargs["json"]
    assert payload["state"] == {"content": "A harmless response"}
    instructions = payload["questions"]["evaluation"]["instructions"]
    assert instructions.startswith("You are a content safety classifier")
    assert "state.content" in instructions
    assert "JSON" not in instructions
    assert "rationale" not in instructions.lower()
