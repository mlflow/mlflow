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
    _build_question,
    _invoke_typesafe_judge,
    _is_typesafe_model,
)
from mlflow.genai.scorers import Safety
from mlflow.tracing.constant import AssessmentMetadataKey

_REQUEST_TARGET = "mlflow.genai.judges.typesafe._get_http_response_with_retries"
_DEFAULT_STATE = object()


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
        ("gateway:/jev-evaluator", False),
        ("typesafe", False),
    ],
)
def test_is_typesafe_model(model_uri, expected):
    assert _is_typesafe_model(model_uri) is expected


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


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        (
            "https://system-one.example.com/v1/",
            "https://system-one.example.com/v1/systemone",
        ),
        (
            "https://system-one.example.com/v1?region=us",
            "https://system-one.example.com/v1/systemone?region=us",
        ),
        (
            "http://127.0.0.1:8080/v1",
            "http://127.0.0.1:8080/v1/systemone",
        ),
    ],
)
def test_direct_invocation_supports_custom_base_url_and_headers(
    monkeypatch, base_url, expected_url
):
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret")
    response = _response({"type": "noul", "noul": 0.8})
    with mock.patch(_REQUEST_TARGET, return_value=response) as request:
        _invoke(
            base_url=base_url,
            extra_headers={"X-Request-ID": "request-123"},
        )

    call = request.call_args.kwargs
    assert call["url"] == expected_url
    assert call["headers"] == {
        "Authorization": "Bearer typesafe-secret",
        "X-Request-ID": "request-123",
    }


@pytest.mark.parametrize("api_key", [None, "typesafe-secret"])
def test_direct_invocation_supports_authorization_header(monkeypatch, api_key):
    if api_key:
        monkeypatch.setenv("TYPESAFE_API_KEY", api_key)
    else:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    response = _response({"type": "noul", "noul": 0.8})
    with mock.patch(_REQUEST_TARGET, return_value=response) as request:
        _invoke(extra_headers={"authorization": "Bearer custom-token"})

    assert request.call_args.kwargs["headers"] == {"authorization": "Bearer custom-token"}


@pytest.mark.parametrize("authorization", ["", " \t ", None])
def test_direct_invocation_rejects_empty_authorization_header(monkeypatch, authorization):
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret")
    with (
        mock.patch(_REQUEST_TARGET) as request,
        pytest.raises(MlflowException, match="Authorization header.*non-empty"),
    ):
        _invoke(extra_headers={"Authorization": authorization})
    request.assert_not_called()


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


def test_rejects_inference_params(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret")
    with (
        mock.patch(_REQUEST_TARGET) as request,
        pytest.raises(MlflowException, match="inference_params"),
    ):
        _invoke(inference_params={"temperature": 0})
    request.assert_not_called()


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("base_url", "https://example.com"),
        ("extra_headers", {"X-Test": "value"}),
    ],
)
def test_gateway_invocation_rejects_direct_connection_options(monkeypatch, option, value):
    with (
        mock.patch("mlflow.genai.judges.typesafe._send_gateway_request") as request,
        pytest.raises(MlflowException, match=option),
    ):
        _invoke(model_uri="gateway:/jev-evaluator", **{option: value})
    request.assert_not_called()


def test_direct_invocation_rejects_empty_base_url(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret")
    with (
        mock.patch(_REQUEST_TARGET) as request,
        pytest.raises(MlflowException, match="base_url must be a non-empty string"),
    ):
        _invoke(base_url="  ///  ")
    request.assert_not_called()


@pytest.mark.parametrize(
    "base_url",
    [
        "http://",
        "://",
        "/v1",
        "v1",
        "ftp://system-one.example.com/v1",
        "https://system-one.example.com:not-a-port/v1",
        "https://system-one.example.com:99999/v1",
    ],
)
def test_direct_invocation_rejects_invalid_base_url(monkeypatch, base_url):
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret")
    with (
        mock.patch(_REQUEST_TARGET) as request,
        pytest.raises(MlflowException, match="absolute HTTP or HTTPS URL"),
    ):
        _invoke(base_url=base_url)
    request.assert_not_called()


@pytest.mark.parametrize(
    "extra_headers",
    [None, {"Authorization": "Bearer custom-token"}],
)
@pytest.mark.parametrize(
    "base_url",
    [
        "https://user@system-one.example.com/v1",
        "https://user:password@system-one.example.com/v1",
        "https://:password@system-one.example.com/v1",
        "https://us%65r:p%40ss@system-one.example.com/v1",
    ],
)
def test_direct_invocation_rejects_credentials_in_base_url(monkeypatch, extra_headers, base_url):
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret")
    with (
        mock.patch(_REQUEST_TARGET) as request,
        pytest.raises(MlflowException, match="Credentials in base_url are not supported"),
    ):
        _invoke(base_url=base_url, extra_headers=extra_headers)
    request.assert_not_called()


def test_authorization_header_takes_precedence_over_netrc(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    response = _response({"type": "noul", "noul": 0.8})
    with (
        mock.patch(
            "requests.sessions.get_netrc_auth",
            return_value=("netrc-user", "netrc-password"),
        ),
        mock.patch.object(
            requests.Session,
            "send",
            autospec=True,
            return_value=response,
        ) as send,
    ):
        _invoke(
            base_url="https://system-one.example.com/v1",
            extra_headers={"Authorization": "Bearer custom-token"},
            num_retries=0,
        )

    prepared_request = send.call_args.args[1]
    assert prepared_request.headers["Authorization"] == "Bearer custom-token"


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
            base_url="https://system-one.example.com/v1",
            extra_headers={"X-Request-ID": "request-123"},
        )(inputs={"question": "Why?"}, outputs={"answer": "Because."})

    assert feedback.value == "pass"
    request.assert_called_once()
    assert request.call_args.kwargs["url"] == "https://system-one.example.com/v1/systemone"
    assert request.call_args.kwargs["headers"] == {
        "Authorization": "Bearer typesafe-secret",
        "X-Request-ID": "request-123",
    }
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
