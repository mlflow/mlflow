import json
from typing import Literal
from unittest import mock

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import SYSTEM_ONE_CHAT_ROUTE_REJECTION_DETAIL
from mlflow.genai.judges.adapters.utils import ChatCompletionError
from mlflow.genai.judges.structured_judge import (
    _gateway_system_one_cache,
    _invoke_gateway_judge,
    _invoke_structured_builtin_judge,
    _is_gateway_model,
)

# ``_resolve_gateway_uri`` is referenced from two module namespaces: the router computes the
# cache key through ``structured_judge``'s binding, and the System One client reaches the
# gateway through ``typesafe``'s. Tests that exercise the System One leg patch both.
_CACHE_URI_TARGET = "mlflow.genai.judges.structured_judge._resolve_gateway_uri"
_CLIENT_URI_TARGET = "mlflow.genai.judges.typesafe._resolve_gateway_uri"

_GATEWAY_JUDGE_KWARGS = {
    "instructions": "Does {{ outputs }} answer {{ inputs }}?",
    "state": {"inputs": "Question", "outputs": "Answer"},
    "feedback_value_type": bool,
    "assessment_name": "quality",
}

_BUILTIN_JUDGE_KWARGS = {
    "instructions": "Is {{ outputs }} safe?",
    "state": {"content": "hello"},
    "assessment_name": "safety",
}


@pytest.fixture(autouse=True)
def clear_gateway_system_one_cache():
    _gateway_system_one_cache.clear()
    yield
    _gateway_system_one_cache.clear()


def _system_one_chat_rejection() -> MlflowException:
    """Build the error a chat invocation raises when the endpoint only serves System One.

    Mirrors the real gateway chat path: ``send_chat_request`` raises ``ChatCompletionError``
    (status 400, body text) and ``gateway_adapter`` wraps it with ``raise ... from e``.
    """
    cause = ChatCompletionError(
        status_code=400,
        message=json.dumps({"detail": SYSTEM_ONE_CHAT_ROUTE_REJECTION_DETAIL}),
    )
    exc = MlflowException(f"Failed to invoke judge model: {cause.message}")
    exc.__cause__ = cause
    return exc


def _system_one_response():
    response = mock.Mock(status_code=200)
    response.json.return_value = {
        "model": "jev-evaluator",
        "answers": {"evaluation": {"type": "noul", "noul": 0.8}},
    }
    return response


def test_is_gateway_model():
    assert _is_gateway_model("gateway:/jev-evaluator") is True
    assert _is_gateway_model("gateway://jev-evaluator") is True
    assert _is_gateway_model("typesafe:/jev-evaluator") is False
    assert _is_gateway_model("gateway") is False


def test_gateway_chat_endpoint_uses_chat_without_system_one_attempt():
    chat_feedback = Feedback(name="quality", value=True)
    chat_invoker = mock.Mock(return_value=chat_feedback)
    with (
        mock.patch(_CACHE_URI_TARGET, return_value="https://mlflow"),
        mock.patch("mlflow.genai.judges.typesafe.http_request") as request,
    ):
        feedback = _invoke_gateway_judge(
            "gateway:/chat-endpoint", chat_invoker=chat_invoker, **_GATEWAY_JUDGE_KWARGS
        )

    assert feedback is chat_feedback
    chat_invoker.assert_called_once()
    request.assert_not_called()
    assert _gateway_system_one_cache == set()


def test_gateway_system_one_endpoint_falls_back_from_chat_rejection():
    chat_invoker = mock.Mock(side_effect=_system_one_chat_rejection())
    with (
        mock.patch(_CACHE_URI_TARGET, return_value="https://mlflow"),
        mock.patch(_CLIENT_URI_TARGET, return_value="https://mlflow"),
        mock.patch("mlflow.genai.judges.typesafe.get_default_host_creds"),
        mock.patch(
            "mlflow.genai.judges.typesafe.http_request", return_value=_system_one_response()
        ) as request,
    ):
        feedback = _invoke_gateway_judge(
            "gateway:/jev-endpoint", chat_invoker=chat_invoker, **_GATEWAY_JUDGE_KWARGS
        )

    assert feedback.value is True
    chat_invoker.assert_called_once()
    assert request.call_args.kwargs["endpoint"] == "/gateway/typesafe/v1/systemone"


def test_gateway_non_system_one_chat_error_propagates():
    chat_invoker = mock.Mock(side_effect=MlflowException("gateway exploded"))
    with (
        mock.patch(_CACHE_URI_TARGET, return_value="https://mlflow"),
        mock.patch("mlflow.genai.judges.typesafe.http_request") as request,
    ):
        with pytest.raises(MlflowException, match="gateway exploded"):
            _invoke_gateway_judge(
                "gateway:/chat-endpoint", chat_invoker=chat_invoker, **_GATEWAY_JUDGE_KWARGS
            )

    request.assert_not_called()
    assert _gateway_system_one_cache == set()


def test_gateway_unrelated_400_chat_error_propagates():
    # A 400 that is NOT the System One rejection (e.g. OpenRouter "not a valid model ID")
    # must propagate, not trigger a System One fallback.
    cause = ChatCompletionError(
        status_code=400, message=json.dumps({"detail": "typesafe/jev-latest is not a valid model"})
    )
    exc = MlflowException(f"Failed to invoke judge model: {cause.message}")
    exc.__cause__ = cause
    chat_invoker = mock.Mock(side_effect=exc)
    with (
        mock.patch(_CACHE_URI_TARGET, return_value="https://mlflow"),
        mock.patch("mlflow.genai.judges.typesafe.http_request") as request,
    ):
        with pytest.raises(MlflowException, match="not a valid model"):
            _invoke_gateway_judge(
                "gateway:/openrouter-jev", chat_invoker=chat_invoker, **_GATEWAY_JUDGE_KWARGS
            )

    request.assert_not_called()
    assert _gateway_system_one_cache == set()


def test_gateway_system_one_endpoint_is_cached_after_detection():
    chat_invoker = mock.Mock(side_effect=_system_one_chat_rejection())
    with (
        mock.patch(_CACHE_URI_TARGET, return_value="https://mlflow"),
        mock.patch(_CLIENT_URI_TARGET, return_value="https://mlflow"),
        mock.patch("mlflow.genai.judges.typesafe.get_default_host_creds"),
        mock.patch(
            "mlflow.genai.judges.typesafe.http_request", return_value=_system_one_response()
        ),
    ):
        _invoke_gateway_judge(
            "gateway:/jev-endpoint", chat_invoker=chat_invoker, **_GATEWAY_JUDGE_KWARGS
        )
        _invoke_gateway_judge(
            "gateway:/jev-endpoint", chat_invoker=chat_invoker, **_GATEWAY_JUDGE_KWARGS
        )

    # Chat is attempted only on the first row; later rows go straight to System One.
    chat_invoker.assert_called_once()


def test_gateway_cached_system_one_recovers_when_reconfigured_to_chat():
    reconfigured = mock.Mock(status_code=422)
    reconfigured.json.return_value = {
        "detail": "Gateway endpoint does not use the TypeSafe provider."
    }
    chat_feedback = Feedback(name="quality", value=False)
    chat_invoker = mock.Mock(side_effect=[_system_one_chat_rejection(), chat_feedback])
    with (
        mock.patch(_CACHE_URI_TARGET, return_value="https://mlflow"),
        mock.patch(_CLIENT_URI_TARGET, return_value="https://mlflow"),
        mock.patch("mlflow.genai.judges.typesafe.get_default_host_creds"),
        mock.patch(
            "mlflow.genai.judges.typesafe.http_request",
            side_effect=[_system_one_response(), reconfigured],
        ),
    ):
        first = _invoke_gateway_judge(
            "gateway:/endpoint", chat_invoker=chat_invoker, **_GATEWAY_JUDGE_KWARGS
        )
        second = _invoke_gateway_judge(
            "gateway:/endpoint", chat_invoker=chat_invoker, **_GATEWAY_JUDGE_KWARGS
        )

    assert first.value is True  # System One
    assert second.value is False  # recovered to chat after reconfiguration
    assert chat_invoker.call_count == 2


def test_structured_builtin_judge_direct_typesafe_skips_chat():
    chat_invoker = mock.Mock()
    with mock.patch(
        "mlflow.genai.judges.structured_judge._invoke_typesafe_judge",
        return_value=Feedback(name="safety", value="yes"),
    ) as mock_ts:
        _invoke_structured_builtin_judge(
            "typesafe:/jev-latest",
            chat_invoker=chat_invoker,
            decision_invoke_params=_BUILTIN_JUDGE_KWARGS,
        )

    mock_ts.assert_called_once()
    # Default built-in output type is the yes/no categorical that System One supports.
    assert mock_ts.call_args.kwargs["feedback_value_type"] == Literal["yes", "no"]
    chat_invoker.assert_not_called()


def test_structured_builtin_judge_non_gateway_uses_chat():
    chat_feedback = Feedback(name="safety", value="yes")
    chat_invoker = mock.Mock(return_value=chat_feedback)
    result = _invoke_structured_builtin_judge(
        "openai:/gpt-4o-mini",
        chat_invoker=chat_invoker,
        decision_invoke_params=_BUILTIN_JUDGE_KWARGS,
    )

    chat_invoker.assert_called_once()
    assert result is chat_feedback


def test_structured_builtin_judge_gateway_falls_back_to_system_one():
    chat_invoker = mock.Mock(side_effect=_system_one_chat_rejection())
    with (
        mock.patch(_CACHE_URI_TARGET, return_value="https://mlflow"),
        mock.patch(
            "mlflow.genai.judges.structured_judge._invoke_typesafe_judge",
            return_value=Feedback(name="safety", value="yes"),
        ) as mock_ts,
    ):
        feedback = _invoke_structured_builtin_judge(
            "gateway:/jev-endpoint",
            chat_invoker=chat_invoker,
            decision_invoke_params=_BUILTIN_JUDGE_KWARGS,
        )

    chat_invoker.assert_called_once()  # chat-first
    mock_ts.assert_called_once()  # then System One fallback
    assert feedback.value == "yes"
    # The endpoint is now cached as System One so later rows skip the chat attempt.
    assert len(_gateway_system_one_cache) == 1


def test_structured_builtin_judge_threads_inference_params_through_gateway_fallback():
    # Scorers (builtin_scorers.py) forward inference_params; confirm it survives the
    # dispatcher -> _invoke_gateway_judge (opaque **kwargs) -> _invoke_typesafe_judge hop.
    chat_invoker = mock.Mock(side_effect=_system_one_chat_rejection())
    inference_params = {"temperature": 0.0}
    with (
        mock.patch(_CACHE_URI_TARGET, return_value="https://mlflow"),
        mock.patch(
            "mlflow.genai.judges.structured_judge._invoke_typesafe_judge",
            return_value=Feedback(name="safety", value="yes"),
        ) as mock_ts,
    ):
        _invoke_structured_builtin_judge(
            "gateway:/jev-endpoint",
            chat_invoker=chat_invoker,
            decision_invoke_params={**_BUILTIN_JUDGE_KWARGS, "inference_params": inference_params},
        )

    assert mock_ts.call_args.kwargs["inference_params"] == inference_params
