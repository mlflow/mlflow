"""Native TypeSafe System One judge invocation."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin

import requests

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.environment_variables import MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import (
    TYPESAFE_API_BASE_URL,
    TYPESAFE_SYSTEM_ONE_PATH,
)
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    UNAUTHENTICATED,
)
from mlflow.telemetry.events import InvokeCustomJudgeModelEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.tracing.constant import AssessmentMetadataKey
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils.request_utils import _get_http_response_with_retries

_DIRECT_ENDPOINT = f"{TYPESAFE_API_BASE_URL}/{TYPESAFE_SYSTEM_ONE_PATH}"
_RETRY_CODES = (408, 429, 500, 502, 503, 504, 529)
_QUESTION_NAME = "evaluation"
_STATE_REFERENCE_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_TRACE_REFERENCE_PATTERN = re.compile(r"\{\{\s*trace\s*\}\}")


@dataclass
class _NoulSpec:
    answer_type: Literal["noul"] = "noul"


@dataclass
class _ChoiceSpec:
    choices: dict[str, Any]
    answer_type: Literal["choice"] = "choice"


_AnswerSpec = _NoulSpec | _ChoiceSpec


def _is_typesafe_model(model_uri: str) -> bool:
    provider, separator, _ = model_uri.partition(":/")
    return bool(separator) and provider == "typesafe"


@record_usage_event(InvokeCustomJudgeModelEvent)
def _invoke_typesafe_judge(
    model_uri: str,
    *,
    instructions: str,
    state: dict[str, Any],
    feedback_value_type: Any,
    assessment_name: str,
    num_retries: int = 10,
    inference_params: dict[str, Any] | None = None,
    base_url: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> Feedback:
    """Invoke a TypeSafe model through its native System One evaluation API."""
    from mlflow.metrics.genai.model_utils import _parse_model_uri

    provider, model_name = _parse_model_uri(model_uri)
    if provider != "typesafe":
        raise MlflowException.invalid_parameter_value(
            f"Expected a typesafe:/ model URI, got {model_uri!r}."
        )
    _validate_options(inference_params, base_url, extra_headers)
    _validate_input(instructions, state)

    question, answer_spec = _build_question(feedback_value_type)
    question["instructions"] = _rewrite_state_references(instructions, state)
    payload = {
        "model": model_name,
        "state": _serialize_state(state),
        "questions": {_QUESTION_NAME: question},
    }

    response = _send_request(payload, num_retries)
    response_data = _parse_json_response(response)
    value, metadata, _, _ = _parse_response(response_data, answer_spec)

    return Feedback(
        name=assessment_name,
        value=value,
        rationale=None,
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id=model_uri,
        ),
        metadata=metadata,
    )


def _validate_options(
    inference_params: dict[str, Any] | None,
    base_url: str | None,
    extra_headers: dict[str, str] | None,
) -> None:
    unsupported_options = [
        name
        for name, value in (
            ("inference_params", inference_params),
            ("base_url", base_url),
            ("extra_headers", extra_headers),
        )
        if value is not None
    ]
    if unsupported_options:
        raise MlflowException.invalid_parameter_value(
            "TypeSafe judge models do not support " + ", ".join(unsupported_options) + "."
        )


def _validate_input(instructions: str, state: dict[str, Any]) -> None:
    if not isinstance(instructions, str) or not instructions.strip():
        raise MlflowException(
            "TypeSafe evaluation instructions must be a non-empty string.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if _TRACE_REFERENCE_PATTERN.search(instructions):
        raise MlflowException(
            "TypeSafe judge models do not support trace-based tool calling. Use structured "
            "inputs, outputs, expectations, or conversation state instead.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if not isinstance(state, dict):
        raise MlflowException(
            "TypeSafe evaluation state must be a dictionary.",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _send_request(payload: dict[str, Any], num_retries: int):
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise MlflowException(
            "Set TYPESAFE_API_KEY to invoke a typesafe:/ judge model.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    try:
        return _get_http_response_with_retries(
            method="POST",
            url=_DIRECT_ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            max_retries=num_retries,
            backoff_factor=1,
            backoff_jitter=0.1,
            retry_codes=_RETRY_CODES,
            timeout=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS.get(),
            raise_on_status=False,
            allow_redirects=False,
        )
    except requests.RequestException:
        raise MlflowException(
            "Failed to connect to the TypeSafe evaluation endpoint.",
            error_code=INTERNAL_ERROR,
        ) from None


def _rewrite_state_references(instructions: str, state: dict[str, Any]) -> str:
    def replace(match: re.Match) -> str:
        field = match.group(1)
        return f"state.{field}" if field in state else match.group(0)

    return _STATE_REFERENCE_PATTERN.sub(replace, instructions)


def _serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    try:
        serialized = json.dumps(state, cls=TraceJSONEncoder, allow_nan=False)
        return json.loads(serialized)
    except (TypeError, ValueError):
        raise MlflowException(
            "TypeSafe evaluation state must contain JSON-serializable values without NaN or "
            "infinity.",
            error_code=INVALID_PARAMETER_VALUE,
        ) from None


def _build_question(feedback_value_type: Any) -> tuple[dict[str, Any], _AnswerSpec]:
    if feedback_value_type is bool:
        return {"type": "noul"}, _NoulSpec()

    if get_origin(feedback_value_type) is not Literal:
        raise MlflowException(
            "TypeSafe judge models support bool or finite Literal feedback value types. "
            f"Got {feedback_value_type!r}.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    values = get_args(feedback_value_type)
    if not 1 <= len(values) <= 255:
        raise MlflowException(
            "TypeSafe Literal feedback value types must contain between 1 and 255 values.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    for value in values:
        _validate_literal_value(value)

    choices: dict[str, Any] = {}
    criteria: dict[str, str] = {}
    for value in values:
        label = _literal_label(value)
        if not label.strip():
            raise MlflowException(
                "TypeSafe Literal feedback values must produce non-empty choice labels.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if label in choices:
            raise MlflowException(
                "TypeSafe Literal feedback values must have distinct string representations. "
                f"The label {label!r} is ambiguous.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        choices[label] = value
        criteria[label] = _describe_literal(value)
    return (
        {"type": "choice", "criteria": criteria},
        _ChoiceSpec(choices=choices),
    )


def _validate_literal_value(value: Any) -> None:
    if type(value) not in (str, int, float, bool):
        raise MlflowException(
            "TypeSafe Literal feedback values must be JSON string, number, or boolean scalars.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if type(value) is float and not math.isfinite(value):
        raise MlflowException(
            "TypeSafe Literal feedback values cannot contain NaN or infinity.",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _literal_label(value: Any) -> str:
    if type(value) is str:
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _describe_literal(value: Any) -> str:
    return f"The evaluation result is {_literal_label(value)}."


def _parse_json_response(response) -> dict[str, Any]:
    if not 200 <= response.status_code < 300:
        if response.status_code in (401, 403):
            error_code = UNAUTHENTICATED
        elif response.status_code < 500:
            error_code = BAD_REQUEST
        else:
            error_code = INTERNAL_ERROR
        raise MlflowException(
            f"TypeSafe evaluation failed with HTTP {response.status_code}. Check the endpoint "
            "configuration, credentials, and evaluation schema.",
            error_code=error_code,
        )
    try:
        response_data = response.json()
    except ValueError:
        raise MlflowException(
            "TypeSafe evaluation returned invalid JSON.", error_code=BAD_REQUEST
        ) from None
    if not isinstance(response_data, dict):
        raise MlflowException(
            "TypeSafe evaluation returned an invalid response.", error_code=BAD_REQUEST
        )
    return response_data


def _parse_response(
    response: dict[str, Any], answer_spec: _AnswerSpec
) -> tuple[Any, dict[str, str], int | None, int | None]:
    try:
        model = response["model"]
        answer = response["answers"][_QUESTION_NAME]
        if not isinstance(model, str) or not model or not isinstance(answer, dict):
            raise ValueError
        if answer.get("type") != answer_spec.answer_type:
            raise ValueError

        metadata = {"typesafe.model": model}
        match answer_spec:
            case _NoulSpec():
                probability = _number(answer["noul"], minimum=0, maximum=1)
                metadata["typesafe.probability"] = json.dumps(probability)
                value = probability >= 0.5
            case _ChoiceSpec(choices=choices):
                choice = answer["choice"]
                if not isinstance(choice, str) or choice not in choices:
                    raise ValueError
                probabilities = _probabilities(answer["probabilities"], set(choices))
                confidence = _number(answer["confidence"], minimum=0, maximum=1)
                metadata["typesafe.probabilities"] = json.dumps(
                    probabilities, ensure_ascii=False, sort_keys=True, allow_nan=False
                )
                metadata["typesafe.confidence"] = json.dumps(confidence)
                if "legend" in answer:
                    legend = answer["legend"]
                    if (
                        not isinstance(legend, dict)
                        or set(legend) != set(choices)
                        or any(not isinstance(key, str) for key in legend)
                        or any(not isinstance(description, str) for description in legend.values())
                    ):
                        raise ValueError
                    metadata["typesafe.legend"] = json.dumps(
                        legend, ensure_ascii=False, sort_keys=True
                    )
                value = choices[choice]

        input_tokens, output_tokens = _parse_usage(response.get("usage"), metadata)
        return value, metadata, input_tokens, output_tokens
    except (KeyError, TypeError, ValueError, AttributeError):
        raise _invalid_response(answer_spec.answer_type) from None


def _number(value: Any, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float) or not minimum <= value <= maximum:
        raise ValueError
    if type(value) is float and not math.isfinite(value):
        raise ValueError
    return float(value)


def _probabilities(value: Any, expected_keys: set[str]) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError
    probabilities = {
        key: _number(probability, minimum=0, maximum=1) for key, probability in value.items()
    }
    if not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-3):
        raise ValueError
    return probabilities


def _parse_usage(usage: Any, metadata: dict[str, str]) -> tuple[int | None, int | None]:
    if usage is None:
        return None, None
    if not isinstance(usage, dict):
        raise ValueError

    counts = []
    for field, metadata_key in (
        ("input_tokens", AssessmentMetadataKey.JUDGE_INPUT_TOKENS),
        ("output_tokens", AssessmentMetadataKey.JUDGE_OUTPUT_TOKENS),
    ):
        count = usage.get(field)
        if count is not None:
            if type(count) is not int or count < 0:
                raise ValueError
            metadata[metadata_key] = str(count)
        counts.append(count)
    return counts[0], counts[1]


def _invalid_response(answer_type: str) -> MlflowException:
    return MlflowException(
        f"TypeSafe evaluation returned an invalid {answer_type} answer.",
        error_code=BAD_REQUEST,
    )


__all__ = ["_invoke_typesafe_judge", "_is_typesafe_model"]
