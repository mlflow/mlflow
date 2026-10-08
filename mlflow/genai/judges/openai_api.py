"""OpenAI Responses and Decisions invocation for instructions judges."""

from __future__ import annotations

import inspect
import json
import math
import os
import re
from typing import Any, Literal, get_args, get_origin

import pydantic

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.metrics.genai.model_utils import _parse_model_uri
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INTERNAL_ERROR
from mlflow.tracing.constant import AssessmentMetadataKey

_QUESTION_NAME = "evaluation"
_REFERENCE_PATTERN = re.compile(r"\{\{\s*([a-z_]+)\s*\}\}")
_RESPONSES_INFERENCE_PARAMS = frozenset({
    "max_output_tokens",
    "temperature",
    "top_p",
    "reasoning",
    "verbosity",
    "service_tier",
    "truncation",
    "safety_identifier",
    "prompt_cache_key",
    "prompt_cache_options",
    "prompt_cache_retention",
    "store",
    "user",
})


def _openai_client(base_url: str | None, extra_headers: dict[str, str] | None):
    try:
        import openai
    except ImportError:
        raise MlflowException.invalid_parameter_value(
            "Install the openai package to use OpenAI Responses or Decisions judges."
        ) from None

    if not hasattr(openai, "OpenAI"):
        raise MlflowException.invalid_parameter_value(
            "Update the openai package to use OpenAI Responses or Decisions judges."
        )

    client_kwargs = {}
    if (
        api_base := base_url
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
    ):
        client_kwargs["base_url"] = api_base
    if extra_headers:
        client_kwargs["default_headers"] = extra_headers
        # The SDK requires an API key even when a custom Authorization header supplies it.
        if not os.environ.get("OPENAI_API_KEY") and any(
            key.lower() == "authorization" for key in extra_headers
        ):
            client_kwargs["api_key"] = "unused"
    try:
        return openai.OpenAI(**client_kwargs)
    except openai.OpenAIError as e:
        raise MlflowException.invalid_parameter_value(str(e)) from e


def _model_name(model_uri: str) -> str:
    provider, model_name = _parse_model_uri(model_uri)
    if provider != "openai":
        raise MlflowException.invalid_parameter_value(
            "OpenAI Responses and Decisions require an openai:/ model URI."
        )
    return model_name


def _feedback(
    model_uri: str,
    assessment_name: str,
    value: Any,
    rationale: str | None,
    metadata: dict[str, str],
) -> Feedback:
    return Feedback(
        name=assessment_name,
        value=value,
        rationale=rationale,
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id=model_uri,
        ),
        metadata=metadata,
    )


def _usage_metadata(usage: Any) -> dict[str, str]:
    if usage is None:
        return {}
    metadata = {}
    for field, key in (
        ("input_tokens", AssessmentMetadataKey.JUDGE_INPUT_TOKENS),
        ("output_tokens", AssessmentMetadataKey.JUDGE_OUTPUT_TOKENS),
    ):
        count = getattr(usage, field, None)
        if count is not None:
            if type(count) is not int or count < 0:
                raise _invalid_response("usage")
            metadata[key] = str(count)
    return metadata


def _invalid_response(subject: str) -> MlflowException:
    return MlflowException(
        f"OpenAI {subject} returned an invalid response.", error_code=BAD_REQUEST
    )


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def _call_openai(action: str, callback):
    import openai

    try:
        return callback()
    except openai.APIStatusError as e:
        error_code = BAD_REQUEST if e.status_code < 500 else INTERNAL_ERROR
        raise MlflowException(
            f"OpenAI {action} request failed with HTTP {e.status_code}.", error_code=error_code
        ) from e
    except openai.APIConnectionError as e:
        raise MlflowException(
            f"Could not connect to the OpenAI {action} API.", error_code=INTERNAL_ERROR
        ) from e


def _invoke_openai_responses_judge(
    model_uri: str,
    *,
    system_prompt: str,
    user_prompt: str,
    response_format: type[pydantic.BaseModel],
    assessment_name: str,
    inference_params: dict[str, Any] | None = None,
    base_url: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> Feedback:
    """Use structured Responses output for a field-based judge."""
    client = _openai_client(base_url, extra_headers)
    if not callable(getattr(getattr(client, "responses", None), "parse", None)):
        raise MlflowException.invalid_parameter_value(
            "Update the openai package to use model_api='responses'."
        )

    params = dict(inference_params or {})
    if unsupported := set(params).difference(_RESPONSES_INFERENCE_PARAMS):
        raise MlflowException.invalid_parameter_value(
            "Unsupported inference_params for an OpenAI Responses judge: "
            f"{', '.join(sorted(unsupported))}."
        )
    parse = client.responses.parse
    try:
        signature = inspect.signature(parse)
    except (TypeError, ValueError) as e:
        raise MlflowException.invalid_parameter_value(
            "Could not determine which Responses parameters the installed openai package supports."
        ) from e
    if not any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    ):
        if unavailable := set(params).difference(signature.parameters):
            raise MlflowException.invalid_parameter_value(
                "The installed openai package does not support these Responses inference_params: "
                f"{', '.join(sorted(unavailable))}. Update openai or remove these parameters."
            )

    response = _call_openai(
        "Responses",
        lambda: parse(
            model=_model_name(model_uri),
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=response_format,
            **params,
        ),
    )
    if getattr(response, "status", None) != "completed" or not isinstance(
        getattr(response, "output_parsed", None), response_format
    ):
        raise MlflowException(
            "OpenAI Responses did not return a completed structured evaluation. "
            "The model may have refused the request.",
            error_code=BAD_REQUEST,
        )
    # The SDK's parsed model can coerce a string like "false" to bool. Validate the raw
    # structured output strictly so a malformed answer cannot become normal judge feedback.
    try:
        raw = json.loads(response.output_text, parse_constant=_reject_nonfinite)
        if not isinstance(raw, dict) or set(raw) != {"result", "rationale"}:
            raise ValueError
        parsed = response_format.model_validate(raw, strict=True)
    except (AttributeError, TypeError, ValueError, pydantic.ValidationError):
        raise _invalid_response("Responses") from None
    metadata = _usage_metadata(getattr(response, "usage", None))
    if isinstance(model := getattr(response, "model", None), str) and model:
        metadata["openai.responses.model"] = model
    return _feedback(model_uri, assessment_name, parsed.result, parsed.rationale, metadata)


def _decision_question(instructions: str, feedback_value_type: Any) -> dict[str, Any]:
    instructions = _REFERENCE_PATTERN.sub(
        lambda match: f'the "{match.group(1)}" field in the input', instructions
    )
    question = {"name": _QUESTION_NAME, "instructions": instructions}
    if feedback_value_type is bool:
        return {**question, "type": "predicate"}

    if get_origin(feedback_value_type) is not Literal:
        raise MlflowException.invalid_parameter_value(
            "OpenAI Decisions judges require feedback_value_type=bool or a "
            "Literal of strings or booleans."
        )
    values = get_args(feedback_value_type)
    if len(values) < 2 or any(type(value) not in (str, bool) for value in values):
        raise MlflowException.invalid_parameter_value(
            "OpenAI Decisions choice judges require at least two string or boolean Literal values."
        )
    return {**question, "type": "choice", "choices": [{"value": value} for value in values]}


def _validate_responses_feedback_value_type(feedback_value_type: Any) -> None:
    if get_origin(feedback_value_type) is dict:
        raise MlflowException.invalid_parameter_value(
            "OpenAI Responses structured outputs do not support dict feedback value types. "
            "Use a primitive, Literal, or list feedback value type."
        )


def _probability(value: Any) -> float:
    if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
        raise _invalid_response("Decisions")
    return float(value)


def _same_choice(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _parse_decision_answer(answer: Any, question: dict[str, Any]) -> tuple[Any, dict[str, str]]:
    if getattr(answer, "name", None) != _QUESTION_NAME:
        raise _invalid_response("Decisions")
    answer_type = getattr(answer, "type", None)
    if answer_type == "refusal":
        raise MlflowException(
            "OpenAI Decisions refused the evaluation question.", error_code=BAD_REQUEST
        )
    if answer_type != question["type"]:
        raise _invalid_response("Decisions")

    if answer_type == "predicate":
        probability = _probability(getattr(answer, "probability", None))
        return probability >= 0.5, {"openai.decisions.probability": json.dumps(probability)}

    choices = [entry["value"] for entry in question["choices"]]
    selected = getattr(answer, "choice", None)
    if not any(_same_choice(selected, choice) for choice in choices):
        raise _invalid_response("Decisions")
    confidence = _probability(getattr(answer, "confidence", None))
    probabilities = getattr(answer, "probabilities", None)
    if not isinstance(probabilities, list) or len(probabilities) != len(choices):
        raise _invalid_response("Decisions")
    parsed = []
    for choice in choices:
        matches = [
            item for item in probabilities if _same_choice(getattr(item, "value", None), choice)
        ]
        if len(matches) != 1:
            raise _invalid_response("Decisions")
        parsed.append({
            "value": choice,
            "probability": _probability(getattr(matches[0], "probability", None)),
        })
    if not math.isclose(sum(item["probability"] for item in parsed), 1, abs_tol=1e-3):
        raise _invalid_response("Decisions")
    return selected, {
        "openai.decisions.confidence": json.dumps(confidence),
        "openai.decisions.probabilities": json.dumps(parsed, ensure_ascii=False),
    }


def _invoke_openai_decisions_judge(
    model_uri: str,
    *,
    instructions: str,
    input_text: str,
    feedback_value_type: Any,
    assessment_name: str,
    base_url: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> Feedback:
    """Use a predicate or finite choice from the Decisions API as judge feedback."""
    client = _openai_client(base_url, extra_headers)
    if not callable(getattr(getattr(client, "decisions", None), "create", None)):
        raise MlflowException.invalid_parameter_value(
            "Update the openai package to a version with Decisions API support."
        )
    question = _decision_question(instructions, feedback_value_type)
    response = _call_openai(
        "Decisions",
        lambda: client.decisions.create(
            model=_model_name(model_uri), input=input_text, questions=[question]
        ),
    )
    answers = getattr(response, "answers", None)
    if not isinstance(answers, list) or len(answers) != 1:
        raise _invalid_response("Decisions")
    value, metadata = _parse_decision_answer(answers[0], question)
    model = getattr(response, "model", None)
    if not isinstance(model, str) or not model:
        raise _invalid_response("Decisions")
    metadata["openai.decisions.model"] = model
    usage = getattr(response, "usage", None)
    if usage is None:
        raise _invalid_response("Decisions")
    usage_metadata = _usage_metadata(usage)
    if AssessmentMetadataKey.JUDGE_INPUT_TOKENS not in usage_metadata or (
        AssessmentMetadataKey.JUDGE_OUTPUT_TOKENS not in usage_metadata
    ):
        raise _invalid_response("Decisions")
    metadata.update(usage_metadata)
    return _feedback(model_uri, assessment_name, value, None, metadata)
