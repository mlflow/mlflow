"""Helpers for the Inspect AI scorer integration."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import TYPE_CHECKING, Any, NamedTuple

from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.utils.trace_utils import (
    _to_dict,
    parse_inputs_to_str,
    parse_outputs_to_str,
    resolve_expectations_from_trace,
    resolve_inputs_from_trace,
    resolve_outputs_from_trace,
)
from mlflow.tracing.utils.truncation import _get_last_message, _get_text_content_from_message

if TYPE_CHECKING:
    from inspect_ai.scorer import Score, Scorer, Target
    from inspect_ai.solver import TaskState

_logger = logging.getLogger(__name__)

INSPECT_AI_NOT_INSTALLED_ERROR_MESSAGE = (
    "Inspect AI scorers require the 'inspect-ai' package. Install it with: pip install inspect-ai"
)

# Expectation keys that can supply the Inspect ``Target``, in priority order.
TARGET_EXPECTATION_KEYS = (
    "expected_response",
    "expected_output",
    "reference",
    "expected_facts",
    "target",
)

# Feedback metadata key recording how the Inspect scorer invocation ended.
TERMINAL_STATE_KEY = "terminal_state"

_EVALUATED_MODEL_NAME = "mlflow/evaluated-app"
_DEFAULT_SAMPLE_ID = "mlflow-sample"


class ScoreFields(NamedTuple):
    value: Any
    rationale: str | None
    metadata: dict[str, Any]


def check_inspect_ai_installed() -> None:
    try:
        import inspect_ai  # noqa: F401
    except ImportError as e:
        raise MlflowException.invalid_parameter_value(INSPECT_AI_NOT_INSTALLED_ERROR_MESSAGE) from e


def resolve_target(expectations: dict[str, Any] | None) -> Target | None:
    """Build the Inspect ``Target`` from the first populated target key in expectations."""
    from inspect_ai.scorer import Target

    if not expectations:
        return None
    for key in TARGET_EXPECTATION_KEYS:
        value = expectations.get(key)
        if value is None or (isinstance(value, (str, list, tuple)) and len(value) == 0):
            continue
        if isinstance(value, (list, tuple)):
            return Target([str(item) for item in value])
        return Target(parse_outputs_to_str(value))
    return None


def _message_dicts(value: Any) -> list[dict[str, Any]] | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        messages = _to_dict(value).get("messages")
    except Exception:
        return None
    if isinstance(messages, list) and messages and all(isinstance(m, dict) for m in messages):
        return messages
    return None


def _chat_messages_from_inputs(inputs: Any) -> list[Any]:
    from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser

    builders = {
        "system": ChatMessageSystem,
        "user": ChatMessageUser,
        "assistant": ChatMessageAssistant,
    }
    chat_messages = []
    for message in _message_dicts(inputs) or []:
        role = message.get("role")
        if role not in builders:
            _logger.debug("Skipping message with role %r when building the Inspect state", role)
            continue
        chat_messages.append(builders[role](content=_get_text_content_from_message(message)))
    if chat_messages:
        return chat_messages
    return [ChatMessageUser(content=parse_inputs_to_str(inputs))]


def _last_user_text(inputs: Any) -> str:
    if messages := _message_dicts(inputs):
        return _get_text_content_from_message(_get_last_message(messages, "user"))
    return parse_inputs_to_str(inputs)


def build_task_state(
    *,
    inputs: Any = None,
    outputs: Any = None,
    expectations: dict[str, Any] | None = None,
    trace: Trace | None = None,
    session: list[Trace] | None = None,
) -> tuple[TaskState, Target | None]:
    """Map MLflow scorer arguments onto an Inspect ``TaskState`` and ``Target``.

    With a ``session`` every turn becomes a user and an assistant message and the last turn
    supplies the input and the output. Otherwise inputs, outputs and expectations are
    resolved from the trace when one is given, the inputs become the message history (a
    ``messages`` list is honored, anything else becomes one user message) and the output is
    appended as the assistant message.
    """
    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelName, ModelOutput
    from inspect_ai.solver import TaskState

    if trace is not None:
        inputs = resolve_inputs_from_trace(inputs, trace)
        outputs = resolve_outputs_from_trace(outputs, trace)
        expectations = resolve_expectations_from_trace(expectations, trace)

    if session:
        messages = []
        for turn in session:
            turn_inputs = resolve_inputs_from_trace(None, turn)
            turn_outputs = resolve_outputs_from_trace(None, turn)
            messages.append(ChatMessageUser(content=_last_user_text(turn_inputs)))
            messages.append(
                ChatMessageAssistant(
                    content=parse_outputs_to_str(turn_outputs) if turn_outputs is not None else ""
                )
            )
        input_text = messages[-2].text
        output_text = messages[-1].text
        sample_id = session[-1].info.trace_id
    else:
        input_text = parse_inputs_to_str(inputs) if inputs is not None else ""
        output_text = parse_outputs_to_str(outputs) if outputs is not None else ""
        messages = _chat_messages_from_inputs(inputs) if inputs is not None else []
        if output_text:
            messages.append(ChatMessageAssistant(content=output_text))
        sample_id = trace.info.trace_id if trace is not None else _DEFAULT_SAMPLE_ID

    state = TaskState(
        model=ModelName(_EVALUATED_MODEL_NAME),
        sample_id=sample_id,
        epoch=1,
        input=input_text,
        messages=messages,
        output=ModelOutput.from_content(model=_EVALUATED_MODEL_NAME, content=output_text),
    )
    return state, resolve_target(expectations)


def _is_jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _json_safe(value: Any) -> Any:
    if _is_jsonable(value):
        return value
    return json.loads(json.dumps(value, default=str))


def score_to_feedback_fields(score: Score, threshold: float) -> ScoreFields:
    """Translate an Inspect ``Score`` into the Feedback value, rationale and metadata.

    Scalar values follow Inspect's own ``value_to_float`` mapping (``C`` is 1.0, ``P`` is
    0.5, ``I`` and ``N`` are 0.0, numbers and booleans pass through) and become a ``yes``
    or ``no`` verdict against ``threshold``. List and dict values have no single number to
    compare, so they are passed through as the Feedback value.
    """
    from inspect_ai.scorer import value_to_float

    metadata: dict[str, Any] = {"raw_value": _json_safe(score.value)}
    if isinstance(score.value, (str, int, float, bool)):
        numeric = value_to_float()(score.value)
        value = CategoricalRating.YES if numeric >= threshold else CategoricalRating.NO
        metadata["score"] = numeric
    else:
        value = _json_safe(score.value)
    if score.answer is not None:
        metadata["answer"] = score.answer
    for key, item in (score.metadata or {}).items():
        if _is_jsonable(item):
            metadata[key] = item
        else:
            _logger.debug("Dropping non-serializable Inspect score metadata entry %r", key)
    return ScoreFields(value=value, rationale=score.explanation, metadata=metadata)


def run_scorer(
    scorer: Scorer, state: TaskState, target: Target, timeout: float | None
) -> Score | None:
    """Await an Inspect scorer from synchronous code with an optional timeout in seconds.

    When an event loop is already running, for example inside a notebook, the coroutine runs
    in a worker thread so ``asyncio.run`` never collides with the caller's loop.
    """

    async def _run() -> Score | None:
        awaitable = scorer(state, target)
        if timeout:
            return await asyncio.wait_for(awaitable, timeout=timeout)
        return await awaitable

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    try:
        if loop is not None and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(
                thread_name_prefix="inspect_ai_scorer"
            ) as pool:
                return pool.submit(asyncio.run, _run()).result()
        return asyncio.run(_run())
    except asyncio.TimeoutError as e:
        raise MlflowException(f"Inspect AI scorer timed out after {timeout} seconds.") from e
