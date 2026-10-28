import json
import logging

from google.adk.plugins.base_plugin import BasePlugin
from opentelemetry import trace

from mlflow.google_adk.autolog import PLUGIN_NAME
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import get_otel_attribute

_logger = logging.getLogger(__name__)

# ADK pauses a tool behind a confirmation card by calling this function; the user answers it
# with a function response of the same name.
_CONFIRMATION_CALL = "adk_request_confirmation"


def _confirmed(response: dict) -> bool:
    # ADK's web client wraps the answer as {"response": "<json>"}. Parsed here rather than with
    # ToolConfirmation.from_response_dict, which some ADK releases don't have.
    if set(response) == {"response"} and isinstance(response["response"], str):
        try:
            response = json.loads(response["response"])
        except ValueError:
            return False
    return bool(response.get("confirmed"))


def _message_text(content) -> str:
    """The user's side of a turn as text: typed text, or the answer to a confirmation card."""
    texts, answers = [], []
    for part in (content.parts if content else None) or []:
        if part.text and not part.thought:
            texts.append(part.text)
        elif part.function_response and part.function_response.name == _CONFIRMATION_CALL:
            confirmed = _confirmed(part.function_response.response or {})
            answers.append(f"[confirmation {'accepted' if confirmed else 'declined'}]")
    return " ".join(filter(None, ["".join(texts), *answers]))


def _visible_reply(event) -> str:
    """Text the user sees from one event; same filter as ADK's BigQuery analytics plugin."""
    if not (
        event.content
        and event.content.parts
        and event.is_final_response()
        and event.partial is not True
        and not event.get_function_calls()
        and not event.get_function_responses()
        and not event.long_running_tool_ids
    ):
        return ""
    return "".join(p.text for p in event.content.parts if p.text and not p.thought)


def _latest_turn(events):
    """
    The user's message and the events after it. The last user event marks the start of the turn:
    resuming a paused tool reuses the invocation of the turn that paused it.
    """
    start = max((i for i, event in enumerate(events) if event.author == "user"), default=None)
    if start is None:
        return None, events
    return events[start].content, events[start + 1 :]


def _turn_reply(events) -> str:
    """What the user saw: the visible replies, then any confirmation the turn asks for."""
    replies, cards = [], []
    for event in events:
        cards += [
            call.args["originalFunctionCall"]["name"]
            for call in event.get_function_calls()
            if call.name == _CONFIRMATION_CALL
        ]
        if reply := _visible_reply(event):
            replies.append(reply)
    if cards:
        replies.append(f"[confirmation requested: {', '.join(cards)}]")
    return "\n".join(replies)


class MlflowTurnPlugin(BasePlugin):
    """
    Records each turn's user message and visible reply on ADK's invocation span, which ADK
    leaves without inputs or outputs. MLflow reads the turn from the trace's root span.

    The turn is read from the session when the run ends rather than collected per callback: ADK
    stops dispatching a callback once a plugin returns a replacement, and the session holds the
    message and events that were kept.
    """

    def __init__(self):
        super().__init__(name=PLUGIN_NAME)

    async def on_user_message_callback(self, *, invocation_context, user_message):
        # Recorded early so a turn that fails still shows what the user sent.
        try:
            _set_turn_input(user_message)
            _set_trace_session(invocation_context.session)
        except Exception as e:
            _logger.debug("Failed to record the ADK turn input", exc_info=e)

    async def after_run_callback(self, *, invocation_context):
        try:
            message, events = _latest_turn(invocation_context.session.events)
            _set_turn_input(message)
            _set_trace_session(invocation_context.session)
            _set_json_attribute(SpanAttributeKey.OUTPUTS, _turn_reply(events))
        except Exception as e:
            _logger.debug("Failed to record the ADK turn output", exc_info=e)


def _set_turn_input(content) -> None:
    _set_json_attribute(
        SpanAttributeKey.INPUTS, {"messages": [{"role": "user", "content": _message_text(content)}]}
    )


def _set_trace_session(session) -> None:
    """
    Group the trace under the session of the runner that started it. Runners nested in the turn
    (e.g. an ``AgentTool``) have their own ADK sessions, which would otherwise be picked up from
    their spans.
    """
    span = trace.get_current_span()
    if getattr(span, "parent", None) is not None:
        return
    trace_id = get_otel_attribute(span, SpanAttributeKey.REQUEST_ID)
    if trace_id is None:
        return
    with InMemoryTraceManager.get_instance().get_trace(trace_id) as mlflow_trace:
        if mlflow_trace:
            mlflow_trace.info.trace_metadata[TraceMetadataKey.TRACE_SESSION] = session.id
            mlflow_trace.info.trace_metadata[TraceMetadataKey.TRACE_USER] = session.user_id


def _set_json_attribute(key: str, value) -> None:
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute(key, json.dumps(value, ensure_ascii=False))
