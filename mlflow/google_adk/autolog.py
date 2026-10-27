import json
import logging
import threading

from google.adk.plugins.base_plugin import BasePlugin
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import (
    NoOpTracer,
    NoOpTracerProvider,
    ProxyTracerProvider,
    get_tracer_provider,
    set_tracer_provider,
)

from mlflow.entities import SpanType
from mlflow.entities.span import create_mlflow_span
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.provider import (
    _add_span_processor_first,
    _get_tracer_for_span_end,
    _get_tracer_for_span_start,
)
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import (
    _bypass_attribute_guard,
    get_mlflow_span_for_otel_span,
    get_otel_attribute,
)

_logger = logging.getLogger(__name__)

# ADK pauses a tool behind a confirmation card by calling this function; the user answers it
# with a function response of the same name.
_CONFIRMATION_CALL = "adk_request_confirmation"
# When one model response calls several tools, ADK adds a span for the merged responses whose
# arguments are the placeholder "N/A". It is not a tool call.
_MERGED_TOOLS_NAME = "(merged tools)"
_PLUGIN_NAME = "mlflow_google_adk"

_SPAN_TYPE_BY_OPERATION = {
    "invoke_agent": SpanType.AGENT,
    "invoke_workflow": SpanType.AGENT,
    "execute_tool": SpanType.TOOL,
    "generate_content": SpanType.LLM,
}


class GoogleADKSpanProcessor(SimpleSpanProcessor):
    """Forwards ADK's OpenTelemetry spans to MLflow's tracer and sets MLflow span types."""

    def __init__(self):
        self.span_exporter = SpanExporter()
        self._processing_local = threading.local()

    def on_start(self, span: OTelSpan, parent_context: Context | None = None):
        # Recursion guard: with a shared tracer provider, forwarding re-enters this processor.
        if getattr(self._processing_local, "in_on_start", False):
            return
        self._processing_local.in_on_start = True
        try:
            tracer, should_forward = _get_tracer_for_span_start(self, __name__)
            if isinstance(tracer, NoOpTracer) or not should_forward:
                return
            tracer.span_processor.on_start(span, parent_context)
            trace_id = get_otel_attribute(span, SpanAttributeKey.REQUEST_ID)
            if trace_id is None:
                return
            InMemoryTraceManager.get_instance().register_span(create_mlflow_span(span, trace_id))
        finally:
            self._processing_local.in_on_start = False

    def on_end(self, span: OTelReadableSpan) -> None:
        if getattr(self._processing_local, "in_on_end", False):
            return
        self._processing_local.in_on_end = True
        try:
            if mlflow_span := get_mlflow_span_for_otel_span(span):
                with _bypass_attribute_guard(mlflow_span._span):
                    if span_type := _span_type(span):
                        mlflow_span.set_span_type(span_type)
            tracer, should_forward = _get_tracer_for_span_end(self, __name__)
            if should_forward:
                tracer.span_processor.on_end(span)
        finally:
            self._processing_local.in_on_end = False


def _span_type(span: OTelReadableSpan) -> str | None:
    attributes = span.attributes or {}
    if attributes.get("gen_ai.tool.name") == _MERGED_TOOLS_NAME:
        return SpanType.CHAIN
    if span.name == "invocation":
        return SpanType.AGENT
    if span.name == "call_llm":
        return SpanType.LLM
    return _SPAN_TYPE_BY_OPERATION.get(attributes.get("gen_ai.operation.name"))


def _message_text(content) -> str:
    """The user's side of a turn as text: typed text, or the answer to a confirmation card."""
    texts = []
    for part in (content.parts if content else None) or []:
        if part.text and not part.thought:
            texts.append(part.text)
        elif part.function_response and part.function_response.name == _CONFIRMATION_CALL:
            from google.adk.tools.tool_confirmation import ToolConfirmation

            response = part.function_response.response or {}
            confirmed = ToolConfirmation.from_response_dict(response).confirmed
            texts.append(f"[confirmation {'accepted' if confirmed else 'declined'}]")
    return "".join(texts)


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


class MlflowTurnPlugin(BasePlugin):
    """
    Records each turn's user message and visible reply on ADK's invocation span, which ADK
    leaves without inputs or outputs. MLflow reads the turn from the trace's root span.
    """

    def __init__(self):
        super().__init__(name=_PLUGIN_NAME)
        self._turns = {}

    async def on_user_message_callback(self, *, invocation_context, user_message):
        try:
            _set_json_attribute(
                SpanAttributeKey.INPUTS,
                {"messages": [{"role": "user", "content": _message_text(user_message)}]},
            )
            self._turns[invocation_context.invocation_id] = {"replies": [], "cards": []}
        except Exception as e:
            _logger.debug("Failed to record the ADK turn input", exc_info=e)

    async def on_event_callback(self, *, invocation_context, event):
        try:
            if turn := self._turns.get(invocation_context.invocation_id):
                turn["cards"] += [
                    call.args["originalFunctionCall"]["name"]
                    for call in event.get_function_calls()
                    if call.name == _CONFIRMATION_CALL
                ]
                if reply := _visible_reply(event):
                    turn["replies"].append(reply)
        except Exception as e:
            _logger.debug("Failed to record the ADK turn output", exc_info=e)

    async def after_run_callback(self, *, invocation_context):
        try:
            if turn := self._turns.pop(invocation_context.invocation_id, None):
                reply = "\n".join(turn["replies"])
                if not reply and turn["cards"]:
                    reply = f"[confirmation requested: {', '.join(turn['cards'])}]"
                _set_json_attribute(SpanAttributeKey.OUTPUTS, reply)
        except Exception as e:
            _logger.debug("Failed to record the ADK turn output", exc_info=e)


def _set_json_attribute(key: str, value) -> None:
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute(key, json.dumps(value, ensure_ascii=False))


def patched_runner_init(original, self, *args, **kwargs):
    original(self, *args, **kwargs)
    plugin_manager = getattr(self, "plugin_manager", None)
    if plugin_manager is not None and plugin_manager.get_plugin(_PLUGIN_NAME) is None:
        plugin_manager.register_plugin(MlflowTurnPlugin())


def setup_google_adk_tracing():
    processor = GoogleADKSpanProcessor()
    provider = get_tracer_provider()
    if isinstance(provider, (NoOpTracerProvider, ProxyTracerProvider)):
        new_provider = SDKTracerProvider()
        new_provider.add_span_processor(processor)
        set_tracer_provider(new_provider)
    elif not any(
        isinstance(p, GoogleADKSpanProcessor)
        for p in provider._active_span_processor._span_processors
    ):
        _add_span_processor_first(provider, processor)


def teardown_google_adk_tracing():
    provider = get_tracer_provider()
    if isinstance(provider, SDKTracerProvider):
        span_processors = getattr(provider._active_span_processor, "_span_processors", ())
        provider._active_span_processor._span_processors = tuple(
            p for p in span_processors if not isinstance(p, GoogleADKSpanProcessor)
        )
