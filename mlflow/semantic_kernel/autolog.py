import json
import logging
import os
from types import MappingProxyType
from typing import Any, Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import (
    NoOpTracerProvider,
    ProxyTracerProvider,
    get_current_span,
    get_tracer_provider,
    set_tracer_provider,
)
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.streaming_content_mixin import StreamingContentMixin
from semantic_kernel.utils.telemetry.model_diagnostics import (
    gen_ai_attributes as model_gen_ai_attributes,
)
from semantic_kernel.utils.telemetry.model_diagnostics.decorators import (
    CHAT_COMPLETION_OPERATION,
    CHAT_STREAMING_COMPLETION_OPERATION,
    TEXT_COMPLETION_OPERATION,
    TEXT_STREAMING_COMPLETION_OPERATION,
    are_sensitive_events_enabled,
)

from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.constant import (
    SpanAttributeKey,
    TokenUsageKey,
)
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.provider import _setup_tracer_provider as setup_mlflow_tracing_provider
from mlflow.tracing.provider import detach_span_from_context, set_span_in_context

_logger = logging.getLogger(__name__)


def _set_logging_env_variables():
    # NB: these environment variables are required to enable the telemetry for
    # genai fields in Semantic Kernel, which are currently marked as experimental.
    # https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/observability/telemetry-with-console
    os.environ["SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS"] = "true"
    os.environ["SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE"] = "true"

    # Reset the diagnostics module which is initialized at import time
    from semantic_kernel.utils.telemetry.model_diagnostics.decorators import (
        MODEL_DIAGNOSTICS_SETTINGS,
    )

    MODEL_DIAGNOSTICS_SETTINGS.enable_otel_diagnostics = (
        os.getenv("SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS", "").lower() == "true"
    )

    MODEL_DIAGNOSTICS_SETTINGS.enable_otel_diagnostics_sensitive = (
        os.getenv("SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE", "").lower()
        == "true"
    )


def setup_semantic_kernel_tracing():
    _set_logging_env_variables()

    if isinstance(get_tracer_provider(), (NoOpTracerProvider, ProxyTracerProvider)):
        _logger.info(
            "An otel tracer was not found so we will use the default MLflow tracer provider."
        )
        setup_mlflow_tracing_provider()

        from mlflow.tracing.provider import _MLFLOW_TRACER_PROVIDER

        _MLFLOW_TRACER_PROVIDER._active_span_processor._span_processors = (
            SemanticKernelSpanProcessor(),
        )
        set_tracer_provider(_MLFLOW_TRACER_PROVIDER)

    else:
        _logger.info(
            "The semantic kernel tracer is already initialized. Using the existing tracer provider."
        )
        get_tracer_provider().add_span_processor(SemanticKernelSpanProcessor())

    sk_span_processor = next(x for x in get_tracer_provider()._active_span_processor._span_processors if isinstance(x, SemanticKernelSpanProcessor))
    print(f"setup_semantic_kernel_tracing tracer provider memory id: {id(get_tracer_provider())}")
    print(f"setup_semantic_kernel_tracing span processor memory id: {id(sk_span_processor)}")


class DummySpanExporter:
    # NB: Dummy NoOp exporter that does nothing, because OTel span processor requires an exporter
    def on_end(self, span: OTelReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        pass


class SemanticKernelSpanProcessor(SimpleSpanProcessor):
    def __init__(self):
        self._otel_span_id_to_mlflow_span_and_token = {}
        self.span_exporter = DummySpanExporter()

    def on_start(self, span: OTelSpan, parent_context: Optional[Context] = None):
        print(f"[on_start] {span.name}")

        otel_span_id = span.get_span_context().span_id
        parent_span_id = span.parent.span_id if span.parent else None

        parent_st = self._otel_span_id_to_mlflow_span_and_token.get(parent_span_id)
        parent_span = parent_st[0] if parent_st else None

        mlflow_span = start_span_no_context(
            name=span.name,
            parent_span=parent_span,
            attributes=span.attributes,
        )
        token = set_span_in_context(mlflow_span)
        self._otel_span_id_to_mlflow_span_and_token[otel_span_id] = (mlflow_span, token)
        print("[on_start] complete")
        print(self._otel_span_id_to_mlflow_span_and_token)

    def on_end(self, span: OTelReadableSpan) -> None:
        print(f"[on_end] {span.name}")
        st = self._otel_span_id_to_mlflow_span_and_token.pop(span.get_span_context().span_id, None)
        if st is None:
            _logger.debug("Span not found in the map. Skipping end.")
            return

        mlflow_span, token = st
        attributes = (
            dict(span.attributes)
            if isinstance(span.attributes, MappingProxyType)
            else span.attributes
        )
        mlflow_span.set_attributes(attributes)
        _set_token_usage(mlflow_span, attributes)

        if mlflow_span.span_type or mlflow_span.span_type == SpanType.UNKNOWN:
            mlflow_span.set_span_type(_get_span_type(span))

        detach_span_from_context(token)
        mlflow_span.end()


def _get_live_span_from_otel_span_id(otel_span_id: str) -> LiveSpan:
    tracer_provider = get_tracer_provider()
    print(f"tracer_provider memory id: {id(tracer_provider)}")
    span_processors = tracer_provider._active_span_processor._span_processors

    otel_to_live_span_map = {}
    for span_processor in span_processors:
        if isinstance(span_processor, SemanticKernelSpanProcessor):
            print(f"_get_live_span_from_otel_span_id span processor memory id: {id(span_processor)}")
            otel_to_live_span_map = span_processor._otel_span_id_to_mlflow_span_and_token

            if span_and_token := otel_to_live_span_map.get(otel_span_id):
                return span_and_token[0]
            else:
                _logger.warning(
                    f"Live span not found for OTel span ID: {otel_span_id}. "
                    "Cannot map OTel span ID to MLflow span ID, so we will skip registering "
                    "additional attributes. "
                    f"Available OTel span IDs: {list(otel_to_live_span_map.keys())}"

                )
                return None
            
    _logger.warning(
        "No SemanticKernelSpanProcessor found. "
        "Cannot map OTel span ID to MLflow span ID, so we will skip registering "
        "additional attributes."
    )

    return None


def _get_span_type(span: OTelSpan) -> str:
    span_type = None

    if hasattr(span, "attributes") and (
        operation := span.attributes.get(model_gen_ai_attributes.OPERATION)
    ):
        span_map = {
            CHAT_COMPLETION_OPERATION: SpanType.CHAT_MODEL,
            CHAT_STREAMING_COMPLETION_OPERATION: SpanType.CHAT_MODEL,
            TEXT_COMPLETION_OPERATION: SpanType.LLM,
            TEXT_STREAMING_COMPLETION_OPERATION: SpanType.LLM,
        }
        span_type = span_map.get(operation)

    return span_type or SpanType.UNKNOWN


def _set_token_usage(mlflow_span: LiveSpan, sk_attributes: dict[str, Any]) -> None:
    if value := sk_attributes.get(model_gen_ai_attributes.INPUT_TOKENS):
        mlflow_span.set_attribute(TokenUsageKey.INPUT_TOKENS, value)
    if value := sk_attributes.get(model_gen_ai_attributes.OUTPUT_TOKENS):
        mlflow_span.set_attribute(TokenUsageKey.OUTPUT_TOKENS, value)

    if (input_tokens := sk_attributes.get(model_gen_ai_attributes.INPUT_TOKENS)) and (
        output_tokens := sk_attributes.get(model_gen_ai_attributes.OUTPUT_TOKENS)
    ):
        mlflow_span.set_attribute(TokenUsageKey.TOTAL_TOKENS, input_tokens + output_tokens)


def _semantic_kernel_chat_completion_input_wrapper(original, *args, **kwargs) -> None:
    print(f"[wrapper] {get_current_span().get_span_context().span_id}")

    # NB: Semantic Kernel logs chat completions, so we need to extract it and add it to the span.
    try:
        prompt = args[1] if len(args) > 1 else kwargs.get("prompt")

        if isinstance(prompt, ChatHistory):
            prompt_value = [msg.to_dict() for msg in prompt.messages]
        elif not isinstance(prompt, list):
            prompt_value = [prompt]
        else:
            prompt_value = prompt

        prompt_value_with_message = {"messages": prompt_value}

        otel_span_id = get_current_span().get_span_context().span_id

        if mlflow_span := _get_live_span_from_otel_span_id(otel_span_id):
            mlflow_span.set_span_type(SpanType.CHAT_MODEL)
            mlflow_span.set_inputs(json.dumps(prompt_value_with_message))
        else:
            _logger.debug(
                "Span is not found or recording. Skipping registering chat "
                f"completion attributes to {SpanAttributeKey.INPUTS}."
            )
    except Exception as e:
        _logger.warning(f"Failed to set inputs attribute: {e}")

    return original(*args, **kwargs)


def _semantic_kernel_chat_completion_response_wrapper(original, *args, **kwargs) -> None:
    print("completionwrappercall")
    # NB: Semantic Kernel logs chat completions, so we need to extract it and add it to the span.
    try:
        current_span = (args[0] if args else kwargs.get("current_span")) or get_current_span()
        completions = args[1] if len(args) > 1 else kwargs.get("completions")

        otel_span_id = current_span.get_span_context().span_id
        mlflow_span = _get_live_span_from_otel_span_id(otel_span_id)
        if not mlflow_span:
            _logger.debug(
                "Span is not found or recording. Skipping registering chat "
                f"completion attributes to {SpanAttributeKey.OUTPUTS}."
            )
            return original(*args, **kwargs)

        if are_sensitive_events_enabled():
            full_responses = []
            for completion in completions:
                full_response: dict[str, Any] = {
                    "message": completion.to_dict(),
                }

                if isinstance(completion, ChatMessageContent):
                    full_response["finish_reason"] = completion.finish_reason.value
                if isinstance(completion, StreamingContentMixin):
                    full_response["index"] = completion.choice_index

                full_responses.append(full_response)

            mlflow_span.set_outputs(json.dumps(full_responses))
            mlflow_span.set_attribute(SpanAttributeKey.CHAT_MESSAGES, json.dumps(full_responses))

    except Exception as e:
        _logger.warning(f"Failed to set outputs attribute: {e}")


def _semantic_kernel_chat_completion_error_wrapper(original, *args, **kwargs) -> None:
    current_span = (args[0] if args else kwargs.get("current_span")) or get_current_span()
    error = args[1] if len(args) > 1 else kwargs.get("error")

    otel_span_id = current_span.get_span_context().span_id
    mlflow_span = _get_live_span_from_otel_span_id(otel_span_id)

    mlflow_span.add_event(SpanEvent.from_exception(error))
    mlflow_span.set_status(SpanStatusCode.ERROR)

    return original(*args, **kwargs)
