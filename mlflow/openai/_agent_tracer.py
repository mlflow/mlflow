from __future__ import annotations

import json
import logging
from typing import Any

import agents.tracing as oai
from agents import add_trace_processor
from agents._run_impl import TraceCtxManager
from agents.tracing.setup import GLOBAL_TRACE_PROVIDER

from mlflow.entities.span import LiveSpan, SpanType
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatus, SpanStatusCode
from mlflow.openai import FLAVOR_NAME
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.fluent import start_span_no_context
from mlflow.types.chat import (
    ChatTool,
    FunctionToolDefinition,
)
from mlflow.utils.autologging_utils.safety import safe_patch

_logger = logging.getLogger(__name__)


class OpenAISpanType:
    """
    https://github.com/openai/openai-agents-python/blob/ca8e8bed5d0f33e8a0bc3eabd5f1b0a183e73765/src/agents/tracing/span_data.py#L11
    """

    AGENT = "agent"
    FUNCTION = "function"
    GENERATION = "generation"
    RESPONSE = "response"
    HANDOFF = "handoff"
    CUSTOM = "custom"
    GUARDRAIL = "guardrail"


_SPAN_TYPE_MAP = {
    OpenAISpanType.AGENT: SpanType.AGENT,
    OpenAISpanType.FUNCTION: SpanType.TOOL,
    OpenAISpanType.GENERATION: SpanType.CHAT_MODEL,
    OpenAISpanType.RESPONSE: SpanType.CHAT_MODEL,
    OpenAISpanType.GUARDRAIL: SpanType.TOOL,
    # Default to chain type
}


def add_mlflow_trace_processor():
    processors = GLOBAL_TRACE_PROVIDER._multi_processor._processors

    if any(isinstance(p, MlflowOpenAgentTracingProcessor) for p in processors):
        return

    add_trace_processor(MlflowOpenAgentTracingProcessor())


def remove_mlflow_trace_processor():
    processors = GLOBAL_TRACE_PROVIDER._multi_processor._processors
    non_mlflow_processors = [
        p for p in processors if not isinstance(p, MlflowOpenAgentTracingProcessor)
    ]
    GLOBAL_TRACE_PROVIDER._multi_processor._processors = non_mlflow_processors


class MlflowOpenAgentTracingProcessor(oai.TracingProcessor):
    def __init__(
        self,
        project_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._span_id_to_mlflow_span: dict[str, LiveSpan] = {}
        self._project_name = project_name

        # Patch TraceCtxManager to handle exceptions from the agent properly
        # The original implementation does not propagate exception to the root span,
        # resulting in the trace to have status OK even if there is an exception.
        def _patched_exit(original, instance, exc_type, exc_val, exc_tb):
            try:
                if exc_val and instance.trace:
                    span = self._span_id_to_mlflow_span.get(instance.trace.trace_id)
                    span.add_event(SpanEvent.from_exception(exc_val))
                    span.set_status(SpanStatusCode.ERROR)
            except Exception:
                _logger.debug("Failed to handle exception in MLflow trace", exc_info=True)

            return original(instance, exc_type, exc_val, exc_tb)

        safe_patch(
            FLAVOR_NAME,
            TraceCtxManager,
            "__exit__",
            _patched_exit,
        )

    def on_trace_start(self, trace: oai.Trace) -> None:
        try:
            mlflow_span = start_span_no_context(
                name=trace.name,
                span_type=SpanType.AGENT,
                # TODO: Trace object doesn't contain input/output. Can we get it somehow?
                inputs="",
                attributes=trace.metadata,
            )
            # NB: Trace ID has different prefix as span ID so will not conflict
            self._span_id_to_mlflow_span[trace.trace_id] = mlflow_span

            if trace.group_id:
                # Group ID is used for grouping multiple agent executions together
                mlflow_span.set_tag("group_id", trace.group_id)

            original_exit = trace.__exit__

            # Patch __exit__ method to handle exception properly
            def _patched_exit(self, exc_type, exc_val, exc_tb):
                if exc_val:
                    mlflow_span.add_event(SpanEvent.from_exception(exc_val))
                    mlflow_span.set_status(SpanStatusCode.ERROR)

                original_exit(exc_type, exc_val, exc_tb)

            safe_patch(
                FLAVOR_NAME,
                trace.__class__,
                "__exit__",
                _patched_exit,
            )

        except Exception:
            _logger.debug("Failed to start MLflow trace", exc_info=True)

    def on_trace_end(self, trace: oai.Trace) -> None:
        try:
            mlflow_span = self._span_id_to_mlflow_span.pop(trace.trace_id, None)
            mlflow_span.end(status=mlflow_span.status, outputs="")
        except Exception:
            _logger.debug("Failed to end MLflow trace", exc_info=True)

    def on_span_start(self, span: oai.Span[Any]) -> None:
        try:
            parent_mlflow_span = self._span_id_to_mlflow_span.get(span.parent_id)

            # Parent might be a trace
            if not parent_mlflow_span:
                parent_mlflow_span = self._span_id_to_mlflow_span.get(span.trace_id)

            inputs, _, attributes = _parse_span_data(span.span_data)
            span_type = _SPAN_TYPE_MAP.get(span.span_data.type, SpanType.CHAIN)

            mlflow_span = start_span_no_context(
                name=_get_span_name(span.span_data),
                span_type=span_type,
                parent_span=parent_mlflow_span,
                inputs=inputs,
                attributes=attributes,
            )

            if span_type == SpanType.CHAT_MODEL:
                mlflow_span.set_attribute(SpanAttributeKey.MESSAGE_FORMAT, "openai-agent")

            self._span_id_to_mlflow_span[span.span_id] = mlflow_span
        except Exception:
            _logger.debug("Failed to start MLflow span", exc_info=True)

    def on_span_end(self, span: oai.Span[Any]) -> None:
        try:
            # parsed_span_data = parse_spandata(span.span_data)
            mlflow_span = self._span_id_to_mlflow_span.pop(span.span_id, None)

            inputs, outputs, attributes = _parse_span_data(span.span_data)

            mlflow_span.set_inputs(inputs)
            mlflow_span.set_outputs(outputs)
            mlflow_span.set_attributes(attributes)

            if span.error:
                status = SpanStatus(
                    status_code=SpanStatusCode.ERROR,
                    description=span.error["message"],
                )
                mlflow_span.add_event(
                    SpanEvent(
                        name="exception",
                        attributes={
                            "exception.message": span.error["message"],
                            "exception.type": "",
                            "exception.stacktrace": json.dumps(span.error["data"]),
                        },
                    )
                )
            else:
                status = SpanStatusCode.OK

            mlflow_span.end(status=status)
        except Exception:
            _logger.debug("Failed to end MLflow span", exc_info=True)

    def force_flush(self) -> None:
        # MLflow doesn't need flush but this method is required by the interface
        pass

    def shutdown(self) -> None:
        self.force_flush()


def _get_span_name(span_data: oai.SpanData) -> str:
    if hasattr(span_data, "name"):
        return span_data.name
    elif isinstance(span_data, oai.GenerationSpanData):
        return "Generation"
    elif isinstance(span_data, oai.ResponseSpanData):
        return "Response"
    elif isinstance(span_data, oai.HandoffSpanData):
        return "Handoff"
    else:
        return "Unknown"


def _parse_span_data(span_data: oai.SpanData) -> tuple[Any, Any, dict[str, Any]]:
    inputs = None
    outputs = None
    attributes = {}

    if span_data.type == OpenAISpanType.AGENT:
        attributes = {
            "handoffs": span_data.handoffs,
            "tools": span_data.tools,
            "output_type": span_data.output_type,
        }
        outputs = {"output_type": span_data.output_type}

    elif span_data.type == OpenAISpanType.FUNCTION:
        try:
            inputs = json.loads(span_data.input)
        except Exception:
            inputs = span_data.input
        outputs = span_data.output

    elif span_data.type == OpenAISpanType.GENERATION:
        inputs = span_data.input
        outputs = span_data.output
        attributes = {
            "model": span_data.model,
            "model_config": span_data.model_config,
            "usage": span_data.usage,
        }

    elif span_data.type == OpenAISpanType.RESPONSE:
        inputs, outputs, attributes = _parse_response_span_data(span_data)

    elif span_data.type == OpenAISpanType.HANDOFF:
        inputs = {"from_agent": span_data.from_agent}
        outputs = {"to_agent": span_data.to_agent}

    elif span_data.type == OpenAISpanType.CUSTOM:
        outputs = span_data.data

    elif span_data.type == OpenAISpanType.GUARDRAIL:
        outputs = {"triggered": span_data.triggered}

    return inputs, outputs, attributes


def _parse_response_span_data(span_data: oai.ResponseSpanData) -> tuple[Any, Any, dict[str, Any]]:
    inputs = span_data.input
    response = span_data.response
    response_dict = response.model_dump() if response else {}
    outputs = response_dict.get("output")
    attributes = {k: v for k, v in response_dict.items() if k != "output"}

    # Extract chat tools
    chat_tools = []
    for tool in response_dict.get("tools", []):
        try:
            tool = ChatTool(
                type="function",
                function=FunctionToolDefinition(
                    name=tool["name"],
                    description=tool.get("description"),
                    parameters=tool.get("parameters"),
                    strict=tool.get("strict"),
                ),
            )
            chat_tools.append(tool)
        except Exception as e:
            _logger.debug(f"Failed to parse chat tool: {tool}. Error: {e}")

    if chat_tools:
        attributes[SpanAttributeKey.CHAT_TOOLS] = chat_tools

    return inputs, outputs, attributes
