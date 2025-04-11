import json
import logging
from typing import Any

import mlflow
import mlflow.anthropic
from mlflow.anthropic.chat import convert_message_to_mlflow_chat, convert_tool_to_mlflow_chat_tool
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.models.model import _MODEL_TRACKER
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.utils import (
    construct_full_inputs,
    end_client_span_or_trace,
    set_span_chat_messages,
    set_span_chat_tools,
    start_client_span_or_trace,
)
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)


def patched_class_call(original, self, *args, **kwargs):
    with TracingSession(original, self, args, kwargs) as manager:
        output = original(self, *args, **kwargs)
        manager.output = output
        return output


async def async_patched_class_call(original, self, *args, **kwargs):
    async with TracingSession(original, self, args, kwargs) as manager:
        output = await original(self, *args, **kwargs)
        manager.output = output
        return output


class TracingSession:
    """Context manager for handling MLflow spans in both sync and async contexts."""

    def __init__(self, original, instance, args, kwargs):
        self.mlflow_client = mlflow.MlflowClient()
        self.original = original
        self.instance = instance
        self.inputs = construct_full_inputs(original, instance, *args, **kwargs)

        # These attributes are set outside the constructor.
        self.span = None
        self.output = None

    def __enter__(self):
        return self._enter_impl()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_impl(exc_type, exc_val, exc_tb)

    async def __aenter__(self):
        return self._enter_impl()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._exit_impl(exc_type, exc_val, exc_tb)

    def _enter_impl(self):
        config = AutoLoggingConfig.init(flavor_name=mlflow.anthropic.FLAVOR_NAME)

        if config.log_traces:
            attributes = {}
            model_dict = _stringify_model_dict(self.inputs)
            model_identity = _generate_model_identity(model_dict)
            model_id = _MODEL_TRACKER.get(model_identity)
            if model_id is None and config.log_models:
                params = _generate_model_params_dict(model_dict)
                logged_model = mlflow.create_external_model(
                    name=mlflow.anthropic.FLAVOR_NAME, params=params
                )
                _MODEL_TRACKER.set(model_identity, logged_model.model_id)
                model_id = logged_model.model_id
            if model_id:
                attributes[SpanAttributeKey.MODEL_ID] = model_id
            self.span = start_client_span_or_trace(
                self.mlflow_client,
                name=f"{self.instance.__class__.__name__}.{self.original.__name__}",
                span_type=_get_span_type(self.original.__name__),
                inputs=self.inputs,
                attributes=attributes,
            )
            _set_tool_attribute(self.span, self.inputs)

        return self

    def _exit_impl(self, exc_type, exc_val, exc_tb) -> None:
        if self.span:
            if exc_val:
                self.span.add_event(SpanEvent.from_exception(exc_val))
                status = SpanStatusCode.ERROR
            else:
                status = SpanStatusCode.OK

            _set_chat_message_attribute(self.span, self.inputs, self.output)

            end_client_span_or_trace(
                self.mlflow_client,
                self.span,
                status=status,
                outputs=self.output,
            )


def _get_span_type(task_name: str) -> str:
    # Anthropic has a few APIs in beta, e.g., count_tokens.
    # Once they are stable, we can add them to the mapping.
    span_type_mapping = {
        "create": SpanType.CHAT_MODEL,
    }
    return span_type_mapping.get(task_name, SpanType.UNKNOWN)


def _set_tool_attribute(span: LiveSpan, inputs: dict[str, Any]):
    if (tools := inputs.get("tools")) is not None:
        try:
            tools = [convert_tool_to_mlflow_chat_tool(tool) for tool in tools]
            set_span_chat_tools(span, tools)
        except Exception as e:
            _logger.debug(f"Failed to set tools for {span}. Error: {e}")


def _set_chat_message_attribute(span: LiveSpan, inputs: dict[str, Any], output: Any):
    try:
        messages = [convert_message_to_mlflow_chat(msg) for msg in inputs.get("messages", [])]
        if output is not None:
            messages.append(convert_message_to_mlflow_chat(output))
        set_span_chat_messages(span, messages)
    except Exception as e:
        _logger.debug(f"Failed to set chat messages for {span}. Error: {e}")


def _stringify_model_dict(model_dict: dict[str, Any]) -> dict[str, str]:
    return {
        k: (v if isinstance(v, str) else json.dumps(v, default=str)) for k, v in model_dict.items()
    }


def _generate_model_params_dict(model_dict: dict[str, str]) -> dict[str, str]:
    # drop input fields
    exclude_fields = {"messages"}
    return {k: v for k, v in model_dict.items() if k not in exclude_fields}


def _generate_model_identity(model_dict: dict[str, str]) -> int:
    if "model" not in model_dict:
        raise ValueError("The model dictionary must contain 'model' key.")
    # drop input and non-model config fields to ensure consistent hashing
    exclude_fields = {
        # input
        "messages",
        # request metadata including user id
        "metadata",
        # extra API configs
        "extra_headers",
        "extra_query",
        "extra_body",
        "timeout",
    }
    model_dict = {k: v for k, v in model_dict.items() if k not in exclude_fields}
    return hash(str(model_dict))
