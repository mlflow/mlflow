import logging
from typing import Any, Optional, Union

import dspy
from dspy.utils.callback import BaseCallback

import mlflow
from mlflow.entities import SpanStatusCode, SpanType
from mlflow.entities.span_event import SpanEvent
from mlflow.exceptions import MlflowException
from mlflow.pyfunc.context import get_prediction_context, maybe_set_prediction_context
from mlflow.tracing.provider import detach_span_from_context, set_span_in_context
from mlflow.tracing.utils import (
    end_client_span_or_trace,
    set_span_chat_messages,
    start_client_span_or_trace,
)
from mlflow.tracing.utils.token import SpanWithToken

_logger = logging.getLogger(__name__)


class MlflowCallback(BaseCallback):
    """Callback for generating MLflow traces for DSPy components"""

    def __init__(self, dependencies_schema: Optional[dict[str, Any]] = None):
        self._client = mlflow.MlflowClient()
        self._dependencies_schema = dependencies_schema
        # call_id: (LiveSpan, OTel token)
        self._call_id_to_span: dict[str, SpanWithToken] = {}

    def set_dependencies_schema(self, dependencies_schema: dict[str, Any]):
        if self._dependencies_schema:
            raise MlflowException(
                "Dependencies schema should be set only once to the callback.",
                error_code=MlflowException.INVALID_PARAMETER_VALUE,
            )
        self._dependencies_schema = dependencies_schema

    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        span_type = self._get_span_type_for_module(instance)
        attributes = self._get_span_attribute_for_module(instance)

        # The __call__ method of dspy.Module has a signature of (self, *args, **kwargs),
        # while all built-in modules only accepts keyword arguments. To avoid recording
        # empty "args" key in the inputs, we remove it if it's empty.
        if "args" in inputs and not inputs["args"]:
            inputs.pop("args")

        self._start_span(
            call_id,
            name=f"{instance.__class__.__name__}.forward",
            span_type=span_type,
            inputs=self._unpack_kwargs(inputs),
            attributes=attributes,
        )

    def on_module_end(
        self, call_id: str, outputs: Optional[Any], exception: Optional[Exception] = None
    ):
        # NB: DSPy's Prediction object is a customized dictionary-like object, but its repr
        # is not easy to read on UI. Therefore, we unpack it to a dictionary.
        # https://github.com/stanfordnlp/dspy/blob/6fe693528323c9c10c82d90cb26711a985e18b29/dspy/primitives/prediction.py#L21-L28
        if isinstance(outputs, dspy.Prediction):
            outputs = outputs.toDict()

        self._end_span(call_id, outputs, exception)

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        span_type = (
            SpanType.CHAT_MODEL if getattr(instance, "model_type", None) == "chat" else SpanType.LLM
        )

        attributes = {
            **instance.kwargs,
            "model": instance.model,
            "model_type": instance.model_type,
            "cache": instance.cache,
        }

        inputs = self._unpack_kwargs(inputs)

        span = self._start_span(
            call_id,
            name=f"{instance.__class__.__name__}.__call__",
            span_type=span_type,
            inputs=inputs,
            attributes=attributes,
        )

        if messages := self._extract_messages_from_lm_inputs(inputs):
            try:
                set_span_chat_messages(span, messages)
            except Exception as e:
                _logger.debug(f"Failed to set input messages for {span}. Error: {e}")

    def on_lm_end(
        self, call_id: str, outputs: Optional[Any], exception: Optional[Exception] = None
    ):
        st = self._call_id_to_span.get(call_id)
        try:
            output_msg = self._extract_messages_from_lm_outputs(outputs)
            set_span_chat_messages(st.span, output_msg, append=True)
        except Exception as e:
            _logger.debug(f"Failed to set output messages for {call_id}. Error: {e}")

        self._end_span(call_id, outputs, exception)

    def _extract_messages_from_lm_inputs(self, inputs: dict[str, Any]) -> list[dict[str, str]]:
        # LM input is either a list of messages or a prompt string
        # https://github.com/stanfordnlp/dspy/blob/ac5bf56bb1ed7261d9637168563328c1dfeb27af/dspy/clients/lm.py#L92
        # TODO: Extract tool definition once https://github.com/stanfordnlp/dspy/pull/2023 is merged
        return inputs.get("messages") or [{"role": "user", "content": inputs.get("prompt")}]

    def _extract_messages_from_lm_outputs(
        self, outputs: list[Union[str, dict[str, Any]]]
    ) -> list[dict[str, str]]:
        # LM output is either a string or a dictionary of text and logprobs
        # https://github.com/stanfordnlp/dspy/blob/ac5bf56bb1ed7261d9637168563328c1dfeb27af/dspy/clients/lm.py#L105-L114
        # TODO: Extract tool calls once https://github.com/stanfordnlp/dspy/pull/2023 is merged
        return [
            {"role": "assistant", "content": o.get("text") if isinstance(o, dict) else o}
            for o in outputs
        ]

    def on_adapter_format_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        self._start_span(
            call_id,
            name=f"{instance.__class__.__name__}.format",
            span_type=SpanType.PARSER,
            inputs=self._unpack_kwargs(inputs),
            attributes={},
        )

    def on_adapter_format_end(
        self, call_id: str, outputs: Optional[Any], exception: Optional[Exception] = None
    ):
        self._end_span(call_id, outputs, exception)

    def on_adapter_parse_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        self._start_span(
            call_id,
            name=f"{instance.__class__.__name__}.parse",
            span_type=SpanType.PARSER,
            inputs=self._unpack_kwargs(inputs),
            attributes={},
        )

    def on_adapter_parse_end(
        self, call_id: str, outputs: Optional[Any], exception: Optional[Exception] = None
    ):
        self._end_span(call_id, outputs, exception)

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        # DSPy uses the special "finish" tool to signal the end of the agent.
        if instance.name == "finish":
            return

        inputs = self._unpack_kwargs(inputs)
        # Tools are always called with keyword arguments only.
        inputs.pop("args", None)

        self._start_span(
            call_id,
            name=f"Tool.{instance.name}",
            span_type=SpanType.TOOL,
            inputs=inputs,
            attributes={
                "name": instance.name,
                "description": instance.desc,
                "args": instance.args,
            },
        )

    def on_tool_end(
        self, call_id: str, outputs: Optional[Any], exception: Optional[Exception] = None
    ):
        if call_id in self._call_id_to_span:
            self._end_span(call_id, outputs, exception)

    def _start_span(
        self,
        call_id: str,
        name: str,
        span_type: SpanType,
        inputs: dict[str, Any],
        attributes: dict[str, Any],
    ):
        prediction_context = get_prediction_context()
        if prediction_context and self._dependencies_schema:
            prediction_context.update(**self._dependencies_schema)

        with maybe_set_prediction_context(prediction_context):
            span = start_client_span_or_trace(
                self._client,
                name=name,
                span_type=span_type,
                parent_span=mlflow.get_current_active_span(),
                inputs=inputs,
                attributes=attributes,
            )

        token = set_span_in_context(span)
        self._call_id_to_span[call_id] = SpanWithToken(span, token)

        return span

    def _end_span(
        self,
        call_id: str,
        outputs: Optional[Any],
        exception: Optional[Exception] = None,
    ):
        st = self._call_id_to_span.pop(call_id, None)

        if not st.span:
            _logger.warning(f"Failed to end a span. Span not found for call_id: {call_id}")
            return

        status = SpanStatusCode.OK if exception is None else SpanStatusCode.ERROR

        if exception:
            st.span.add_event(SpanEvent.from_exception(exception))

        try:
            end_client_span_or_trace(
                client=self._client,
                span=st.span,
                outputs=outputs,
                status=status,
            )
        finally:
            detach_span_from_context(st.token)

    def _get_span_type_for_module(self, instance):
        if isinstance(instance, dspy.Retrieve):
            return SpanType.RETRIEVER
        elif isinstance(instance, dspy.ReAct):
            return SpanType.AGENT
        elif isinstance(instance, dspy.Predict):
            return SpanType.LLM
        elif isinstance(instance, dspy.Adapter):
            return SpanType.PARSER
        else:
            return SpanType.CHAIN

    def _get_span_attribute_for_module(self, instance):
        if isinstance(instance, dspy.Predict):
            return {"signature": instance.signature.signature}
        elif isinstance(instance, dspy.ChainOfThought):
            if hasattr(instance, "signature"):
                signature = instance.signature.signature
            else:
                signature = instance.predict.signature.signature

            attributes = {"signature": signature}
            if hasattr(instance, "extended_signature"):
                attributes["extended_signature"] = instance.extended_signature.signature
            return attributes
        return {}

    def _unpack_kwargs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Unpacks the kwargs from the inputs dictionary"""
        # NB: Not using pop() to avoid modifying the original inputs dictionary
        kwargs = inputs.get("kwargs", {})
        inputs_wo_kwargs = {k: v for k, v in inputs.items() if k != "kwargs"}
        return {**inputs_wo_kwargs, **kwargs}
