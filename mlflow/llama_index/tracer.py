import inspect
import json
import logging
from functools import singledispatchmethod
from typing import Any, Dict, Optional

from llama_index.core.base.agent.types import BaseAgent, BaseAgentWorker
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.agent import AgentToolCallEvent
from llama_index.core.instrumentation.events.embedding import EmbeddingStartEvent
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatInProgressEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionInProgressEvent,
    LLMCompletionStartEvent,
    LLMPredictStartEvent,
)
from llama_index.core.instrumentation.events.rerank import ReRankStartEvent
from llama_index.core.instrumentation.span.base import BaseSpan
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.core.tools import BaseTool
from pydantic import PrivateAttr

import mlflow
from mlflow.entities import LiveSpan, SpanEvent, SpanType
from mlflow.tracing.constant import SpanAttributeKey

_logger = logging.getLogger(__name__)


class _LlamaSpan(BaseSpan, extra="allow"):
    _mlflow_span: LiveSpan = PrivateAttr()

    def __init__(self, id_: str, parent_id: Optional[str], mlflow_span: LiveSpan):
        super().__init__(id_=id_, parent_id=parent_id)
        self._mlflow_span = mlflow_span


class MlflowSpanHandler(BaseSpanHandler[_LlamaSpan], extra="allow"):
    _mlflow_client: mlflow.MlflowClient = PrivateAttr()

    def __init__(self, client: mlflow.MlflowClient):
        super().__init__()
        self._mlflow_client = client

    @classmethod
    def class_name(cls) -> str:
        return "MlflowSpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
    ) -> _LlamaSpan:
        try:
            input_args = bound_args.arguments
            attributes = self._get_instance_attributes(instance)
            span_type = self._get_span_type(instance) or SpanType.UNKNOWN
            if parent_span_id and (parent := self.open_spans.get(parent_span_id)):
                parent_span = parent._mlflow_span
                span = self._mlflow_client.start_span(
                    request_id=parent_span.request_id,
                    parent_id=parent_span.span_id,
                    name=id_.partition("-")[0],
                    span_type=span_type,
                    inputs=input_args,
                    attributes=attributes,
                )
            else:
                span = self._mlflow_client.start_trace(
                    name=id_.partition("-")[0],
                    span_type=span_type,
                    inputs=input_args,
                    attributes=attributes,
                )
            return _LlamaSpan(id_=id_, parent_id=parent_span_id, mlflow_span=span)
        except BaseException as e:
            _logger.debug(f"Failed to create a new span: {e}", exc_info=True)

    def prepare_to_exit_span(
        self,
        id_: str,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> _LlamaSpan:
        try:
            llama_span = self.open_spans.get(id_)
            span = llama_span._mlflow_span
            if span.parent_id is None:
                self._mlflow_client.end_trace(span.request_id, outputs=result)
            else:
                self._mlflow_client.end_span(span.request_id, span.span_id, outputs=result)
            return llama_span
        except BaseException as e:
            _logger.debug(f"Failed to end a span: {e}", exc_info=True)

    def prepare_to_drop_span(self, id_: str, err: Optional[Exception], **kwargs) -> _LlamaSpan:
        """Logic for handling errors during the model execution."""
        llama_span = self.open_spans.get(id_)
        span = llama_span._mlflow_span
        span.add_event(SpanEvent.from_exception(err))

        if span.parent_id is None:
            self._mlflow_client.end_trace(span.request_id, status="ERROR")
        else:
            self._mlflow_client.end_span(span.request_id, span.span_id, status="ERROR")
        return llama_span

    def _get_span_type(self, instance: Any) -> SpanType:
        """
        Map LlamaIndex instance type to MLflow span type. Some span type cannot be determined
        by instance type alone, rather need event info e.g. ChatModel, ReRanker
        """
        if isinstance(instance, (BaseLLM, MultiModalLLM)):
            return SpanType.LLM
        elif isinstance(instance, BaseRetriever):
            return SpanType.RETRIEVER
        elif isinstance(instance, (BaseAgent, BaseAgentWorker)):
            return SpanType.AGENT
        elif isinstance(instance, BaseEmbedding):
            return SpanType.EMBEDDING
        elif isinstance(instance, BaseTool):
            return SpanType.TOOL
        else:
            return SpanType.CHAIN

    @singledispatchmethod
    def _get_instance_attributes(self, instance: Any) -> Dict[str, Any]:
        """
        Extract span attributes from LlamaIndex objects.

        NB: There are some overlap between attributes extracted from instance metadata and the
        events. For example, model name for an LLM is available in both. However, events might
        not always be triggered (e.g. 3P llm integration doesn't implement the event logic),
        so the instance metadata serves as a fallback source of information.
        """

    # TODO: Union type hint doesn't work with singledispatchmethod, so we have to define
    #  two separate methods for BaseLLM and MultiModalLLM. Once we upgrade to Python 3.10,
    #  we can use `BaseLLM | MultiModelLLM` type hint and it works with singledispatchmethod.
    @_get_instance_attributes.register
    def _(self, instance: BaseLLM):
        return self._get_llm_attributes(instance)

    @_get_instance_attributes.register
    def _(self, instance: MultiModalLLM):
        return self._get_llm_attributes(instance)

    def _get_llm_attributes(self, instance) -> Dict[str, Any]:
        attr = {}
        if metadata := instance.metadata:
            attr["model_name"] = metadata.model_name
            if params_str := metadata.json(exclude_unset=True):
                attr["invocation_params"] = json.loads(params_str)
        return attr

    @_get_instance_attributes.register
    def _(self, instance: BaseEmbedding):
        return {
            "model_name": instance.model_name,
            "embed_batch_size": instance.embed_batch_size,
        }

    @_get_instance_attributes.register
    def _(self, instance: BaseTool):
        metadata = instance.metadata
        attributes = {"description": metadata.description}
        try:
            attributes["name"] = metadata.name
        except ValueError:
            # ToolMetadata.get_name() raises ValueError if name is None
            pass
        try:
            attributes["parameters"] = json.loads(metadata.fn_schema_str)
        except ValueError:
            # ToolMetadata.get_fn_schema_str() raises ValueError if fn_schema is None
            pass
        return attributes


class MlflowEventHandler(BaseEventHandler, extra="allow"):
    """
    Event handler processes various events that are triggered during execution.

    Events are used as supplemental source for recording additional metadata to the span,
    such as model name, parameters to the span, because they are not available in the inputs
    and outputs in SpanHandler.
    """

    _span_handler: MlflowSpanHandler

    @classmethod
    def class_name(cls) -> str:
        return "MlflowEventHandler"

    def __init__(self, _span_handler):
        super().__init__()
        self._span_handler = _span_handler

    def handle(self, event: BaseEvent) -> Any:
        # TODO: These two InProgress events are triggered per streaming chunk. We
        # ignore them for now but they should be accumulated as SpanEvents of the parent span.
        if isinstance(event, (LLMChatInProgressEvent, LLMCompletionInProgressEvent)):
            return

        if span := self._span_handler.open_spans.get(event.span_id):
            self._handle_event(event, span._mlflow_span)

    @singledispatchmethod
    def _handle_event(self, event: BaseEvent, span: LiveSpan):
        ...

    @_handle_event.register
    def _(self, event: AgentToolCallEvent, span: LiveSpan):
        span.set_attribute("name", event.tool.name)
        span.set_attribute("description", event.tool.description)
        span.set_attribute("parameters", event.tool.get_parameters_dict())

    @_handle_event.register
    def _(self, event: EmbeddingStartEvent, span: LiveSpan):
        span.set_attribute("model_dict", event.model_dict)

    @_handle_event.register
    def _(self, event: LLMPredictStartEvent, span: LiveSpan):
        """
        An event triggered when LLM's predict() is called.

        In LlamaIndex, predict() is a gateway method that dispatch the request to
        either chat() or completion() method depending on the model type, as well
        as crafting prompt from the template.
        """
        template = event.template
        template_args = {
            **template.kwargs,
            **(event.template_args if event.template_args else {}),
        }
        span.set_attributes(
            {
                "prmopt_template": template.get_template(),
                "template_arguments": {var: template_args.get(var) for var in template_args},
            }
        )

    @_handle_event.register
    def _(self, event: LLMCompletionStartEvent, span: LiveSpan):
        span.set_attribute("prompt", event.prompt)
        span.set_attribute("model_dict", event.model_dict)

    @_handle_event.register
    def _(self, event: LLMCompletionEndEvent, span: LiveSpan):
        span.set_attribute("usage", self._extract_token_counts(event.response))

    @_handle_event.register
    def _(self, event: LLMChatStartEvent, span: LiveSpan):
        span.set_attribute(SpanAttributeKey.SPAN_TYPE, SpanType.CHAT_MODEL)
        span.set_attribute("model_dict", event.model_dict)

    @_handle_event.register
    def _(self, event: LLMChatEndEvent, span: LiveSpan):
        span.set_attribute("usage", self._extract_token_counts(event.response))

    @_handle_event.register
    def _(self, event: ReRankStartEvent, span: LiveSpan):
        span.set_attribute(SpanAttributeKey.SPAN_TYPE, SpanType.RERANKER)
        span.set_attributes(
            {
                "model_name": event.model_name,
                "top_n": event.top_n,
            }
        )

    def _extract_token_counts(self, response) -> Dict[str, int]:
        if (
            (raw := getattr(response, "raw", None))
            and hasattr(raw, "get")
            and (usage := raw.get("usage"))
        ):
            return usage
        else:
            usage = {}
            # Look for token counts in additional_kwargs of the completion payload
            if additional_kwargs := getattr(response, "additional_kwargs", None):
                for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                    if (v := additional_kwargs.get(k)) is not None:
                        usage[k] = v
            return usage
