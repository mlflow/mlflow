import inspect
import json
import logging
from functools import singledispatchmethod
from typing import Any, Generator, Optional, Union

import llama_index.core
import pydantic
from llama_index.core.base.agent.types import BaseAgent, BaseAgentWorker, TaskStepOutput
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatResponse, CompletionResponse
from llama_index.core.base.response.schema import AsyncStreamingResponse, StreamingResponse
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.agent import AgentToolCallEvent
from llama_index.core.instrumentation.events.embedding import EmbeddingStartEvent
from llama_index.core.instrumentation.events.exception import ExceptionEvent
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
    LLMPredictStartEvent,
)
from llama_index.core.instrumentation.events.rerank import ReRankStartEvent
from llama_index.core.instrumentation.span.base import BaseSpan
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import BaseTool
from packaging.version import Version

import mlflow
from mlflow.entities import LiveSpan, SpanEvent, SpanType
from mlflow.entities.document import Document
from mlflow.entities.span_status import SpanStatusCode
from mlflow.llama_index.chat import get_chat_messages_from_event
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.provider import detach_span_from_context, set_span_in_context
from mlflow.tracing.utils import set_span_chat_messages, set_span_chat_tools
from mlflow.utils.pydantic_utils import model_dump_compat

_logger = logging.getLogger(__name__)


def _get_llama_index_version() -> Version:
    return Version(llama_index.core.__version__)


def set_llama_index_tracer():
    """
    Set the MlflowSpanHandler and MlflowEventHandler to the global dispatcher.
    If the handlers are already set, skip setting.
    """
    from llama_index.core.instrumentation import get_dispatcher

    dsp = get_dispatcher()

    span_handler = None
    for handler in dsp.span_handlers:
        if isinstance(handler, MlflowSpanHandler):
            _logger.debug("MlflowSpanHandler is already set to the dispatcher. Skip setting.")
            span_handler = handler
            break
    else:
        span_handler = MlflowSpanHandler()
        dsp.add_span_handler(span_handler)

    for handler in dsp.event_handlers:
        if isinstance(handler, MlflowEventHandler):
            _logger.debug("MlflowEventHandler is already set to the dispatcher. Skip setting.")
            break
    else:
        dsp.add_event_handler(MlflowEventHandler(span_handler))


def remove_llama_index_tracer():
    """
    Remove the MlflowSpanHandler and MlflowEventHandler from the global dispatcher.
    """
    from llama_index.core.instrumentation import get_dispatcher

    dsp = get_dispatcher()
    dsp.span_handlers = [h for h in dsp.span_handlers if h.class_name() != "MlflowSpanHandler"]
    dsp.event_handlers = [h for h in dsp.event_handlers if h.class_name() != "MlflowEventHandler"]


class _LlamaSpan(BaseSpan, extra="allow"):
    _mlflow_span: LiveSpan = pydantic.PrivateAttr()

    def __init__(self, id_: str, parent_id: Optional[str], mlflow_span: LiveSpan):
        super().__init__(id_=id_, parent_id=parent_id)
        self._mlflow_span = mlflow_span


def _end_span(span: LiveSpan, status=SpanStatusCode.OK, outputs=None, token=None):
    """An utility function to end the span or trace."""
    if isinstance(outputs, (StreamingResponse, AsyncStreamingResponse, StreamingAgentChatResponse)):
        _logger.warning(
            "Trying to record streaming response to the MLflow trace. This may consume "
            "the generator and result in an empty response."
        )

    # for retriever spans, convert the outputs to Document objects
    # so they can be rendered in a more user-friendly way in the UI
    if (
        span.span_type == SpanType.RETRIEVER
        and isinstance(outputs, list)
        and all(isinstance(item, NodeWithScore) for item in outputs)
    ):
        try:
            outputs = [Document.from_llama_index_node_with_score(node) for node in outputs]
        except Exception as e:
            _logger.debug(
                f"Failed to convert NodeWithScore to Document objects: {e}", exc_info=True
            )

    if outputs is None:
        outputs = span.outputs

    try:
        span.end(status=status, outputs=outputs)
    finally:
        # We should detach span even when end_span / end_trace API call fails
        if token:
            detach_span_from_context(token)


class MlflowSpanHandler(BaseSpanHandler[_LlamaSpan], extra="allow"):
    def __init__(self):
        super().__init__()
        self._span_id_to_token = {}
        self._stream_resolver = StreamResolver()
        self._pending_spans: dict[str, _LlamaSpan] = {}

    @classmethod
    def class_name(cls) -> str:
        return "MlflowSpanHandler"

    def get_span_for_event(self, event: BaseEvent) -> LiveSpan:
        llama_span = self.open_spans.get(event.span_id) or self._pending_spans.get(event.span_id)
        return llama_span._mlflow_span if llama_span else None

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        **kwargs: Any,
    ) -> _LlamaSpan:
        with self.lock:
            parent = self.open_spans.get(parent_span_id) if parent_span_id else None

        parent_span = parent._mlflow_span if parent else mlflow.get_current_active_span()

        try:
            input_args = bound_args.arguments
            attributes = self._get_instance_attributes(instance)
            span_type = self._get_span_type(instance) or SpanType.UNKNOWN
            span = start_span_no_context(
                name=id_.partition("-")[0],
                parent_span=parent_span,
                span_type=span_type,
                inputs=input_args,
                attributes=attributes,
            )

            token = set_span_in_context(span)
            self._span_id_to_token[span.span_id] = token

            # NB: The tool definition is passed to LLM via kwargs, but it is not set
            # to the LLM/Chat start event. Therefore, we need to handle it here.
            tools = input_args.get("kwargs", {}).get("tools")
            if tools and span_type in [SpanType.LLM, SpanType.CHAT_MODEL]:
                try:
                    set_span_chat_tools(span, tools)
                except Exception as e:
                    _logger.debug(f"Failed to set tools for {span}: {e}")

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
            with self.lock:
                llama_span = self.open_spans.get(id_)
            if not llama_span:
                return

            span = llama_span._mlflow_span
            token = self._span_id_to_token.pop(span.span_id, None)

            if self._stream_resolver.is_streaming_result(result):
                # If the result is a generator, we keep the span in progress for streaming
                # and end it when the generator is exhausted.
                is_pended = self._stream_resolver.register_stream_span(span, result)
                if is_pended:
                    self._pending_spans[id_] = llama_span
                    # We still need to detach the span from the context, otherwise it will
                    # be considered as "active"
                    detach_span_from_context(token)
                else:
                    # If the span is not pended successfully, end it immediately
                    _end_span(span=span, outputs=result, token=token)
            else:
                _end_span(span=span, outputs=result, token=token)
            return llama_span
        except BaseException as e:
            _logger.debug(f"Failed to end a span: {e}", exc_info=True)

    def resolve_pending_stream_span(self, span: LiveSpan, event: Any):
        """End the pending streaming span(s)"""
        self._stream_resolver.resolve(span, event)
        self._pending_spans.pop(event.span_id, None)

    def prepare_to_drop_span(self, id_: str, err: Optional[Exception], **kwargs) -> _LlamaSpan:
        """Logic for handling errors during the model execution."""
        with self.lock:
            llama_span = self.open_spans.get(id_)
        span = llama_span._mlflow_span
        token = self._span_id_to_token.pop(span.span_id, None)

        if _get_llama_index_version() >= Version("0.10.59"):
            # LlamaIndex determines if a workflow is terminated or not by propagating an special
            # exception WorkflowDone. We should treat this exception as a successful termination.
            from llama_index.core.workflow.errors import WorkflowDone

            if err and isinstance(err, WorkflowDone):
                return _end_span(span=span, status=SpanStatusCode.OK, token=token)

        span.add_event(SpanEvent.from_exception(err))
        _end_span(span=span, status="ERROR", token=token)
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
    def _get_instance_attributes(self, instance: Any) -> dict[str, Any]:
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

    def _get_llm_attributes(self, instance) -> dict[str, Any]:
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
        try:
            if span := self._span_handler.get_span_for_event(event):
                self._handle_event(event, span)
        except Exception as e:
            _logger.debug(f"Failed to handle event: {e}", exc_info=True)

    @singledispatchmethod
    def _handle_event(self, event: BaseEvent, span: LiveSpan):
        # Pass through the events we are not interested in
        pass

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
        self._extract_and_set_chat_messages(span, event)

    @_handle_event.register
    def _(self, event: LLMCompletionEndEvent, span: LiveSpan):
        span.set_attribute("usage", self._extract_token_usage(event.response))
        self._extract_and_set_chat_messages(span, event)
        self._span_handler.resolve_pending_stream_span(span, event)

    @_handle_event.register
    def _(self, event: LLMChatStartEvent, span: LiveSpan):
        span.set_attribute(SpanAttributeKey.SPAN_TYPE, SpanType.CHAT_MODEL)
        span.set_attribute("model_dict", event.model_dict)
        self._extract_and_set_chat_messages(span, event)

    @_handle_event.register
    def _(self, event: LLMChatEndEvent, span: LiveSpan):
        span.set_attribute("usage", self._extract_token_usage(event.response))
        self._extract_and_set_chat_messages(span, event)
        self._span_handler.resolve_pending_stream_span(span, event)

    @_handle_event.register
    def _(self, event: ReRankStartEvent, span: LiveSpan):
        span.set_attribute(SpanAttributeKey.SPAN_TYPE, SpanType.RERANKER)
        span.set_attributes(
            {
                "model_name": event.model_name,
                "top_n": event.top_n,
            }
        )

    @_handle_event.register
    def _(self, event: ExceptionEvent, span: LiveSpan):
        """
        Handle an exception event for stream spans.

        For non-stream spans, exception is processed by the prepare_to_drop_span() handler of
        the span handler. However, for stream spans, the exception may raised during the
        streaming after it exit. Therefore, we need to resolve the span here.
        """
        self._span_handler.resolve_pending_stream_span(span, event)

    def _extract_token_usage(
        self, response: Union[ChatResponse, CompletionResponse]
    ) -> dict[str, int]:
        if raw := response.raw:
            # The raw response can be a Pydantic model or a dictionary
            if isinstance(raw, pydantic.BaseModel):
                raw = model_dump_compat(raw)

            if usage := raw.get("usage"):
                return usage

        # If the usage is not found in the raw response, look for token counts
        # in additional_kwargs of the completion payload
        usage = {}
        if additional_kwargs := getattr(response, "additional_kwargs", None):
            for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                if (v := additional_kwargs.get(k)) is not None:
                    usage[k] = v
        return usage

    def _extract_and_set_chat_messages(self, span: LiveSpan, event: BaseEvent):
        try:
            messages = get_chat_messages_from_event(event)
            set_span_chat_messages(span, messages, append=True)
        except Exception as e:
            _logger.debug(f"Failed to set chat messages to the span: {e}", exc_info=True)


_StreamEndEvent = Union[LLMChatEndEvent, LLMCompletionEndEvent, ExceptionEvent]


class StreamResolver:
    """
    A class is responsible for closing the pending streaming spans that are waiting
    for the stream to be exhausted. Once the associated stream is exhausted, this
    class will resolve the span, as well as recursively resolve the parent spans
    that returns the same (or derived) stream.
    """

    def __init__(self):
        self._span_id_to_span_and_gen: dict[str, tuple[LiveSpan, Generator]] = {}

    def is_streaming_result(self, result: Any) -> bool:
        return (
            inspect.isgenerator(result)  # noqa: SIM101
            or isinstance(result, (StreamingResponse, AsyncStreamingResponse))
            or isinstance(result, StreamingAgentChatResponse)
            or (isinstance(result, TaskStepOutput) and self.is_streaming_result(result.output))
        )

    def register_stream_span(self, span: LiveSpan, result: Any) -> bool:
        """
        Register the pending streaming span with the associated generator.

        Args:
            span: The span that has a streaming output.
            result: The streaming result that is being processed.

        Returns:
            True if the span is registered successfully, False otherwise.
        """
        if inspect.isgenerator(result):
            stream = result
        elif isinstance(result, (StreamingResponse, AsyncStreamingResponse)):
            stream = result.response_gen
        elif isinstance(result, StreamingAgentChatResponse):
            stream = result.chat_stream
        elif isinstance(result, TaskStepOutput):
            stream = result.output.chat_stream
        else:
            raise ValueError(f"Unsupported streaming response type: {type(result)}")

        if inspect.getgeneratorstate(stream) == inspect.GEN_CLOSED:
            # Not registering the span because the generator is already exhausted.
            # It's counter-intuitive that the generator is closed before the response
            # is returned, but it can happen because some agents run streaming request
            # in a separate thread. In this case, the generator can be closed before
            # the response is returned in the main thread.
            return False

        self._span_id_to_span_and_gen[span.span_id] = (span, stream)
        return True

    def resolve(self, span: LiveSpan, event: _StreamEndEvent):
        """
        Finish the streaming span and recursively resolve the parent spans that
        returns the same (or derived) stream.
        """
        _, stream = self._span_id_to_span_and_gen.pop(span.span_id, (None, None))
        if not stream:
            return

        if isinstance(event, (LLMChatEndEvent, LLMCompletionEndEvent)):
            outputs = event.response
            status = SpanStatusCode.OK
        elif isinstance(event, ExceptionEvent):
            outputs = None
            status = SpanStatusCode.ERROR
            span.add_event(SpanEvent.from_exception(event.exception))
        else:
            raise ValueError(f"Unsupported event type to resolve streaming: {type(event)}")

        _end_span(span=span, status=status, outputs=outputs)

        # Extract the complete text from the event.
        if isinstance(outputs, ChatResponse):
            output_text = outputs.message.content
        elif isinstance(outputs, CompletionResponse):
            output_text = outputs.response.text
        else:
            output_text = None

        # Recursively resolve the parent spans that are also waiting for the same token
        # stream to be exhausted.
        while span.parent_id in self._span_id_to_span_and_gen:
            if span_and_stream := self._span_id_to_span_and_gen.pop(span.parent_id, None):
                span, stream = span_and_stream
                # We reuse the same output text for parent spans. This may not be 100% correct
                # as token stream can be modified by callers. However, it is technically
                # challenging to track the modified stream across multiple spans.
                _end_span(span=span, status=status, outputs=output_text)
