import ast
import logging
from contextvars import ContextVar
from typing import Any, Optional, Sequence, Union
from uuid import UUID

import pydantic
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.load.dump import dumps
from langchain_core.messages import BaseMessage
from langchain_core.outputs import (
    ChatGenerationChunk,
    GenerationChunk,
    LLMResult,
)
from tenacity import RetryCallState

import mlflow
from mlflow.entities import Document as MlflowDocument
from mlflow.entities import LiveSpan, SpanEvent, SpanStatus, SpanStatusCode, SpanType
from mlflow.entities.span import NO_OP_SPAN_TRACE_ID
from mlflow.exceptions import MlflowException
from mlflow.langchain.utils.chat import (
    convert_lc_generation_to_chat_message,
    convert_lc_message_to_chat_message,
)
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.provider import detach_span_from_context, set_span_in_context
from mlflow.tracing.utils import (
    maybe_set_prediction_context,
    set_span_chat_messages,
    set_span_chat_tools,
)
from mlflow.tracing.utils.token import SpanWithToken
from mlflow.types.chat import ChatMessage, ChatTool, FunctionToolDefinition
from mlflow.utils import IS_PYDANTIC_V2_OR_NEWER
from mlflow.utils.autologging_utils import ExceptionSafeAbstractClass
from mlflow.version import IS_TRACING_SDK_ONLY

if not IS_TRACING_SDK_ONLY:
    from mlflow.pyfunc.context import Context


_logger = logging.getLogger(__name__)

_should_attach_span_to_context = ContextVar("should_attach_span_to_context", default=True)


class MlflowLangchainTracer(BaseCallbackHandler, metaclass=ExceptionSafeAbstractClass):
    """
    Callback for auto-logging traces.
    We need to inherit ExceptionSafeAbstractClass to avoid invalid new
    input arguments added to original function call.

    Args:
        prediction_context: Optional prediction context object to be set for the
            thread-local context. Occasionally this has to be passed manually because
            the callback may be invoked asynchronously and Langchain doesn't correctly
            propagate the thread-local context.
    """

    def __init__(
        self,
        prediction_context: Optional["Context"] = None,
    ):
        # NB: The tracer can handle multiple traces in parallel under multi-threading scenarios.
        # DO NOT use instance variables to manage the state of single trace.
        super().__init__()
        # run_id: (LiveSpan, OTel token)
        self._run_span_mapping: dict[str, SpanWithToken] = {}
        self._prediction_context = prediction_context

    def _get_span_by_run_id(self, run_id: UUID) -> Optional[LiveSpan]:
        if span_with_token := self._run_span_mapping.get(str(run_id), None):
            return span_with_token.span
        raise MlflowException(f"Span for run_id {run_id!s} not found.")

    def _serialize_invocation_params(
        self, attributes: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        """
        Serialize the 'invocation_params' in the attributes dictionary.
        If 'invocation_params' contains a key 'response_format' whose value is a subclass
        of pydantic.BaseModel, replace it with its JSON schema.
        """
        if not attributes:
            return attributes

        invocation_params = attributes.get("invocation_params")
        if not isinstance(invocation_params, dict):
            return attributes

        response_format = invocation_params.get("response_format")
        if isinstance(response_format, type) and issubclass(response_format, pydantic.BaseModel):
            try:
                invocation_params["response_format"] = (
                    response_format.model_json_schema()
                    if IS_PYDANTIC_V2_OR_NEWER
                    else response_format.schema()
                )
            except Exception as e:
                _logger.error(
                    "Failed to generate JSON schema for response_format: %s", e, exc_info=True
                )
        return attributes

    def _start_span(
        self,
        span_name: str,
        parent_run_id: Optional[UUID],
        span_type: str,
        run_id: UUID,
        inputs: Optional[Union[str, dict[str, Any]]] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> LiveSpan:
        """Start MLflow Span (or Trace if it is root component)"""
        serialized_attributes = self._serialize_invocation_params(attributes)
        dependencies_schemas = (
            self._prediction_context.dependencies_schemas if self._prediction_context else None
        )
        with maybe_set_prediction_context(
            self._prediction_context
        ):  # When parent_run_id is None, this is root component so start trace
            span = start_span_no_context(
                name=span_name,
                span_type=span_type,
                parent_span=self._get_parent_span(parent_run_id),
                inputs=inputs,
                attributes=serialized_attributes,
                tags=dependencies_schemas,
            )

            # Debugging purpose
            if span.trace_id == NO_OP_SPAN_TRACE_ID:
                _logger.debug("No Op span was created, the trace will not be recorded.")

        # Attach the span to the current context to mark it "active"
        token = set_span_in_context(span) if _should_attach_span_to_context.get() else None
        self._run_span_mapping[str(run_id)] = SpanWithToken(span, token)
        return span

    def _get_parent_span(self, parent_run_id) -> Optional[LiveSpan]:
        """
        Get parent span from multiple sources:
        1. If there is an active span in current context, use it as parent span
        2. If parent_run_id is provided, get the corresponding span from the run -> span mapping
        3. If none of the above, return None
        """
        if active_span := mlflow.get_current_active_span():
            return active_span
        elif parent_run_id:
            return self._get_span_by_run_id(parent_run_id)
        return None

    def _end_span(
        self,
        run_id: UUID,
        span: LiveSpan,
        outputs=None,
        attributes=None,
        status=SpanStatus(SpanStatusCode.OK),
    ):
        """Close MLflow Span (or Trace if it is root component)"""
        try:
            with maybe_set_prediction_context(self._prediction_context):
                span.end(
                    outputs=outputs,
                    attributes=attributes,
                    status=status,
                )
        finally:
            # Span should be detached from the context even when the client.end_span fails
            st = self._run_span_mapping.pop(str(run_id), None)
            if _should_attach_span_to_context.get():
                if st.token is None:
                    raise MlflowException(
                        f"Token for span {st.span} is not found. "
                        "Cannot detach the span from context."
                    )
                detach_span_from_context(st.token)

    def flush(self):
        """Flush the state of the tracer."""
        # Ideally, all spans should be popped and ended. However, LangChain sometimes
        # does not trigger the end event properly and some spans may be left open.
        # To avoid leaking tracing context, we remove all spans from the mapping.
        for st in self._run_span_mapping.values():
            if st.token:
                _logger.debug(f"Found leaked span {st.span}. Force ending it.")
                detach_span_from_context(st.token)

        self._run_span_mapping = {}

    def _assign_span_name(self, serialized: dict[str, Any], default_name="unknown") -> str:
        return serialized.get("name", serialized.get("id", [default_name])[-1])

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        """Run when a chat model starts running."""

        if metadata:
            kwargs.update({"metadata": metadata})

        span = self._start_span(
            span_name=name or self._assign_span_name(serialized, "chat model"),
            parent_run_id=parent_run_id,
            span_type=SpanType.CHAT_MODEL,
            run_id=run_id,
            inputs=messages,
            attributes=kwargs,
        )

        mlflow_messages = [
            convert_lc_message_to_chat_message(msg)
            for message_list in messages
            for msg in message_list
        ]
        set_span_chat_messages(span, mlflow_messages)

        if tools := self._extract_tool_definitions(kwargs):
            set_span_chat_tools(span, tools)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM (non-chat models) starts running."""
        if metadata:
            kwargs.update({"metadata": metadata})

        span = self._start_span(
            span_name=name or self._assign_span_name(serialized, "llm"),
            parent_run_id=parent_run_id,
            span_type=SpanType.LLM,
            run_id=run_id,
            inputs=prompts,
            attributes=kwargs,
        )

        mlflow_messages = [ChatMessage(role="user", content=prompt) for prompt in prompts]
        set_span_chat_messages(span, mlflow_messages)

        if tools := self._extract_tool_definitions(kwargs):
            set_span_chat_tools(span, tools)

    def _extract_tool_definitions(self, kwargs: dict[str, Any]) -> list[ChatTool]:
        raw_tools = kwargs.get("invocation_params", {}).get("tools", [])
        tools = []
        for raw_tool in raw_tools:
            # First, try to parse the raw tool dictionary as OpenAI-style tool
            try:
                tool = ChatTool.validate_compat(raw_tool)
                tools.append(tool)
            except pydantic.ValidationError:
                # If not OpenAI style, just try to extract the name and descriptions.
                if name := raw_tool.get("name"):
                    tool = ChatTool(
                        type="function",
                        function=FunctionToolDefinition(
                            name=name, description=raw_tool.get("description")
                        ),
                    )
                    tools.append(tool)
                else:
                    _logger.warning(f"Failed to parse tool definition for tracing: {raw_tool}.")

        return tools

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ):
        """Run on new LLM token. Only available when streaming is enabled."""
        llm_span = self._get_span_by_run_id(run_id)
        event_kwargs = {"token": token}
        if chunk:
            event_kwargs["chunk"] = dumps(chunk)
        llm_span.add_event(
            SpanEvent(
                name="new_token",
                attributes=event_kwargs,
            )
        )

    def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        **kwargs: Any,
    ):
        """Run on a retry event."""
        span = self._get_span_by_run_id(run_id)
        retry_d: dict[str, Any] = {
            "slept": retry_state.idle_for,
            "attempt": retry_state.attempt_number,
        }
        if retry_state.outcome is None:
            retry_d["outcome"] = "N/A"
        elif retry_state.outcome.failed:
            retry_d["outcome"] = "failed"
            exception = retry_state.outcome.exception()
            retry_d["exception"] = str(exception)
            retry_d["exception_type"] = exception.__class__.__name__
        else:
            retry_d["outcome"] = "success"
            retry_d["result"] = str(retry_state.outcome.result())
        span.add_event(
            SpanEvent(
                name="retry",
                attributes=retry_d,
            )
        )

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any):
        """End the span for an LLM run."""
        llm_span = self._get_span_by_run_id(run_id)

        # Record the chat messages attribute
        input_messages = llm_span.get_attribute(SpanAttributeKey.CHAT_MESSAGES) or []
        output_messages = [
            convert_lc_generation_to_chat_message(gen)
            for gen_list in response.generations
            for gen in gen_list
        ]
        set_span_chat_messages(llm_span, input_messages + output_messages)
        self._end_span(run_id, llm_span, outputs=response)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ):
        """Handle an error for an LLM run."""
        llm_span = self._get_span_by_run_id(run_id)
        llm_span.add_event(SpanEvent.from_exception(error))
        self._end_span(run_id, llm_span, status=SpanStatus(SpanStatusCode.ERROR, str(error)))

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: Union[dict[str, Any], Any],
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        run_type: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        """Start span for a chain run."""
        if metadata:
            kwargs.update({"metadata": metadata})
        # not considering streaming events for now
        self._start_span(
            span_name=name or self._assign_span_name(serialized, "chain"),
            parent_run_id=parent_run_id,
            span_type=SpanType.CHAIN,
            run_id=run_id,
            inputs=inputs,
            attributes=kwargs,
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        inputs: Optional[Union[dict[str, Any], Any]] = None,
        **kwargs: Any,
    ):
        """Run when chain ends running."""
        chain_span = self._get_span_by_run_id(run_id)
        if inputs:
            chain_span.set_inputs(inputs)
        self._end_span(run_id, chain_span, outputs=outputs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        inputs: Optional[Union[dict[str, Any], Any]] = None,
        run_id: UUID,
        **kwargs: Any,
    ):
        """Run when chain errors."""
        chain_span = self._get_span_by_run_id(run_id)
        if inputs:
            chain_span.set_inputs(inputs)
        chain_span.add_event(SpanEvent.from_exception(error))
        self._end_span(run_id, chain_span, status=SpanStatus(SpanStatusCode.ERROR, str(error)))

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        # We don't use inputs here because LangChain override the original inputs
        # with None for some cases. In order to avoid losing the original inputs,
        # we try to parse the input_str instead.
        # https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/tools/base.py#L636-L640
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Start span for a tool run."""
        if metadata:
            kwargs.update({"metadata": metadata})

        # For function calling, input_str can be a stringified dictionary
        # like "{'key': 'value'}". We try parsing it for better rendering,
        # but conservatively fallback to original if it fails.
        try:
            inputs = ast.literal_eval(input_str)
        except Exception:
            inputs = input_str

        self._start_span(
            span_name=name or self._assign_span_name(serialized, "tool"),
            parent_run_id=parent_run_id,
            span_type=SpanType.TOOL,
            run_id=run_id,
            inputs=inputs,
            attributes=kwargs,
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any):
        """Run when tool ends running."""
        tool_span = self._get_span_by_run_id(run_id)
        self._end_span(run_id, tool_span, outputs=output)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ):
        """Run when tool errors."""
        tool_span = self._get_span_by_run_id(run_id)
        tool_span.add_event(SpanEvent.from_exception(error))
        self._end_span(run_id, tool_span, status=SpanStatus(SpanStatusCode.ERROR, str(error)))

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        """Run when Retriever starts running."""
        if metadata:
            kwargs.update({"metadata": metadata})
        self._start_span(
            span_name=name or self._assign_span_name(serialized, "retriever"),
            parent_run_id=parent_run_id,
            span_type=SpanType.RETRIEVER,
            run_id=run_id,
            inputs=query,
            attributes=kwargs,
        )

    def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, **kwargs: Any):
        """Run when Retriever ends running."""
        retriever_span = self._get_span_by_run_id(run_id)
        try:
            # attempt to convert documents to MlflowDocument
            documents = [MlflowDocument.from_langchain_document(doc) for doc in documents]
        except Exception as e:
            _logger.debug(
                f"Failed to convert LangChain Document to MLflow Document: {e}",
                exc_info=True,
            )
        self._end_span(
            run_id,
            retriever_span,
            outputs=documents,
        )

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ):
        """Run when Retriever errors."""
        retriever_span = self._get_span_by_run_id(run_id)
        retriever_span.add_event(SpanEvent.from_exception(error))
        self._end_span(run_id, retriever_span, status=SpanStatus(SpanStatusCode.ERROR, str(error)))

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Any:
        """
        Run on agent action.

        NB: Agent action doesn't create a new LangChain Run, so instead of creating a new span,
        an action will be recorded as an event of the existing span created by a parent chain.
        """
        span = self._get_span_by_run_id(run_id)
        span.add_event(
            SpanEvent(
                name="agent_action",
                attributes={
                    "tool": action.tool,
                    "tool_input": dumps(action.tool_input),
                    "log": action.log,
                },
            )
        )

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Any:
        """Run on agent end."""
        span = self._get_span_by_run_id(run_id)
        span.add_event(
            SpanEvent(
                name="agent_finish",
                attributes={"return_values": dumps(finish.return_values), "log": finish.log},
            )
        )

    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on arbitrary text."""
        try:
            span = self._get_span_by_run_id(run_id)
        except MlflowException:
            _logger.warning("Span not found for text event. Skipping text event logging.")
        else:
            span.add_event(
                SpanEvent(
                    "text",
                    attributes={"text": text},
                )
            )
