import ast
import logging
from contextvars import ContextVar
from typing import Any, Optional, Sequence
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
from mlflow.langchain.utils.chat import parse_token_usage
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.provider import detach_span_from_context, set_span_in_context
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import maybe_set_prediction_context, set_span_chat_tools
from mlflow.tracing.utils.token import SpanWithToken
from mlflow.types.chat import ChatTool, FunctionToolDefinition
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

    def _get_span_by_run_id(self, run_id: UUID) -> LiveSpan | None:
        if span_with_token := self._run_span_mapping.get(str(run_id), None):
            return span_with_token.span
        raise MlflowException(f"Span for run_id {run_id!s} not found.")

    def _serialize_invocation_params(
        self, attributes: dict[str, Any] | None
    ) -> dict[str, Any] | None:
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
        parent_run_id: UUID | None,
        span_type: str,
        run_id: UUID,
        inputs: str | dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
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

    def _get_parent_span(self, parent_run_id) -> LiveSpan | None:
        """
        Get parent span to create a new span under.

        Ideally, we can simply rely on the active span in current context. However, LangChain
        execution heavily uses threads and asyncio, and sometimes ContextVar is not correctly
        propagated, resulting in missing parent span.

        To address this, we check two sources of parent span:

        1. An active span in current MLflow tracing context (get_current_active_span)
        2. If parent_run_id is given by LangChain, get the corresponding span from the mapping

        The complex case is when BOTH are present but different. In this case, we need to
        resolve the correct parent span by traversing the span tree.
        """
        parent_mlflow_span = mlflow.get_current_active_span()
        parent_lc_span = self._get_span_by_run_id(parent_run_id) if parent_run_id else None

        if parent_mlflow_span and parent_lc_span:
            if parent_mlflow_span.span_id == parent_lc_span.span_id:
                return parent_mlflow_span
            else:
                return self._resolve_parent_span(parent_mlflow_span, parent_lc_span)
        elif parent_mlflow_span:
            return parent_mlflow_span
        elif parent_lc_span:
            return parent_lc_span

    def _resolve_parent_span(self, parent_mlflow_span, parent_lc_span):
        """
        Resolve the correct parent span when both MLflow and LangChain provide different
        parent spans.

        For example, the following two examples are mostly same but slightly different: where the
        mlflow.start_span() is used.


        For example, the following two examples are mostly same but slightly different: where the
        mlflow.start_span() is used.

        ```python
        llm = ChatOpenAI()


        @tool
        def custom_tool_node(inputs):
            response = ChatOpenAI().invoke(...)
            return response.content


        graph = create_react_agent(llm, [custom_tool_node])

        with mlflow.start_span("parent"):
            graph.invoke({"prompt": "Hello"})
        ```

        The correct span structure for this case is [parent] -> [tool] -> [ChatOpenAI]

        ```python
        @tool
        def custom_tool_node(inputs):
            with mlflow.start_span("parent"):
                response = ChatOpenAI().invoke(...)
                return response.content


        graph = create_react_agent(llm, [custom_tool_node])
        graph.invoke({"prompt": "Hello"})
        ```

        The correct span structure for this case is [tool] -> [parent] -> [ChatOpenAI]

        When we try to create a new span for ChatOpenAI, we need to determine which span is the
        parent span, "parent" or "tool". Unfortunately, there is no way to decide this from
        metadata provided in the span itself, so we need to traverse the span tree and check
        if one is parent of the other.
        """
        trace_manager = InMemoryTraceManager.get_instance()
        span = parent_mlflow_span
        while span.parent_id:
            if span.parent_id == parent_lc_span.span_id:
                # MLflow parent span is under the LangChain
                # langchain_span
                #  └──  mlflow_span
                #       └── current span
                return parent_mlflow_span

            span = trace_manager.get_span_from_id(span.trace_id, span.parent_id)

        # MLflow span is parent of LangChain span
        # mlflow_span
        #  └── langchain_span
        #       └── current span
        #
        # or two spans are not related at all, then fallback to LangChain one.
        return parent_lc_span

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
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ):
        """Run when a chat model starts running."""

        if metadata:
            kwargs.update({"metadata": metadata})
        kwargs[SpanAttributeKey.MESSAGE_FORMAT] = "langchain"

        span = self._start_span(
            span_name=name or self._assign_span_name(serialized, "chat model"),
            parent_run_id=parent_run_id,
            span_type=SpanType.CHAT_MODEL,
            run_id=run_id,
            inputs=messages,
            attributes=kwargs,
        )

        if tools := self._extract_tool_definitions(kwargs):
            set_span_chat_tools(span, tools)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM (non-chat models) starts running."""
        if metadata:
            kwargs.update({"metadata": metadata})
        kwargs[SpanAttributeKey.MESSAGE_FORMAT] = "langchain"

        span = self._start_span(
            span_name=name or self._assign_span_name(serialized, "llm"),
            parent_run_id=parent_run_id,
            span_type=SpanType.LLM,
            run_id=run_id,
            inputs=prompts,
            attributes=kwargs,
        )

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
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
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
        # response.generations is a nested list of messages
        generations = [g for gen_list in response.generations for g in gen_list]

        # Record the token usage attribute
        try:
            if usage := parse_token_usage(generations):
                llm_span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)
        except Exception as e:
            _logger.debug(f"Failed to log token usage for LangChain: {e}", exc_info=True)

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
        inputs: dict[str, Any] | Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        run_type: str | None = None,
        name: str | None = None,
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
        inputs: dict[str, Any] | Any | None = None,
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
        inputs: dict[str, Any] | Any | None = None,
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
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        # We don't use inputs here because LangChain override the original inputs
        # with None for some cases. In order to avoid losing the original inputs,
        # we try to parse the input_str instead.
        # https://github.com/langchain-ai/langchain/blob/2813e8640703b8066d8dd6c739829bb4f4aa634e/libs/core/langchain_core/tools/base.py#L636-L640
        inputs: dict[str, Any] | None = None,
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
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
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
        parent_run_id: UUID | None = None,
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
