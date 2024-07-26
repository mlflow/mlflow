import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Union
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
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
from mlflow import MlflowClient
from mlflow.entities import LiveSpan, SpanEvent, SpanStatus, SpanStatusCode, SpanType
from mlflow.exceptions import MlflowException
from mlflow.pyfunc.context import Context, maybe_set_prediction_context
from mlflow.utils.autologging_utils import ExceptionSafeAbstractClass

_logger = logging.getLogger(__name__)
# Vector Search index column names
VS_INDEX_ID_COL = "chunk_id"
VS_INDEX_DOC_URL_COL = "doc_uri"


class MlflowLangchainTracer(BaseCallbackHandler, metaclass=ExceptionSafeAbstractClass):
    """
    Callback for auto-logging traces.
    We need to inherit ExceptionSafeAbstractClass to avoid invalid new
    input arguments added to original function call.

    Args:
        parent_span: Optional parent span for the trace. If provided, spans will be
            created under the given parent span. Otherwise, a single trace will be
            created. Example usage:

            .. code-block:: python

                from mlflow.langchain.langchain_tracer import MlflowLangchainTracer
                from langchain_community.chat_models import ChatDatabricks

                chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat")
                with mlflow.start_span("Custom root span") as root_span:
                    chat_model.invoke(
                        "What is MLflow?",
                        config={"callbacks": [MlflowLangchainTracer(root_span)]},
                    )
        prediction_context: Optional prediction context object to be set for the
            thread-local context. Occasionally this has to be passed manually because
            the callback may be invoked asynchronously and Langchain doesn't correctly
            propagate the thread-local context.
    """

    def __init__(
        self, parent_span: Optional[LiveSpan] = None, prediction_context: Optional[Context] = None
    ):
        # NB: The tracer can handle multiple traces in parallel under multi-threading scenarios.
        # DO NOT use instance variables to manage the state of single trace.
        super().__init__()
        self._mlflow_client = MlflowClient()
        self._parent_span = parent_span
        self._run_span_mapping: Dict[str, LiveSpan] = {}
        self._active_request_ids: Set[str] = set()
        self._prediction_context = prediction_context

    def _get_span_by_run_id(self, run_id: UUID) -> Optional[LiveSpan]:
        if span := self._run_span_mapping.get(str(run_id)):
            return span
        raise MlflowException(f"Span for run_id {run_id!s} not found.")

    def _start_span(
        self,
        span_name: str,
        parent_run_id: Optional[UUID],
        span_type: str,
        run_id: UUID,
        inputs: Optional[Dict[str, Any]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> LiveSpan:
        """Start MLflow Span (or Trace if it is root component)"""
        with maybe_set_prediction_context(self._prediction_context):
            parent = self._get_parent_span(parent_run_id)
            if parent:
                span = self._mlflow_client.start_span(
                    name=span_name,
                    request_id=parent.request_id,
                    parent_id=parent.span_id,
                    span_type=span_type,
                    inputs=inputs,
                    attributes=attributes,
                )
            else:
                # When parent_run_id is None, this is root component so start trace
                dependencies_schemas = (
                    self._prediction_context.dependencies_schemas
                    if self._prediction_context
                    else None
                )
                span = self._mlflow_client.start_trace(
                    name=span_name,
                    span_type=span_type,
                    inputs=inputs,
                    attributes=attributes,
                    tags=dependencies_schemas,
                )
                self._active_request_ids.add(span.request_id)

            self._run_span_mapping[str(run_id)] = span
        return span

    def _get_parent_span(self, parent_run_id) -> Optional[LiveSpan]:
        """
        Get parent span from multiple sources:
        1. If parent_run_id is provided, get the corresponding span from the run -> span mapping
        2. If parent_span argument is passed to the callback, use it as parent span
        3. If there is an active span, use it as parent span
        4. If none of the above, return None
        """
        if parent_run_id:
            return self._get_span_by_run_id(parent_run_id)
        elif self._parent_span:
            return self._parent_span
        elif active_span := mlflow.get_current_active_span():
            return active_span
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
        self._run_span_mapping.pop(str(run_id), None)

        if not self._is_trace_active(span.request_id):
            # A trace (root span) may be already ended i.e. a parent span ends earlier then its
            # child. For example, this occurs during streaming inference if the generator
            # returned by stream() is not consumed completely while the child span still
            # wait until the stream is exhausted.
            _logger.debug(
                f"Request ID {span.request_id} is not started or already ended. "
                f"Skipping end span for {span}."
            )
            return

        # Remove the request ID from the active list if the span being ended is the root span
        if (span.parent_id is None) and (span.request_id in self._active_request_ids):
            self._active_request_ids.remove(span.request_id)

        with maybe_set_prediction_context(self._prediction_context):
            self._mlflow_client.end_span(
                request_id=span.request_id,
                span_id=span.span_id,
                outputs=outputs,
                attributes=attributes,
                status=status,
            )

    def _is_trace_active(self, request_id: str) -> bool:
        """Check if a trace with the given request ID is active (i.e. not ended yet)"""
        return (
            # Case 1: The root span is started by this callback, the ID
            # should be in the active list, otherwise it's already ended.
            request_id in self._active_request_ids
            # Case 2: The root span is created by fluent API outside this callback.
            # In this case, we check the context to see if the trace is active or not.
            or (
                (active_span := mlflow.get_current_active_span())
                and (active_span.request_id == request_id)
            )
            # Case 3: The root span is created by client API outside this callback,
            # and passed via the `parent_span` argument of the callback. In this case,
            # we have no way to check if it is active or not, so just assume it is.
            or self._parent_span
        )

    def _reset(self):
        self._run_span_mapping = {}

    def _assign_span_name(self, serialized: Dict[str, Any], default_name="unknown") -> str:
        return serialized.get("name", serialized.get("id", [default_name])[-1])

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        """Run when a chat model starts running."""
        if metadata:
            kwargs.update({"metadata": metadata})
        self._start_span(
            span_name=name or self._assign_span_name(serialized, "chat model"),
            parent_run_id=parent_run_id,
            span_type=SpanType.CHAT_MODEL,
            run_id=run_id,
            inputs=messages,
            attributes=kwargs,
        )

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM (non-chat models) starts running."""
        if metadata:
            kwargs.update({"metadata": metadata})
        self._start_span(
            span_name=name or self._assign_span_name(serialized, "llm"),
            parent_run_id=parent_run_id,
            span_type=SpanType.LLM,
            run_id=run_id,
            inputs=prompts,
            attributes=kwargs,
        )

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
        retry_d: Dict[str, Any] = {
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
        outputs = response.dict()
        self._end_span(run_id, llm_span, outputs=outputs)

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
        serialized: Dict[str, Any],
        inputs: Union[Dict[str, Any], Any],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        inputs: Optional[Union[Dict[str, Any], Any]] = None,
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
        inputs: Optional[Union[Dict[str, Any], Any]] = None,
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
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Start span for a tool run."""
        if metadata:
            kwargs.update({"metadata": metadata})
        self._start_span(
            span_name=name or self._assign_span_name(serialized, "tool"),
            parent_run_id=parent_run_id,
            span_type=SpanType.TOOL,
            run_id=run_id,
            inputs=input_str,
            attributes=kwargs,
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any):
        """Run when tool ends running."""
        tool_span = self._get_span_by_run_id(run_id)
        self._end_span(run_id, tool_span, outputs=str(output))

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
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
        self._end_span(run_id, retriever_span, outputs=documents)

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

    def flush_tracker(self):
        try:
            self._reset()
        except Exception as e:
            _logger.debug(f"Failed to flush MLflow tracer due to error {e}.")
