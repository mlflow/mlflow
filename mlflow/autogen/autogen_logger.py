import functools
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Union

from autogen import Agent, ConversableAgent
from autogen.logger.base_logger import BaseLogger
from openai.types.chat import ChatCompletion

from mlflow import MlflowClient
from mlflow.entities.span import NoOpSpan, Span, SpanType
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatus, SpanStatusCode
from mlflow.tracing.utils import capture_function_input_args
from mlflow.utils.autologging_utils import autologging_is_disabled
from mlflow.utils.autologging_utils.safety import safe_patch

# For GroupChat, a single "received_message" events are passed around multiple
# internal layers and thus too verbose if we show them all. Therefore we ignore
# some of the message senders listed below.
_EXCLUDED_MESSAGE_SENDERS = ["chat_manager", "checking_agent"]

_logger = logging.getLogger(__name__)


FLAVOR_NAME = "autogen"


@dataclass
class ChatState:
    """
    Represents the state of a chat session.
    """

    # The root span object that scopes the entire single chat session. All spans
    # such as LLM, function calls, in the chat session should be children of this span.
    session_span: Optional[Span] = None
    # The last message object in the chat session.
    last_message: Optional[Any] = None
    # The timestamp (ns) of the last message in the chat session.
    last_message_timestamp: int = 0
    # LLM/Tool Spans created after the last message in the chat session.
    # We consider them as operations for generating the next message and
    # re-locate them under the corresponding message span.
    pending_spans: list[Span] = field(default_factory=list)

    def clear(self):
        self.session_span = None
        self.last_message = None
        self.last_message_timestamp = 0
        self.pending_spans = []


def _catch_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            _logger.error(f"Error occurred during AutoGen tracing: {e}")

    return wrapper


class MlflowAutogenLogger(BaseLogger):
    def __init__(self):
        self._client = MlflowClient()
        self._chat_state = ChatState()

    def start(self) -> str:
        return "session_id"

    @_catch_exception
    def log_new_agent(self, agent: ConversableAgent, init_args: dict[str, Any]) -> None:
        """
        This handler is called whenever a new agent instance is created.
        Here we patch the agent's methods to start and end a trace around its chat session.
        """
        # TODO: Patch generate_reply() method as well
        if hasattr(agent, "initiate_chat"):
            safe_patch(
                FLAVOR_NAME,
                agent.__class__,
                "initiate_chat",
                # Setting root_only = True because sometimes compounded agent calls initiate_chat()
                # method of its sub-agents, which should not start a new trace.
                self._get_patch_function(root_only=True),
            )
        if hasattr(agent, "register_function"):

            def patched(original, _self, function_map):
                original(_self, function_map)
                # Wrap the newly registered tools to start and end a span around its invocation.
                for name, f in function_map.items():
                    if f is not None:
                        _self._function_map[name] = functools.partial(
                            self._get_patch_function(span_type=SpanType.TOOL), f
                        )

            safe_patch(FLAVOR_NAME, agent.__class__, "register_function", patched)

    def _get_patch_function(self, span_type: str = SpanType.UNKNOWN, root_only: bool = False):
        """
        Patch a function to start and end a span around its invocation.

        Args:
            f: The function to patch.
            span_name: The name of the span. If None, the function name is used.
            span_type: The type of the span. Default is SpanType.UNKNOWN.
            root_only: If True, only create a span if it is the root of the chat session.
                When there is an existing root span for the chat session, the function will
                not create a new span.
        """

        def _wrapper(original, *args, **kwargs):
            # If autologging is disabled, just run the original function. This is a safety net to
            # prevent patching side effects from being effective after autologging is disabled.
            if autologging_is_disabled(FLAVOR_NAME):
                return original(*args, **kwargs)

            if self._chat_state.session_span is None:
                # Create the trace per chat session
                span = self._client.start_trace(
                    name=original.__name__,
                    span_type=span_type,
                    inputs=capture_function_input_args(original, args, kwargs),
                )
                self._chat_state.session_span = span
                try:
                    result = original(*args, **kwargs)
                except Exception as e:
                    result = None
                    self._record_exception(span, e)
                    raise e
                finally:
                    self._client.end_trace(
                        request_id=span.request_id, outputs=result, status=span.status
                    )
                    # Clear the state to start a new chat session
                    self._chat_state.clear()
            elif not root_only:
                span = self._start_span_in_session(
                    name=original.__name__,
                    span_type=span_type,
                    inputs=capture_function_input_args(original, args, kwargs),
                )
                try:
                    result = original(*args, **kwargs)
                except Exception as e:
                    result = None
                    self._record_exception(span, e)
                    raise e
                finally:
                    self._client.end_span(
                        request_id=span.request_id,
                        span_id=span.span_id,
                        outputs=result,
                        status=span.status,
                    )
                    self._chat_state.pending_spans.append(span)
            else:
                result = original(*args, **kwargs)
            return result

        return _wrapper

    def _record_exception(self, span: Span, e: Exception):
        try:
            span.set_status(SpanStatus(SpanStatusCode.ERROR, str(e)))
            span.add_event(SpanEvent.from_exception(e))
        except Exception as e:
            _logger.warning(
                "Failed to record exception in span.", exc_info=_logger.isEnabledFor(logging.DEBUG)
            )

    def _start_span_in_session(
        self,
        name: str,
        span_type: str,
        inputs: dict[str, Any],
        attributes: Optional[dict[str, Any]] = None,
        start_time_ns: Optional[int] = None,
    ) -> Span:
        """
        Start a span in the current chat session.
        """
        if self._chat_state.session_span is None:
            _logger.warning("Failed to start span. No active chat session.")
            return NoOpSpan()

        return self._client.start_span(
            request_id=self._chat_state.session_span.request_id,
            # Tentatively set the parent ID to the session root span, because we
            # cannot create a span without a parent span (otherwise it will start
            # a new trace). The actual parent will be determined once the chat
            # message is received.
            parent_id=self._chat_state.session_span.span_id,
            name=name,
            span_type=span_type,
            inputs=inputs,
            attributes=attributes,
            start_time_ns=start_time_ns,
        )

    @_catch_exception
    def log_event(self, source: Union[str, Agent], name: str, **kwargs: dict[str, Any]):
        event_end_time = time.time_ns()
        if name == "received_message":
            if (self._chat_state.last_message is not None) and (
                kwargs.get("sender") not in _EXCLUDED_MESSAGE_SENDERS
            ):
                span = self._start_span_in_session(
                    name=kwargs["sender"],
                    # Last message is recorded as the input of the next message
                    inputs=self._chat_state.last_message,
                    span_type=SpanType.AGENT,
                    start_time_ns=self._chat_state.last_message_timestamp,
                )
                self._client.end_span(
                    request_id=span.request_id,
                    span_id=span.span_id,
                    outputs=kwargs,
                    end_time_ns=event_end_time,
                )

                # Re-locate the pended spans under this message span
                for child_span in self._chat_state.pending_spans:
                    child_span._span._parent = span._span.context
                self._chat_state.pending_spans = []

            self._chat_state.last_message = kwargs
            self._chat_state.last_message_timestamp = event_end_time

    @_catch_exception
    def log_chat_completion(
        self,
        invocation_id: uuid.UUID,
        client_id: int,
        wrapper_id: int,
        source: Union[str, Agent],
        request: dict[str, Union[float, str, list[dict[str, str]]]],
        response: Union[str, ChatCompletion],
        is_cached: int,
        cost: float,
        start_time: str,
    ) -> None:
        # The start_time passed from AutoGen is in UTC timezone.
        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")
        start_dt = start_dt.replace(tzinfo=timezone.utc)
        start_time_ns = int(start_dt.timestamp() * 1e9)
        span = self._start_span_in_session(
            name="chat_completion",
            span_type=SpanType.LLM,
            inputs=request,
            attributes={
                "source": source,
                "client_id": client_id,
                "invocation_id": invocation_id,
                "wrapper_id": wrapper_id,
                "cost": cost,
                "is_cached": is_cached,
            },
            start_time_ns=start_time_ns,
        )
        self._client.end_span(
            request_id=span.request_id,
            span_id=span.span_id,
            outputs=response,
            end_time_ns=time.time_ns(),
        )
        self._chat_state.pending_spans.append(span)

    # The following methods are not used but are required to implement the BaseLogger interface.
    @_catch_exception
    def log_function_use(self, *args: Any, **kwargs: Any):
        pass

    @_catch_exception
    def log_new_wrapper(self, wrapper, init_args):
        pass

    @_catch_exception
    def log_new_client(self, client, wrapper, init_args):
        pass

    @_catch_exception
    def stop(self) -> None:
        pass

    @_catch_exception
    def get_connection(self):
        pass
