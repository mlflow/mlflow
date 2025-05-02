import logging
from typing import Any

from mlflow.entities.span import LiveSpan
from mlflow.exceptions import MlflowException
from mlflow.tracing import set_span_chat_messages

_logger = logging.getLogger(__name__)


def set_span_chat_attributes(span: LiveSpan, messages: list[dict[str, Any]], output: Any):
    """
    This method logs chat messages to a span when LLM class is called.

    Args:
        span: A span object to record the chat messages.
        messages: Input messages for the LLM call.
        output: Response from LLM. Though the signature is str, tool response might be returned.
    """
    content = str(output.get("content", "")) if isinstance(output, dict) else str(output)
    output_message = {"role": "assistant", "content": content}

    try:
        set_span_chat_messages(span, [*messages, output_message])
    except MlflowException:
        _logger.debug(
            "Failed to set chat messages on span",
            exc_info=True,
        )
