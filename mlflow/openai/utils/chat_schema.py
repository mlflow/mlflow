import logging
from typing import Any

from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.exceptions import MlflowException
from mlflow.tracing import set_span_chat_messages, set_span_chat_tools

_logger = logging.getLogger(__name__)


def set_span_chat_attributes(span: LiveSpan, inputs: dict[str, Any], output: Any):
    from openai.types.chat import ChatCompletion

    if span.span_type not in (SpanType.CHAT_MODEL, SpanType.LLM):
        return

    messages = []
    if "messages" in inputs:
        messages.extend(inputs["messages"])

    if isinstance(output, ChatCompletion):
        messages.append(output.choices[0].message.to_dict(exclude_unset=True))
    elif isinstance(output, str):
        messages.append({"role": "assistant", "content": output})

    try:
        set_span_chat_messages(span, messages)
    except MlflowException:
        _logger.debug(
            "Failed to set chat messages on span",
            exc_info=True,
        )

    if tools := inputs.get("tools"):
        try:
            set_span_chat_tools(span, tools)
        except MlflowException:
            _logger.debug("Failed to set chat tools on span", exc_info=True)
