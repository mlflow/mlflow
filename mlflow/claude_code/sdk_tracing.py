"""Claude Agent SDK trace construction."""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any

import mlflow
from mlflow.claude_code.tracing import (
    CLAUDE_TRACING_LEVEL,
    NANOSECONDS_PER_MS,
    NANOSECONDS_PER_S,
    _finalize_trace,
    _set_token_usage_attribute,
    get_logger,
)
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey


def _find_sdk_user_prompt(messages: list[Any]) -> str | None:
    from claude_agent_sdk.types import TextBlock, UserMessage

    for msg in messages:
        if not isinstance(msg, UserMessage) or msg.tool_use_result is not None:
            continue
        content = msg.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(block.text for block in content if isinstance(block, TextBlock))
        else:
            continue
        if text and text.strip():
            return text
    return None


def _build_tool_result_map(messages: list[Any]) -> dict[str, str]:
    from claude_agent_sdk.types import ToolResultBlock, UserMessage

    tool_result_map: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, UserMessage) and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    result = block.content
                    if isinstance(result, list):
                        result = str(result)
                    tool_result_map[block.tool_use_id] = result or ""
    return tool_result_map


_CONTENT_BLOCK_TYPES = {
    "TextBlock": "text",
    "ToolUseBlock": "tool_use",
    "ToolResultBlock": "tool_result",
}


def _serialize_content_block(block) -> dict[str, Any] | None:
    block_type = _CONTENT_BLOCK_TYPES.get(type(block).__name__)
    if not block_type:
        return None
    fields = {key: value for key, value in dataclasses.asdict(block).items() if value is not None}
    fields["type"] = block_type
    return fields


def _serialize_sdk_message(msg) -> dict[str, Any] | None:
    from claude_agent_sdk.types import AssistantMessage, UserMessage

    if isinstance(msg, UserMessage):
        content = msg.content
        if isinstance(content, str):
            return {"role": "user", "content": content} if content.strip() else None
        if isinstance(content, list):
            if parts := [
                serialized for block in content if (serialized := _serialize_content_block(block))
            ]:
                return {"role": "user", "content": parts}
    elif isinstance(msg, AssistantMessage) and msg.content:
        if parts := [
            serialized for block in msg.content if (serialized := _serialize_content_block(block))
        ]:
            return {"role": "assistant", "content": parts}
    return None


def _create_sdk_child_spans(
    messages: list[Any],
    parent_span,
    tool_result_map: dict[str, str],
) -> str | None:
    from claude_agent_sdk.types import AssistantMessage, TextBlock, ToolUseBlock

    final_response = None
    pending_messages: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, AssistantMessage) and msg.content:
            text_blocks = [block for block in msg.content if isinstance(block, TextBlock)]
            tool_blocks = [block for block in msg.content if isinstance(block, ToolUseBlock)]

            if text_blocks and not tool_blocks:
                text = "\n".join(block.text for block in text_blocks)
                if text.strip():
                    final_response = text

                llm_span = mlflow.start_span_no_context(
                    name="llm",
                    parent_span=parent_span,
                    span_type=SpanType.LLM,
                    inputs={
                        "model": getattr(msg, "model", "unknown"),
                        "messages": pending_messages,
                    },
                    attributes={
                        "model": getattr(msg, "model", "unknown"),
                        SpanAttributeKey.MESSAGE_FORMAT: "anthropic",
                    },
                )
                llm_span.set_outputs({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": block.text} for block in text_blocks],
                })
                llm_span.end()
                pending_messages = []
                continue

            for tool_block in tool_blocks:
                tool_span = mlflow.start_span_no_context(
                    name=f"tool_{tool_block.name}",
                    parent_span=parent_span,
                    span_type=SpanType.TOOL,
                    inputs=tool_block.input,
                    attributes={"tool_name": tool_block.name, "tool_id": tool_block.id},
                )
                tool_span.set_outputs({"result": tool_result_map.get(tool_block.id, "")})
                tool_span.end()

        if anthropic_msg := _serialize_sdk_message(msg):
            pending_messages.append(anthropic_msg)

    return final_response


def process_sdk_messages(
    messages: list[Any], session_id: str | None = None
) -> mlflow.entities.Trace | None:
    """Build an MLflow trace from Claude Agent SDK messages."""
    from claude_agent_sdk.types import ResultMessage

    try:
        if not messages:
            get_logger().warning("Empty messages list, skipping")
            return None

        user_prompt = _find_sdk_user_prompt(messages)
        if user_prompt is None:
            get_logger().warning("No user prompt found in SDK messages")
            return None

        result_msg = next((msg for msg in messages if isinstance(msg, ResultMessage)), None)
        session_id = (result_msg.session_id if result_msg else None) or session_id
        get_logger().log(CLAUDE_TRACING_LEVEL, "Creating MLflow trace for session: %s", session_id)

        tool_result_map = _build_tool_result_map(messages)

        if duration_ms := (getattr(result_msg, "duration_ms", None) if result_msg else None):
            duration_ns = int(duration_ms * NANOSECONDS_PER_MS)
            now_ns = int(datetime.now().timestamp() * NANOSECONDS_PER_S)
            start_time_ns = now_ns - duration_ns
            end_time_ns = now_ns
        else:
            start_time_ns = None
            end_time_ns = None

        parent_span = mlflow.start_span_no_context(
            name="claude_code_conversation",
            inputs={"prompt": user_prompt},
            span_type=SpanType.AGENT,
            start_time_ns=start_time_ns,
        )

        final_response = _create_sdk_child_spans(messages, parent_span, tool_result_map)

        usage = getattr(result_msg, "usage", None) if result_msg else None
        if usage:
            _set_token_usage_attribute(parent_span, usage)

        return _finalize_trace(
            parent_span,
            user_prompt,
            final_response,
            session_id,
            end_time_ns=end_time_ns,
            usage=usage,
        )
    except Exception as e:
        get_logger().error("Error processing SDK messages: %s", e, exc_info=True)
        return None
