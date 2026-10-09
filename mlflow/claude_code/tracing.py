"""MLflow tracing integration for Claude Code interactions."""

import dataclasses
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey, TraceMetadataKey
from mlflow.tracing.provider import _get_trace_exporter
from mlflow.tracing.trace_manager import InMemoryTraceManager

# ============================================================================
# CONSTANTS
# ============================================================================

# Used multiple times across the module
NANOSECONDS_PER_MS = 1e6
NANOSECONDS_PER_S = 1e9
MAX_PREVIEW_LENGTH = 1000

# Custom logging level for Claude tracing
CLAUDE_TRACING_LEVEL = logging.WARNING - 5


# ============================================================================
# LOGGING AND SETUP
# ============================================================================


def setup_logging() -> logging.Logger:
    """Set up logging directory and return configured logger.

    Creates .claude/mlflow directory structure and configures file-based logging
    with INFO level. Prevents log propagation to avoid duplicate messages.
    """
    # Create logging directory structure
    log_dir = Path(os.getcwd()) / ".claude" / "mlflow"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.handlers.clear()  # Remove any existing handlers

    # Configure file handler with timestamp formatting
    log_file = log_dir / "claude_tracing.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logging.addLevelName(CLAUDE_TRACING_LEVEL, "CLAUDE_TRACING")
    logger.setLevel(CLAUDE_TRACING_LEVEL)
    logger.propagate = False  # Prevent duplicate log messages

    return logger


_MODULE_LOGGER: logging.Logger | None = None


def get_logger() -> logging.Logger:
    """Get the configured module logger."""
    global _MODULE_LOGGER

    if _MODULE_LOGGER is None:
        _MODULE_LOGGER = setup_logging()
    return _MODULE_LOGGER


def _get_current_user() -> str:
    return os.environ.get("USER", "") or os.environ.get("USERNAME", "")


# ============================================================================
# INPUT/OUTPUT UTILITIES
# ============================================================================


def get_hook_response(error: str | None = None, **kwargs) -> dict[str, Any]:
    """Build hook response dictionary for Claude Code hook protocol.

    Args:
        error: Error message if hook failed, None if successful
        kwargs: Additional fields to include in response

    Returns:
        Hook response dictionary
    """
    if error is not None:
        return {"continue": False, "stopReason": error, **kwargs}
    return {"continue": True, **kwargs}


def _build_usage_dict(usage: dict[str, Any]) -> dict[str, int]:
    """Normalize a Claude Code usage payload into the CHAT_USAGE schema.

    Includes cache-read and cache-creation tokens in ``input_tokens`` and
    ``total_tokens``, matching ``mlflow.anthropic.autolog`` and cost calculation.
    Cache tokens are also exposed as separate optional keys so consumers can
    compute cache hit rate.
    """
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    usage_dict: dict[str, int] = {
        TokenUsageKey.INPUT_TOKENS: input_tokens,
        TokenUsageKey.OUTPUT_TOKENS: output_tokens,
        TokenUsageKey.TOTAL_TOKENS: input_tokens + output_tokens,
    }
    if (cached := usage.get("cache_read_input_tokens")) is not None:
        usage_dict[TokenUsageKey.CACHE_READ_INPUT_TOKENS] = cached
    if (created := usage.get("cache_creation_input_tokens")) is not None:
        usage_dict[TokenUsageKey.CACHE_CREATION_INPUT_TOKENS] = created
    # Anthropic reports input_tokens excluding cache tokens.
    if cache_total := (cached or 0) + (created or 0):
        usage_dict[TokenUsageKey.INPUT_TOKENS] += cache_total
        usage_dict[TokenUsageKey.TOTAL_TOKENS] += cache_total
    return usage_dict


def _set_token_usage_attribute(span, usage: dict[str, Any]) -> None:
    """Set token usage on a span using the standardized CHAT_USAGE attribute.

    Args:
        span: The MLflow span to set token usage on
        usage: Dictionary containing token usage info from Claude Agent SDK
    """
    if not usage:
        return

    span.set_attribute(SpanAttributeKey.CHAT_USAGE, _build_usage_dict(usage))


def _finalize_trace(
    parent_span,
    user_prompt: str,
    final_response: str | None,
    session_id: str | None,
    end_time_ns: int | None = None,
    usage: dict[str, Any] | None = None,
) -> mlflow.entities.Trace:
    try:
        # Set trace previews and metadata for UI display
        with InMemoryTraceManager.get_instance().get_trace(parent_span.trace_id) as in_memory_trace:
            if user_prompt:
                in_memory_trace.info.request_preview = user_prompt[:MAX_PREVIEW_LENGTH]
            if final_response:
                in_memory_trace.info.response_preview = final_response[:MAX_PREVIEW_LENGTH]

            metadata = {
                TraceMetadataKey.TRACE_USER: _get_current_user(),
                "mlflow.trace.working_directory": os.getcwd(),
            }
            if session_id:
                metadata[TraceMetadataKey.TRACE_SESSION] = session_id

            # Set token usage directly on trace metadata so it survives
            # even if span-level aggregation doesn't pick it up
            if usage:
                metadata[TraceMetadataKey.TOKEN_USAGE] = json.dumps(_build_usage_dict(usage))

            in_memory_trace.info.trace_metadata = {
                **in_memory_trace.info.trace_metadata,
                **metadata,
            }
    except Exception as e:
        get_logger().warning("Failed to update trace metadata and previews: %s", e)

    outputs = {"status": "completed"}
    if final_response:
        outputs["response"] = final_response
    parent_span.set_outputs(outputs)
    parent_span.end(end_time_ns=end_time_ns)
    _flush_trace_async_logging()
    get_logger().log(CLAUDE_TRACING_LEVEL, "Created MLflow trace: %s", parent_span.trace_id)
    return mlflow.get_trace(parent_span.trace_id)


def _flush_trace_async_logging() -> None:
    try:
        if hasattr(_get_trace_exporter(), "_async_queue"):
            mlflow.flush_trace_async_logging()
    except Exception as e:
        get_logger().debug("Failed to flush trace async logging: %s", e)


# ============================================================================
# SDK MESSAGE PROCESSING
# ============================================================================


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
    """Map tool_use_id to its result content so tool spans can show outputs."""
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


# Maps SDK dataclass names to Anthropic API "type" discriminators.
# dataclasses.asdict() gives us the fields but not the type tag that
# the Anthropic message format requires on every content block.
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
        elif isinstance(content, list):
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
    """Create LLM and tool child spans under ``parent_span`` from SDK messages."""
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
    """
    Build an MLflow trace from Claude Agent SDK message objects.

    Args:
        messages: List of SDK message objects (UserMessage, AssistantMessage,
            ResultMessage, etc.) captured during a conversation.
        session_id: Optional session identifier for grouping traces.

    Returns:
        MLflow Trace if successful, None if no user prompt is found or processing fails.
    """
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

        # Prefer the SDK's own session_id, fall back to caller arg
        session_id = (result_msg.session_id if result_msg else None) or session_id

        get_logger().log(
            CLAUDE_TRACING_LEVEL,
            "Creating MLflow trace for session: %s",
            session_id,
        )

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

        # Set token usage on the root span so it aggregates into trace-level usage
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
