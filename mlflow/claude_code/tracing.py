"""MLflow tracing integration for Claude Code interactions."""

import dataclasses
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import dateutil.parser

import mlflow
from mlflow.claude_code.config import (
    MLFLOW_TRACING_ENABLED,
    get_env_var,
)
from mlflow.entities import SpanType
from mlflow.environment_variables import (
    MLFLOW_EXPERIMENT_ID,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
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

MESSAGE_TYPE_USER = "user"
MESSAGE_TYPE_ASSISTANT = "assistant"
CONTENT_TYPE_TEXT = "text"
CONTENT_TYPE_TOOL_USE = "tool_use"
CONTENT_TYPE_TOOL_RESULT = "tool_result"
MESSAGE_FIELD_CONTENT = "content"
MESSAGE_FIELD_TYPE = "type"
MESSAGE_FIELD_MESSAGE = "message"
MESSAGE_FIELD_TIMESTAMP = "timestamp"

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


def setup_mlflow() -> None:
    """Configure MLflow tracking URI and experiment."""
    if not is_tracing_enabled():
        return

    # Get tracking URI from environment/settings
    mlflow.set_tracking_uri(get_env_var(MLFLOW_TRACKING_URI.name))

    # Set experiment if specified via environment variables
    experiment_id = get_env_var(MLFLOW_EXPERIMENT_ID.name)
    experiment_name = get_env_var(MLFLOW_EXPERIMENT_NAME.name)

    try:
        if experiment_id:
            mlflow.set_experiment(experiment_id=experiment_id)
        elif experiment_name:
            mlflow.set_experiment(experiment_name)
    except Exception as e:
        get_logger().warning("Failed to set experiment: %s", e)


def is_tracing_enabled() -> bool:
    """Check if MLflow Claude tracing is enabled via environment variable."""
    return get_env_var(MLFLOW_TRACING_ENABLED).lower() in ("true", "1", "yes")


# ============================================================================
# INPUT/OUTPUT UTILITIES
# ============================================================================


def read_hook_input() -> dict[str, Any]:
    """Read JSON input from stdin for Claude Code hook processing."""
    try:
        input_data = sys.stdin.read()
        return json.loads(input_data)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Failed to parse hook input: {e}", input_data, 0) from e


def read_transcript(transcript_path: str) -> list[dict[str, Any]]:
    """Read and parse a Claude Code conversation transcript from JSONL file."""
    with open(transcript_path, encoding="utf-8") as f:
        lines = f.readlines()
        return [json.loads(line) for line in lines if line.strip()]


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


# ============================================================================
# TIMESTAMP AND CONTENT PARSING UTILITIES
# ============================================================================


def parse_timestamp_to_ns(timestamp: str | int | float | None) -> int | None:
    """Convert various timestamp formats to nanoseconds since Unix epoch.

    Args:
        timestamp: Can be ISO string, Unix timestamp (seconds/ms), or nanoseconds

    Returns:
        Nanoseconds since Unix epoch, or None if parsing fails
    """
    if not timestamp:
        return None

    if isinstance(timestamp, str):
        try:
            dt = dateutil.parser.parse(timestamp)
            return int(dt.timestamp() * NANOSECONDS_PER_S)
        except Exception:
            get_logger().warning("Could not parse timestamp: %s", timestamp)
            return None
    if isinstance(timestamp, (int, float)):
        if timestamp < 1e10:
            return int(timestamp * NANOSECONDS_PER_S)
        if timestamp < 1e13:
            return int(timestamp * NANOSECONDS_PER_MS)
        return int(timestamp)

    return None


def extract_text_content(content: str | list[dict[str, Any]] | Any) -> str:
    """Extract text content from Claude message content (handles both string and list formats).

    Args:
        content: Either a string or list of content parts from Claude API

    Returns:
        Extracted text content, empty string if none found
    """
    if isinstance(content, list):
        text_parts = [
            part.get(CONTENT_TYPE_TEXT, "")
            for part in content
            if isinstance(part, dict) and part.get(MESSAGE_FIELD_TYPE) == CONTENT_TYPE_TEXT
        ]
        return "\n".join(text_parts)
    if isinstance(content, str):
        return content
    return str(content)


def find_last_user_message_index(transcript: list[dict[str, Any]]) -> int | None:
    """Find the index of the last actual user message (ignoring tool results and empty messages).

    Args:
        transcript: List of conversation entries from Claude Code transcript

    Returns:
        Index of last user message, or None if not found
    """
    for i in range(len(transcript) - 1, -1, -1):
        entry = transcript[i]
        if entry.get(MESSAGE_FIELD_TYPE) == MESSAGE_TYPE_USER and not entry.get("toolUseResult"):
            msg = entry.get(MESSAGE_FIELD_MESSAGE, {})
            content = msg.get(MESSAGE_FIELD_CONTENT, "")

            if isinstance(content, list) and len(content) > 0:
                if (
                    isinstance(content[0], dict)
                    and content[0].get(MESSAGE_FIELD_TYPE) == CONTENT_TYPE_TOOL_RESULT
                ):
                    continue

            if isinstance(content, str) and "<local-command-stdout>" in content:
                continue

            if not content or (isinstance(content, str) and content.strip() == ""):
                continue

            return i
    return None


# ============================================================================
# TRANSCRIPT PROCESSING HELPERS
# ============================================================================


def _get_next_timestamp_ns(transcript: list[dict[str, Any]], current_idx: int) -> int | None:
    """Get the timestamp of the next entry for duration calculation."""
    for i in range(current_idx + 1, len(transcript)):
        if timestamp := transcript[i].get(MESSAGE_FIELD_TIMESTAMP):
            return parse_timestamp_to_ns(timestamp)
    return None


def _extract_content_and_tools(content: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Extract text content and tool uses from assistant response content."""
    text_content = ""
    tool_uses = []

    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if part.get(MESSAGE_FIELD_TYPE) == CONTENT_TYPE_TEXT:
                    text_content += part.get(CONTENT_TYPE_TEXT, "")
                elif part.get(MESSAGE_FIELD_TYPE) == CONTENT_TYPE_TOOL_USE:
                    tool_uses.append(part)

    return text_content, tool_uses


def _find_tool_results(transcript: list[dict[str, Any]], start_idx: int) -> dict[str, Any]:
    """Find tool results following the current assistant response.

    Returns a mapping from tool_use_id to tool result content.
    """
    tool_results = {}

    # Look for tool results in subsequent entries
    for i in range(start_idx + 1, len(transcript)):
        entry = transcript[i]
        if entry.get(MESSAGE_FIELD_TYPE) != MESSAGE_TYPE_USER:
            continue

        msg = entry.get(MESSAGE_FIELD_MESSAGE, {})
        content = msg.get(MESSAGE_FIELD_CONTENT, [])

        if isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get(MESSAGE_FIELD_TYPE) == CONTENT_TYPE_TOOL_RESULT
                ):
                    tool_use_id = part.get("tool_use_id")
                    result_content = part.get("content", "")
                    if tool_use_id:
                        tool_results[tool_use_id] = result_content

        # Stop looking once we hit the next assistant response
        if entry.get(MESSAGE_FIELD_TYPE) == MESSAGE_TYPE_ASSISTANT:
            break

    return tool_results


def _get_input_messages(transcript: list[dict[str, Any]], current_idx: int) -> list[dict[str, Any]]:
    """Get all messages between the previous text-bearing assistant response and the current one.

    Claude Code emits separate transcript entries for text and tool_use content.
    A typical sequence looks like:
        assistant [text]        ← previous LLM boundary (stop here)
        assistant [tool_use]    ← include
        user [tool_result]      ← include
        assistant [tool_use]    ← include
        user [tool_result]      ← include
        assistant [text]        ← current (the span we're building inputs for)

    We walk backward and collect everything, only stopping when we hit an
    assistant entry that contains text content (which marks the previous LLM span).

    Args:
        transcript: List of conversation entries from Claude Code transcript
        current_idx: Index of the current assistant response

    Returns:
        List of messages in Anthropic format
    """
    messages = []
    for i in range(current_idx - 1, -1, -1):
        entry = transcript[i]
        msg = entry.get(MESSAGE_FIELD_MESSAGE, {})

        # Stop at a previous assistant entry that has text content (previous LLM span)
        if entry.get(MESSAGE_FIELD_TYPE) == MESSAGE_TYPE_ASSISTANT:
            content = msg.get(MESSAGE_FIELD_CONTENT, [])
            has_text = False
            if isinstance(content, str):
                has_text = bool(content.strip())
            elif isinstance(content, list):
                has_text = any(
                    isinstance(p, dict) and p.get(MESSAGE_FIELD_TYPE) == CONTENT_TYPE_TEXT
                    for p in content
                )
            if has_text:
                break

        if msg.get("role") and msg.get(MESSAGE_FIELD_CONTENT):
            messages.append(msg)
    messages.reverse()
    return messages


def _set_token_usage_attribute(span, usage: dict[str, Any]) -> None:
    """Set token usage on a span using the standardized CHAT_USAGE attribute.

    Args:
        span: The MLflow span to set token usage on
        usage: Dictionary containing token usage info from Claude Code transcript
    """
    if not usage:
        return

    # Include cache_creation_input_tokens (similar cost to input tokens) but not
    # cache_read_input_tokens (much cheaper, would inflate cost estimates)
    input_tokens = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    usage_dict = {
        TokenUsageKey.INPUT_TOKENS: input_tokens,
        TokenUsageKey.OUTPUT_TOKENS: output_tokens,
        TokenUsageKey.TOTAL_TOKENS: input_tokens + output_tokens,
    }

    span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)


def _create_llm_and_tool_spans(
    parent_span, transcript: list[dict[str, Any]], start_idx: int
) -> None:
    """Create LLM and tool spans for assistant responses with proper timing."""
    for i in range(start_idx, len(transcript)):
        entry = transcript[i]
        if entry.get(MESSAGE_FIELD_TYPE) != MESSAGE_TYPE_ASSISTANT:
            continue

        timestamp_ns = parse_timestamp_to_ns(entry.get(MESSAGE_FIELD_TIMESTAMP))

        # Calculate duration based on next timestamp or use default
        if next_timestamp_ns := _get_next_timestamp_ns(transcript, i):
            duration_ns = next_timestamp_ns - timestamp_ns
        else:
            duration_ns = int(1000 * NANOSECONDS_PER_MS)  # 1 second default

        msg = entry.get(MESSAGE_FIELD_MESSAGE, {})
        content = msg.get(MESSAGE_FIELD_CONTENT, [])
        usage = msg.get("usage", {})

        # First check if we have meaningful content to create a span for
        text_content, tool_uses = _extract_content_and_tools(content)

        # Only create LLM span if there's text content (no tools)
        llm_span = None
        if text_content and text_content.strip() and not tool_uses:
            messages = _get_input_messages(transcript, i)

            llm_span = mlflow.start_span_no_context(
                name="llm",
                parent_span=parent_span,
                span_type=SpanType.LLM,
                start_time_ns=timestamp_ns,
                inputs={
                    "model": msg.get("model", "unknown"),
                    "messages": messages,
                },
                attributes={
                    "model": msg.get("model", "unknown"),
                    SpanAttributeKey.MESSAGE_FORMAT: "anthropic",
                },
            )

            # Set token usage using the standardized CHAT_USAGE attribute
            _set_token_usage_attribute(llm_span, usage)

            # Output in Anthropic response format for Chat UI rendering
            llm_span.set_outputs(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": content,
                }
            )
            llm_span.end(end_time_ns=timestamp_ns + duration_ns)

        # Create tool spans with proportional timing and actual results
        if tool_uses:
            tool_results = _find_tool_results(transcript, i)
            tool_duration_ns = duration_ns // len(tool_uses)

            for idx, tool_use in enumerate(tool_uses):
                tool_start_ns = timestamp_ns + (idx * tool_duration_ns)
                tool_use_id = tool_use.get("id", "")
                tool_result = tool_results.get(tool_use_id, "No result found")

                tool_span = mlflow.start_span_no_context(
                    name=f"tool_{tool_use.get('name', 'unknown')}",
                    parent_span=parent_span,
                    span_type=SpanType.TOOL,
                    start_time_ns=tool_start_ns,
                    inputs=tool_use.get("input", {}),
                    attributes={
                        "tool_name": tool_use.get("name", "unknown"),
                        "tool_id": tool_use_id,
                    },
                )

                tool_span.set_outputs({"result": tool_result})
                tool_span.end(end_time_ns=tool_start_ns + tool_duration_ns)


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
                TraceMetadataKey.TRACE_USER: os.environ.get("USER", ""),
                "mlflow.trace.working_directory": os.getcwd(),
            }
            if session_id:
                metadata[TraceMetadataKey.TRACE_SESSION] = session_id

            # Set token usage directly on trace metadata so it survives
            # even if span-level aggregation doesn't pick it up
            if usage:
                input_tokens = usage.get("input_tokens", 0) + usage.get(
                    "cache_creation_input_tokens", 0
                )
                output_tokens = usage.get("output_tokens", 0)
                metadata[TraceMetadataKey.TOKEN_USAGE] = json.dumps(
                    {
                        TokenUsageKey.INPUT_TOKENS: input_tokens,
                        TokenUsageKey.OUTPUT_TOKENS: output_tokens,
                        TokenUsageKey.TOTAL_TOKENS: input_tokens + output_tokens,
                    }
                )

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


def find_final_assistant_response(transcript: list[dict[str, Any]], start_idx: int) -> str | None:
    """Find the final text response from the assistant for trace preview.

    Args:
        transcript: List of conversation entries from Claude Code transcript
        start_idx: Index to start searching from (typically after last user message)

    Returns:
        Final assistant response text or None
    """
    final_response = None

    for i in range(start_idx, len(transcript)):
        entry = transcript[i]
        if entry.get(MESSAGE_FIELD_TYPE) != MESSAGE_TYPE_ASSISTANT:
            continue

        msg = entry.get(MESSAGE_FIELD_MESSAGE, {})
        content = msg.get(MESSAGE_FIELD_CONTENT, [])

        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get(MESSAGE_FIELD_TYPE) == CONTENT_TYPE_TEXT:
                    text = part.get(CONTENT_TYPE_TEXT, "")
                    if text.strip():
                        final_response = text

    return final_response


# ============================================================================
# MAIN TRANSCRIPT PROCESSING
# ============================================================================


def process_transcript(
    transcript_path: str, session_id: str | None = None
) -> mlflow.entities.Trace | None:
    """Process a Claude conversation transcript and create an MLflow trace with spans.

    Args:
        transcript_path: Path to the Claude Code transcript.jsonl file
        session_id: Optional session identifier, defaults to timestamp-based ID

    Returns:
        MLflow trace object if successful, None if processing fails
    """
    try:
        transcript = read_transcript(transcript_path)
        if not transcript:
            get_logger().warning("Empty transcript, skipping")
            return None

        last_user_idx = find_last_user_message_index(transcript)
        if last_user_idx is None:
            get_logger().warning("No user message found in transcript")
            return None

        last_user_entry = transcript[last_user_idx]
        last_user_prompt = last_user_entry.get(MESSAGE_FIELD_MESSAGE, {}).get(
            MESSAGE_FIELD_CONTENT, ""
        )

        if not session_id:
            session_id = f"claude-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        get_logger().log(CLAUDE_TRACING_LEVEL, "Creating MLflow trace for session: %s", session_id)

        conv_start_ns = parse_timestamp_to_ns(last_user_entry.get(MESSAGE_FIELD_TIMESTAMP))

        parent_span = mlflow.start_span_no_context(
            name="claude_code_conversation",
            inputs={"prompt": extract_text_content(last_user_prompt)},
            start_time_ns=conv_start_ns,
            span_type=SpanType.AGENT,
        )

        # Create spans for all assistant responses and tool uses
        _create_llm_and_tool_spans(parent_span, transcript, last_user_idx + 1)

        # Update trace with preview content and end timing
        final_response = find_final_assistant_response(transcript, last_user_idx + 1)
        user_prompt_text = extract_text_content(last_user_prompt)

        # Calculate end time based on last entry or use default duration
        last_entry = transcript[-1] if transcript else last_user_entry
        conv_end_ns = parse_timestamp_to_ns(last_entry.get(MESSAGE_FIELD_TIMESTAMP))
        if not conv_end_ns or conv_end_ns <= conv_start_ns:
            conv_end_ns = conv_start_ns + int(10 * NANOSECONDS_PER_S)

        return _finalize_trace(
            parent_span,
            user_prompt_text,
            final_response,
            session_id,
            conv_end_ns,
        )

    except Exception as e:
        get_logger().error("Error processing transcript: %s", e, exc_info=True)
        return None


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
                llm_span.set_outputs(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": block.text} for block in text_blocks],
                    }
                )
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
