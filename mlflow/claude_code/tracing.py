"""MLflow tracing integration for Claude Code interactions."""

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
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracking.fluent import _get_trace_exporter

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
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get(MESSAGE_FIELD_TYPE) == CONTENT_TYPE_TEXT:
                text_parts.append(part.get(CONTENT_TYPE_TEXT, ""))
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
        timestamp = transcript[i].get(MESSAGE_FIELD_TIMESTAMP)
        if timestamp:
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


def _reconstruct_conversation_messages(
    transcript: list[dict[str, Any]], end_idx: int
) -> list[dict[str, Any]]:
    """Reconstruct conversation messages in OpenAI format for LLM span inputs.

    This function builds the message array that represents what was sent to the LLM.
    It processes the transcript up to (but not including) end_idx to build the context.

    Args:
        transcript: List of conversation entries from Claude Code transcript
        end_idx: Index to stop at (exclusive) - typically the current assistant response

    Returns:
        List of messages in format [{"role": "system"|"user"|"assistant"|"tool", "content": "..."}]
    """
    messages = []

    for i in range(end_idx):
        entry = transcript[i]
        entry_type = entry.get(MESSAGE_FIELD_TYPE)
        msg = entry.get(MESSAGE_FIELD_MESSAGE, {})

        # Check for system role explicitly
        if msg.get("role") == "system":
            _process_system_entry(msg, messages)
        elif entry_type == MESSAGE_TYPE_USER:
            _process_user_entry(msg, messages)
        elif entry_type == MESSAGE_TYPE_ASSISTANT:
            _process_assistant_entry(msg, messages)

    return messages


def _process_system_entry(msg: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    """Process a system entry from the transcript.

    Args:
        msg: The message object from the entry
        messages: The messages list to append to
    """
    if content := msg.get(MESSAGE_FIELD_CONTENT):
        text_content = extract_text_content(content)
        if text_content.strip():
            messages.append({"role": "system", "content": text_content})


def _process_user_entry(msg: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    """Process a user entry from the transcript and add appropriate messages.

    User entries can contain:
    - Regular user messages (text)
    - Tool results from previous tool calls

    Args:
        msg: The message object from the entry
        messages: The messages list to append to
    """
    content = msg.get(MESSAGE_FIELD_CONTENT, [])

    # Handle list content (typical structure)
    if isinstance(content, list):
        # Use a buffer to preserve original message ordering
        message_buffer = []
        current_text_parts = []

        for part in content:
            if not isinstance(part, dict):
                continue

            part_type = part.get(MESSAGE_FIELD_TYPE)

            if part_type == CONTENT_TYPE_TOOL_RESULT:
                # If we have accumulated text, add it as a user message first
                if current_text_parts:
                    combined_text = "\n".join(current_text_parts).strip()
                    if combined_text:
                        message_buffer.append({"role": "user", "content": combined_text})
                    current_text_parts = []

                # Extract tool result information
                tool_id = part.get("tool_use_id")
                result_content = part.get("content")

                # Add tool results with proper "tool" role
                if result_content:
                    tool_msg = {
                        "role": "tool",
                        "content": result_content,
                    }
                    if tool_id:
                        tool_msg["tool_use_id"] = tool_id
                    message_buffer.append(tool_msg)

            elif part_type == CONTENT_TYPE_TEXT:
                # Accumulate text content
                text = part.get(CONTENT_TYPE_TEXT)
                if text:
                    current_text_parts.append(text)

        # Add any remaining text content as user message
        if current_text_parts:
            combined_text = "\n".join(current_text_parts).strip()
            if combined_text:
                message_buffer.append({"role": "user", "content": combined_text})

        # Add all messages in order to preserve sequence
        messages.extend(message_buffer)

    # Handle string content (simpler format)
    elif isinstance(content, str) and content.strip():
        messages.append({"role": "user", "content": content})


def _process_assistant_entry(msg: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    """Process an assistant entry from the transcript and add to messages.

    Assistant entries represent previous LLM responses that are part of the conversation context.

    Args:
        msg: The message object from the entry
        messages: The messages list to append to
    """
    if content := msg.get(MESSAGE_FIELD_CONTENT):
        text_content = extract_text_content(content)
        if text_content.strip():
            messages.append({"role": "assistant", "content": text_content})


def _create_llm_and_tool_spans(
    parent_span, transcript: list[dict[str, Any]], start_idx: int
) -> None:
    """Create LLM and tool spans for assistant responses with proper timing."""
    llm_call_num = 0
    for i in range(start_idx, len(transcript)):
        entry = transcript[i]
        if entry.get(MESSAGE_FIELD_TYPE) != MESSAGE_TYPE_ASSISTANT:
            continue

        timestamp_ns = parse_timestamp_to_ns(entry.get(MESSAGE_FIELD_TIMESTAMP))
        next_timestamp_ns = _get_next_timestamp_ns(transcript, i)

        # Calculate duration based on next timestamp or use default
        if next_timestamp_ns:
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
            llm_call_num += 1
            conversation_messages = _reconstruct_conversation_messages(transcript, i)

            llm_span = mlflow.start_span_no_context(
                name=f"llm_call_{llm_call_num}",
                parent_span=parent_span,
                span_type=SpanType.LLM,
                start_time_ns=timestamp_ns,
                inputs={
                    "model": msg.get("model", "unknown"),
                    "messages": conversation_messages,
                },
                attributes={
                    "model": msg.get("model", "unknown"),
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                },
            )

            llm_span.set_outputs({"response": text_content})
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

        # Set trace previews for UI display
        try:
            with InMemoryTraceManager.get_instance().get_trace(
                parent_span.trace_id
            ) as in_memory_trace:
                if user_prompt_text:
                    in_memory_trace.info.request_preview = user_prompt_text[:MAX_PREVIEW_LENGTH]
                if final_response:
                    in_memory_trace.info.response_preview = final_response[:MAX_PREVIEW_LENGTH]
                in_memory_trace.info.trace_metadata = {
                    **in_memory_trace.info.trace_metadata,
                    TraceMetadataKey.TRACE_SESSION: session_id,
                    TraceMetadataKey.TRACE_USER: os.environ.get("USER", ""),
                    "mlflow.trace.working_directory": os.getcwd(),
                }
        except Exception as e:
            get_logger().warning("Failed to update trace metadata and previews: %s", e)

        # Calculate end time based on last entry or use default duration
        last_entry = transcript[-1] if transcript else last_user_entry
        conv_end_ns = parse_timestamp_to_ns(last_entry.get(MESSAGE_FIELD_TIMESTAMP))
        if not conv_end_ns or conv_end_ns <= conv_start_ns:
            conv_end_ns = conv_start_ns + int(10 * NANOSECONDS_PER_S)  # 10 second default

        parent_span.set_outputs(
            {"response": final_response or "Conversation completed", "status": "completed"}
        )
        parent_span.end(end_time_ns=conv_end_ns)

        try:
            # Use this to check if async trace logging is enabled
            if hasattr(_get_trace_exporter(), "_async_queue"):
                mlflow.flush_trace_async_logging()
        except Exception as e:
            # This is not a critical error, so we log it as debug
            get_logger().debug("Failed to flush trace async logging: %s", e)

        get_logger().log(CLAUDE_TRACING_LEVEL, "Created MLflow trace: %s", parent_span.trace_id)

        return mlflow.get_trace(parent_span.trace_id)

    except Exception as e:
        get_logger().error("Error processing transcript: %s", e, exc_info=True)
        return None
