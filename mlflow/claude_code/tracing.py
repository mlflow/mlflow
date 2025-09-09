"""MLflow tracing integration for Claude Code interactions."""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import dateutil.parser

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
    logger.setLevel(logging.WARNING)
    logger.propagate = False  # Prevent duplicate log messages

    return logger


_MODULE_LOGGER = setup_logging()


def get_logger() -> logging.Logger:
    """Get the configured module logger."""
    return _MODULE_LOGGER


def setup_mlflow() -> None:
    """Configure MLflow tracking URI and experiment."""
    if not is_tracing_enabled():
        return

    import mlflow
    from mlflow.claude_code.config import get_env_var
    from mlflow.environment_variables import (
        MLFLOW_EXPERIMENT_ID,
        MLFLOW_EXPERIMENT_NAME,
        MLFLOW_TRACKING_URI,
    )

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
    try:
        import mlflow  # noqa: F401
    except ImportError as e:
        get_logger().error("MLflow not available: %s", e)
        return False

    from mlflow.claude_code.config import MLFLOW_TRACING_ENABLED, get_env_var

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
        get_logger().error("Failed to parse input JSON: %s", e)
        return {}


def read_transcript(transcript_path: str) -> list[dict[str, Any]]:
    """Read and parse a Claude Code conversation transcript from JSONL file."""
    try:
        with open(transcript_path, encoding="utf-8") as f:
            lines = f.readlines()
            return [json.loads(line) for line in lines if line.strip()]
    except Exception as e:
        get_logger().error("Failed to read transcript %s: %s", transcript_path, e)
        return []


def output_hook_response(error: str | None = None, **kwargs) -> None:
    """Output hook response JSON to stdout for Claude Code hook protocol."""
    if error is not None:
        response = {"continue": False, "stopReason": error, **kwargs}
    else:
        response = {"continue": True, **kwargs}

    print(json.dumps(response))  # noqa: T201


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


def _create_llm_and_tool_spans(
    client, trace, transcript: list[dict[str, Any]], start_idx: int
) -> None:
    """Create LLM and tool spans for assistant responses with proper timing."""
    from mlflow.entities import SpanType

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
            llm_span = client.start_span(
                name=f"llm_call_{llm_call_num}",
                trace_id=trace.trace_id,
                parent_id=trace.span_id,
                span_type=SpanType.LLM,
                start_time_ns=timestamp_ns,
                inputs={"model": msg.get("model", "unknown")},
                attributes={
                    "model": msg.get("model", "unknown"),
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                },
            )

            client.end_span(
                trace_id=llm_span.trace_id,
                span_id=llm_span.span_id,
                outputs={"response": text_content},
                end_time_ns=timestamp_ns + duration_ns,
            )

        # Create tool spans with proportional timing and actual results
        if tool_uses:
            tool_results = _find_tool_results(transcript, i)
            tool_duration_ns = duration_ns // len(tool_uses)

            for idx, tool_use in enumerate(tool_uses):
                tool_start_ns = timestamp_ns + (idx * tool_duration_ns)
                tool_use_id = tool_use.get("id", "")
                tool_result = tool_results.get(tool_use_id, "No result found")

                tool_span = client.start_span(
                    name=f"tool_{tool_use.get('name', 'unknown')}",
                    trace_id=trace.trace_id,
                    parent_id=trace.span_id,
                    span_type=SpanType.TOOL,
                    start_time_ns=tool_start_ns,
                    inputs=tool_use.get("input", {}),
                    attributes={
                        "tool_name": tool_use.get("name", "unknown"),
                        "tool_id": tool_use_id,
                    },
                )

                client.end_span(
                    trace_id=tool_span.trace_id,
                    span_id=tool_span.span_id,
                    outputs={"result": tool_result},
                    end_time_ns=tool_start_ns + tool_duration_ns,
                )


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


def process_transcript(transcript_path: str, session_id: str | None = None) -> Any | None:
    """Process a Claude conversation transcript and create an MLflow trace with spans.

    Args:
        transcript_path: Path to the Claude Code transcript.jsonl file
        session_id: Optional session identifier, defaults to timestamp-based ID

    Returns:
        MLflow trace object if successful, None if processing fails
    """
    if not is_tracing_enabled():
        get_logger().info("MLflow Claude tracing is disabled")
        return None

    try:
        import mlflow
        from mlflow import MlflowClient
        from mlflow.tracing.constant import TraceMetadataKey
        from mlflow.tracing.trace_manager import InMemoryTraceManager

        setup_mlflow()
        client = MlflowClient()

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

        get_logger().info("Creating MLflow trace for session: %s", session_id)

        conv_start_ns = parse_timestamp_to_ns(last_user_entry.get(MESSAGE_FIELD_TIMESTAMP))

        trace = client.start_trace(
            name="claude_code_conversation",
            inputs={"prompt": extract_text_content(last_user_prompt)},
            start_time_ns=conv_start_ns,
        )

        # Create spans for all assistant responses and tool uses
        _create_llm_and_tool_spans(client, trace, transcript, last_user_idx + 1)

        # Update trace with preview content and end timing
        final_response = find_final_assistant_response(transcript, last_user_idx + 1)
        user_prompt_text = extract_text_content(last_user_prompt)

        # Set trace previews for UI display
        try:
            with InMemoryTraceManager.get_instance().get_trace(trace.trace_id) as in_memory_trace:
                if user_prompt_text:
                    in_memory_trace.info.request_preview = user_prompt_text[:MAX_PREVIEW_LENGTH]
                if final_response:
                    in_memory_trace.info.response_preview = final_response[:MAX_PREVIEW_LENGTH]
                in_memory_trace.info.trace_metadata = {
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

        client.end_trace(
            trace_id=trace.trace_id,
            outputs={"response": final_response or "Conversation completed", "status": "completed"},
            end_time_ns=conv_end_ns,
        )

        mlflow.flush_trace_async_logging()
        get_logger().info("Created MLflow trace: %s", trace.trace_id)

        return mlflow.get_trace(trace.trace_id)

    except Exception as e:
        get_logger().error("Error processing transcript: %s", e, exc_info=True)
        return None
