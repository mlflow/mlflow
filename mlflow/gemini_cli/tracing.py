"""MLflow tracing integration for Gemini CLI interactions."""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.environment_variables import (
    MLFLOW_EXPERIMENT_ID,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from mlflow.gemini_cli.config import MLFLOW_TRACING_ENABLED, get_env_var
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey, TraceMetadataKey
from mlflow.tracing.provider import _get_trace_exporter
from mlflow.tracing.trace_manager import InMemoryTraceManager

# ============================================================================
# CONSTANTS
# ============================================================================

NANOSECONDS_PER_MS = 1e6
NANOSECONDS_PER_S = 1e9
MAX_PREVIEW_LENGTH = 1000

# Gemini CLI message types
MESSAGE_TYPE_USER = "user"
MESSAGE_TYPE_GEMINI = "gemini"
MESSAGE_TYPE_SESSION_METADATA = "session_metadata"
MESSAGE_TYPE_MESSAGE_UPDATE = "message_update"

# Content field names
FIELD_TYPE = "type"
FIELD_CONTENT = "content"
FIELD_ID = "id"
FIELD_TIMESTAMP = "timestamp"
FIELD_TOKENS = "tokens"
FIELD_TOOL_CALL = "functionCall"
FIELD_TOOL_RESPONSE = "functionResponse"

# Custom logging level for Gemini CLI tracing
GEMINI_TRACING_LEVEL = logging.WARNING - 5


# ============================================================================
# LOGGING AND SETUP
# ============================================================================


def setup_logging() -> logging.Logger:
    """Set up logging directory and return configured logger.

    Creates .gemini/mlflow directory structure and configures file-based logging.
    """
    log_dir = Path(os.getcwd()) / ".gemini" / "mlflow"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.handlers.clear()

    log_file = log_dir / "gemini_tracing.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logging.addLevelName(GEMINI_TRACING_LEVEL, "GEMINI_TRACING")
    logger.setLevel(GEMINI_TRACING_LEVEL)
    logger.propagate = False

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

    tracking_uri = get_env_var(MLFLOW_TRACKING_URI.name)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

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
    """Check if MLflow Gemini CLI tracing is enabled via environment variable."""
    return get_env_var(MLFLOW_TRACING_ENABLED).lower() in ("true", "1", "yes")


# ============================================================================
# INPUT/OUTPUT UTILITIES
# ============================================================================


def read_hook_input() -> dict[str, Any]:
    """Read JSON input from stdin for Gemini CLI hook processing."""
    try:
        input_data = sys.stdin.read()
        return json.loads(input_data)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Failed to parse hook input: {e}", input_data, 0) from e


def read_transcript(transcript_path: str) -> list[dict[str, Any]]:
    """Read and parse a Gemini CLI conversation transcript from JSONL file.

    Args:
        transcript_path: Path to the transcript JSONL file

    Returns:
        List of parsed JSON entries from the transcript
    """
    with open(transcript_path, encoding="utf-8") as f:
        lines = f.readlines()
        return [json.loads(line) for line in lines if line.strip()]


def get_hook_response(error: str | None = None, **kwargs) -> dict[str, Any]:
    """Build hook response dictionary for Gemini CLI hook protocol.

    Args:
        error: Error message if hook failed, None if successful
        kwargs: Additional fields to include in response

    Returns:
        Hook response dictionary compatible with Gemini CLI hook protocol
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
            import dateutil.parser

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
    """Extract text content from Gemini message content.

    Gemini CLI messages have content as a list of parts, each with a "text" field.

    Args:
        content: Either a string or list of content parts from Gemini API

    Returns:
        Extracted text content, empty string if none found
    """
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and "text" in part
        ]
        return "\n".join(text_parts)
    if isinstance(content, str):
        return content
    return str(content) if content else ""


def find_last_user_message_index(transcript: list[dict[str, Any]]) -> int | None:
    """Find the index of the last user message in the transcript.

    Args:
        transcript: List of conversation entries from Gemini CLI transcript

    Returns:
        Index of last user message, or None if not found
    """
    for i in range(len(transcript) - 1, -1, -1):
        entry = transcript[i]
        if entry.get(FIELD_TYPE) == MESSAGE_TYPE_USER:
            content = entry.get(FIELD_CONTENT, [])
            text = extract_text_content(content)
            if text and text.strip():
                return i
    return None


# ============================================================================
# TRANSCRIPT PROCESSING HELPERS
# ============================================================================


def _get_next_timestamp_ns(transcript: list[dict[str, Any]], current_idx: int) -> int | None:
    """Get the timestamp of the next entry for duration calculation."""
    for i in range(current_idx + 1, len(transcript)):
        if timestamp := transcript[i].get(FIELD_TIMESTAMP):
            return parse_timestamp_to_ns(timestamp)
    return None


def _extract_content_and_tools(
    content: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Extract text content and tool calls from Gemini response content.

    Args:
        content: List of content parts from Gemini response

    Returns:
        Tuple of (text_content, tool_calls)
    """
    text_content = ""
    tool_calls = []

    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    text_content += part.get("text", "")
                elif FIELD_TOOL_CALL in part:
                    tool_calls.append(part[FIELD_TOOL_CALL])

    return text_content, tool_calls


def _find_tool_results(
    transcript: list[dict[str, Any]], start_idx: int
) -> dict[str, Any]:
    """Find tool results following the current Gemini response.

    Returns a mapping from function name to tool result content.
    """
    tool_results: dict[str, Any] = {}

    for i in range(start_idx + 1, len(transcript)):
        entry = transcript[i]
        if entry.get(FIELD_TYPE) == MESSAGE_TYPE_GEMINI:
            break

        content = entry.get(FIELD_CONTENT, [])
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and FIELD_TOOL_RESPONSE in part:
                    func_response = part[FIELD_TOOL_RESPONSE]
                    func_name = func_response.get("name", "")
                    func_result = func_response.get("response", {})
                    if func_name:
                        tool_results[func_name] = func_result

    return tool_results


def _get_token_usage(transcript: list[dict[str, Any]], msg_id: str) -> dict[str, Any]:
    """Get token usage for a specific message from message_update entries.

    Args:
        transcript: Full transcript
        msg_id: Message ID to find usage for

    Returns:
        Token usage dictionary with input_tokens, output_tokens, total_tokens
    """
    for entry in transcript:
        if (
            entry.get(FIELD_TYPE) == MESSAGE_TYPE_MESSAGE_UPDATE
            and entry.get(FIELD_ID) == msg_id
            and FIELD_TOKENS in entry
        ):
            tokens = entry[FIELD_TOKENS]
            input_tokens = tokens.get("input", 0)
            output_tokens = tokens.get("output", 0)
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
    return {}


def _set_token_usage_attribute(span, usage: dict[str, Any]) -> None:
    """Set token usage on a span using the standardized CHAT_USAGE attribute.

    Args:
        span: The MLflow span to set token usage on
        usage: Dictionary containing token usage info
    """
    if not usage:
        return

    input_tokens = usage.get("input_tokens", 0)
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
    """Create LLM and tool spans for Gemini responses with proper timing."""
    for i in range(start_idx, len(transcript)):
        entry = transcript[i]
        if entry.get(FIELD_TYPE) != MESSAGE_TYPE_GEMINI:
            continue

        timestamp_ns = parse_timestamp_to_ns(entry.get(FIELD_TIMESTAMP))
        msg_id = entry.get(FIELD_ID, "")

        # Calculate duration based on next timestamp or use default
        if next_timestamp_ns := _get_next_timestamp_ns(transcript, i):
            duration_ns = next_timestamp_ns - timestamp_ns if timestamp_ns else int(
                1000 * NANOSECONDS_PER_MS
            )
        else:
            duration_ns = int(1000 * NANOSECONDS_PER_MS)  # 1 second default

        content = entry.get(FIELD_CONTENT, [])
        text_content, tool_calls = _extract_content_and_tools(content)

        # Get token usage from message_update entries
        usage = _get_token_usage(transcript, msg_id)

        # Create LLM span if there's text content (no tool calls)
        if text_content and text_content.strip() and not tool_calls:
            llm_span = mlflow.start_span_no_context(
                name="llm",
                parent_span=parent_span,
                span_type=SpanType.LLM,
                start_time_ns=timestamp_ns,
                inputs={
                    "model": entry.get("model", "gemini"),
                    "messages": [{"role": "assistant", "content": text_content}],
                },
                attributes={
                    "model": entry.get("model", "gemini"),
                },
            )

            _set_token_usage_attribute(llm_span, usage)

            llm_span.set_outputs(
                {
                    "role": "assistant",
                    "content": text_content,
                }
            )
            if timestamp_ns:
                llm_span.end(end_time_ns=timestamp_ns + duration_ns)
            else:
                llm_span.end()

        # Create tool spans
        if tool_calls:
            tool_results = _find_tool_results(transcript, i)
            tool_duration_ns = duration_ns // max(len(tool_calls), 1)

            for idx, tool_call in enumerate(tool_calls):
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})
                tool_result = tool_results.get(tool_name, "No result found")

                tool_start_ns = (
                    timestamp_ns + (idx * tool_duration_ns) if timestamp_ns else None
                )

                tool_span = mlflow.start_span_no_context(
                    name=f"tool_{tool_name}",
                    parent_span=parent_span,
                    span_type=SpanType.TOOL,
                    start_time_ns=tool_start_ns,
                    inputs=tool_args,
                    attributes={
                        "tool_name": tool_name,
                    },
                )

                tool_span.set_outputs({"result": tool_result})
                if tool_start_ns:
                    tool_span.end(end_time_ns=tool_start_ns + tool_duration_ns)
                else:
                    tool_span.end()


def _finalize_trace(
    parent_span,
    user_prompt: str,
    final_response: str | None,
    session_id: str | None,
    end_time_ns: int | None = None,
    total_usage: dict[str, Any] | None = None,
) -> mlflow.entities.Trace:
    """Finalize the trace with metadata and end it."""
    try:
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

            if total_usage:
                metadata[TraceMetadataKey.TOKEN_USAGE] = json.dumps(
                    {
                        TokenUsageKey.INPUT_TOKENS: total_usage.get("input_tokens", 0),
                        TokenUsageKey.OUTPUT_TOKENS: total_usage.get("output_tokens", 0),
                        TokenUsageKey.TOTAL_TOKENS: total_usage.get("total_tokens", 0),
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
    get_logger().log(GEMINI_TRACING_LEVEL, "Created MLflow trace: %s", parent_span.trace_id)
    return mlflow.get_trace(parent_span.trace_id)


def _flush_trace_async_logging() -> None:
    """Flush any pending async trace logging."""
    try:
        if hasattr(_get_trace_exporter(), "_async_queue"):
            mlflow.flush_trace_async_logging()
    except Exception as e:
        get_logger().debug("Failed to flush trace async logging: %s", e)


def find_final_gemini_response(transcript: list[dict[str, Any]], start_idx: int) -> str | None:
    """Find the final text response from Gemini for trace preview.

    Args:
        transcript: List of conversation entries
        start_idx: Index to start searching from

    Returns:
        Final Gemini response text or None
    """
    final_response = None

    for i in range(start_idx, len(transcript)):
        entry = transcript[i]
        if entry.get(FIELD_TYPE) != MESSAGE_TYPE_GEMINI:
            continue

        content = entry.get(FIELD_CONTENT, [])
        text = extract_text_content(content)
        if text and text.strip():
            final_response = text

    return final_response


def _aggregate_token_usage(transcript: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate token usage from all message_update entries in the transcript.

    Args:
        transcript: Full transcript

    Returns:
        Aggregated token usage dictionary
    """
    total_input = 0
    total_output = 0

    for entry in transcript:
        if entry.get(FIELD_TYPE) == MESSAGE_TYPE_MESSAGE_UPDATE and FIELD_TOKENS in entry:
            tokens = entry[FIELD_TOKENS]
            total_input += tokens.get("input", 0)
            total_output += tokens.get("output", 0)

    if total_input > 0 or total_output > 0:
        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
        }
    return {}


# ============================================================================
# MAIN TRANSCRIPT PROCESSING
# ============================================================================


def process_transcript(
    transcript_path: str, session_id: str | None = None
) -> mlflow.entities.Trace | None:
    """Process a Gemini CLI conversation transcript and create an MLflow trace with spans.

    Args:
        transcript_path: Path to the Gemini CLI transcript JSONL file
        session_id: Optional session identifier

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
        last_user_prompt = last_user_entry.get(FIELD_CONTENT, "")

        if not session_id:
            # Try to get session ID from session_metadata entry
            for entry in transcript:
                if entry.get(FIELD_TYPE) == MESSAGE_TYPE_SESSION_METADATA:
                    session_id = entry.get("sessionId")
                    break
            if not session_id:
                session_id = f"gemini-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        get_logger().log(
            GEMINI_TRACING_LEVEL, "Creating MLflow trace for session: %s", session_id
        )

        conv_start_ns = parse_timestamp_to_ns(last_user_entry.get(FIELD_TIMESTAMP))

        parent_span = mlflow.start_span_no_context(
            name="gemini_cli_conversation",
            inputs={"prompt": extract_text_content(last_user_prompt)},
            start_time_ns=conv_start_ns,
            span_type=SpanType.AGENT,
        )

        # Create spans for all Gemini responses and tool uses
        _create_llm_and_tool_spans(parent_span, transcript, last_user_idx + 1)

        # Aggregate token usage across all messages
        total_usage = _aggregate_token_usage(transcript)
        if total_usage:
            _set_token_usage_attribute(parent_span, total_usage)

        # Update trace with preview content and end timing
        final_response = find_final_gemini_response(transcript, last_user_idx + 1)
        user_prompt_text = extract_text_content(last_user_prompt)

        # Calculate end time
        last_entry = transcript[-1] if transcript else last_user_entry
        conv_end_ns = parse_timestamp_to_ns(last_entry.get(FIELD_TIMESTAMP))
        if not conv_end_ns or (conv_start_ns and conv_end_ns <= conv_start_ns):
            if conv_start_ns:
                conv_end_ns = conv_start_ns + int(10 * NANOSECONDS_PER_S)
            else:
                conv_end_ns = None

        return _finalize_trace(
            parent_span,
            user_prompt_text,
            final_response,
            session_id,
            conv_end_ns,
            total_usage,
        )

    except Exception as e:
        get_logger().error("Error processing transcript: %s", e, exc_info=True)
        return None
