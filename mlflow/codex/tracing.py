"""MLflow tracing integration for Codex CLI interactions.

Codex CLI transcripts use OpenAI's message format:
- Assistant tool calls use ``tool_calls`` array with ``function.name``/``function.arguments``
- Tool results use ``role: "tool"`` entries with ``tool_call_id``

This contrasts with Claude Code's Anthropic format (content blocks with ``type: tool_use``).
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import dateutil.parser

import mlflow
from mlflow.entities import SpanType
from mlflow.environment_variables import (
    MLFLOW_EXPERIMENT_ID,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from mlflow.telemetry.events import AutologgingEvent
from mlflow.telemetry.track import _record_event
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey, TraceMetadataKey
from mlflow.tracing.provider import _get_trace_exporter
from mlflow.tracing.trace_manager import InMemoryTraceManager

# Constants
NANOSECONDS_PER_MS = 1e6
NANOSECONDS_PER_S = 1e9
MAX_PREVIEW_LENGTH = 1000

# Transcript field names
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"

# Custom logging level for Codex tracing
CODEX_TRACING_LEVEL = logging.WARNING - 5


def setup_logging() -> logging.Logger:
    log_dir = Path(os.getcwd()) / ".codex" / "mlflow"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.handlers.clear()

    log_file = log_dir / "codex_tracing.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logging.addLevelName(CODEX_TRACING_LEVEL, "CODEX_TRACING")
    logger.setLevel(CODEX_TRACING_LEVEL)
    logger.propagate = False

    return logger


_MODULE_LOGGER: logging.Logger | None = None


def get_logger() -> logging.Logger:
    global _MODULE_LOGGER

    if _MODULE_LOGGER is None:
        _MODULE_LOGGER = setup_logging()
    return _MODULE_LOGGER


def setup_mlflow() -> None:
    """Configure MLflow tracking URI and experiment from environment variables."""
    if tracking_uri := os.environ.get(MLFLOW_TRACKING_URI.name):
        mlflow.set_tracking_uri(tracking_uri)

    experiment_id = os.environ.get(MLFLOW_EXPERIMENT_ID.name)
    experiment_name = os.environ.get(MLFLOW_EXPERIMENT_NAME.name)

    try:
        if experiment_id:
            mlflow.set_experiment(experiment_id=experiment_id)
        elif experiment_name:
            mlflow.set_experiment(experiment_name)
    except Exception as e:
        get_logger().warning("Failed to set experiment: %s", e)

    _record_event(AutologgingEvent, {"flavor": "codex"})


def read_hook_input() -> dict[str, Any]:
    try:
        input_data = sys.stdin.read()
        return json.loads(input_data)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Failed to parse hook input: {e}", input_data, 0) from e


def read_transcript(transcript_path: str) -> list[dict[str, Any]]:
    with open(transcript_path, encoding="utf-8") as f:
        lines = f.readlines()
        return [json.loads(line) for line in lines if line.strip()]


def get_hook_response(error: str | None = None, **kwargs) -> dict[str, Any]:
    if error is not None:
        return {"continue": False, "stopReason": error, **kwargs}
    return {"continue": True, **kwargs}


def parse_timestamp_to_ns(timestamp: str | int | float | None) -> int | None:
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


def _find_last_user_message_index(transcript: list[dict[str, Any]]) -> int | None:
    """Find the index of the last user message in the transcript."""
    for i in range(len(transcript) - 1, -1, -1):
        entry = transcript[i]
        msg = entry.get("message", entry)
        role = entry.get("type", msg.get("role"))
        if role == ROLE_USER:
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                return i
    return None


def _get_user_prompt_text(entry: dict[str, Any]) -> str:
    """Extract user prompt text from a transcript entry."""
    msg = entry.get("message", entry)
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return "\n".join(parts)
    return str(content)


def _set_token_usage_attribute(span, usage: dict[str, Any]) -> None:
    if not usage:
        return

    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    usage_dict = {
        TokenUsageKey.INPUT_TOKENS: input_tokens,
        TokenUsageKey.OUTPUT_TOKENS: output_tokens,
        TokenUsageKey.TOTAL_TOKENS: total_tokens,
    }

    span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)


def _get_next_timestamp_ns(transcript: list[dict[str, Any]], current_idx: int) -> int | None:
    for i in range(current_idx + 1, len(transcript)):
        if timestamp := transcript[i].get("timestamp"):
            return parse_timestamp_to_ns(timestamp)
    return None


def _build_tool_result_map(transcript: list[dict[str, Any]], start_idx: int) -> dict[str, str]:
    """Build a map from tool_call_id to tool result content."""
    results: dict[str, str] = {}
    for i in range(start_idx, len(transcript)):
        entry = transcript[i]
        msg = entry.get("message", entry)
        role = entry.get("type", msg.get("role"))
        if role == ROLE_TOOL:
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            if tool_call_id:
                results[tool_call_id] = content
    return results


def _create_llm_and_tool_spans(
    parent_span, transcript: list[dict[str, Any]], start_idx: int
) -> str | None:
    """Create LLM and TOOL child spans from assistant responses in the transcript.

    Returns the final assistant text response (for trace preview).
    """
    tool_result_map = _build_tool_result_map(transcript, start_idx)
    final_response = None

    for i in range(start_idx, len(transcript)):
        entry = transcript[i]
        msg = entry.get("message", entry)
        role = entry.get("type", msg.get("role"))

        if role != ROLE_ASSISTANT:
            continue

        timestamp_ns = parse_timestamp_to_ns(entry.get("timestamp"))
        if next_ts := _get_next_timestamp_ns(transcript, i):
            duration_ns = next_ts - timestamp_ns
        else:
            duration_ns = int(1000 * NANOSECONDS_PER_MS)

        content = msg.get("content")
        tool_calls = msg.get("tool_calls", [])
        usage = msg.get("usage", {})
        model = msg.get("model", "unknown")

        # Text-only response → LLM span
        if content and not tool_calls:
            text = content if isinstance(content, str) else str(content)
            if text.strip():
                final_response = text

                llm_span = mlflow.start_span_no_context(
                    name="llm",
                    parent_span=parent_span,
                    span_type=SpanType.LLM,
                    start_time_ns=timestamp_ns,
                    inputs={"model": model},
                    attributes={"model": model},
                )
                _set_token_usage_attribute(llm_span, usage)
                llm_span.set_outputs({"content": text})
                llm_span.end(end_time_ns=timestamp_ns + duration_ns)

        # Tool calls → TOOL spans
        if tool_calls:
            tool_duration_ns = duration_ns // len(tool_calls)

            for idx, tool_call in enumerate(tool_calls):
                tool_start_ns = timestamp_ns + (idx * tool_duration_ns)
                func = tool_call.get("function", {})
                tool_call_id = tool_call.get("id", "")

                # Parse arguments from JSON string
                try:
                    arguments = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    arguments = func.get("arguments", {})

                tool_result = tool_result_map.get(tool_call_id, "No result found")

                tool_span = mlflow.start_span_no_context(
                    name=f"tool_{func.get('name', 'unknown')}",
                    parent_span=parent_span,
                    span_type=SpanType.TOOL,
                    start_time_ns=tool_start_ns,
                    inputs=arguments,
                    attributes={
                        "tool_name": func.get("name", "unknown"),
                        "tool_id": tool_call_id,
                    },
                )
                tool_span.set_outputs({"result": tool_result})
                tool_span.end(end_time_ns=tool_start_ns + tool_duration_ns)

    return final_response


def _flush_trace_async_logging() -> None:
    try:
        if hasattr(_get_trace_exporter(), "_async_queue"):
            mlflow.flush_trace_async_logging()
    except Exception as e:
        get_logger().debug("Failed to flush trace async logging: %s", e)


def _finalize_trace(
    parent_span,
    user_prompt: str,
    final_response: str | None,
    session_id: str | None,
    end_time_ns: int | None = None,
) -> mlflow.entities.Trace:
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
    get_logger().log(CODEX_TRACING_LEVEL, "Created MLflow trace: %s", parent_span.trace_id)
    return mlflow.get_trace(parent_span.trace_id)


def process_transcript(
    transcript_path: str | None, session_id: str | None = None
) -> mlflow.entities.Trace | None:
    """Process a Codex conversation transcript and create an MLflow trace.

    Args:
        transcript_path: Path to the Codex transcript JSONL file
        session_id: Optional session identifier

    Returns:
        MLflow trace object if successful, None otherwise
    """
    try:
        if not transcript_path:
            get_logger().warning("No transcript path provided, skipping")
            return None

        transcript = read_transcript(transcript_path)
        if not transcript:
            get_logger().warning("Empty transcript, skipping")
            return None

        last_user_idx = _find_last_user_message_index(transcript)
        if last_user_idx is None:
            get_logger().warning("No user message found in transcript")
            return None

        user_prompt = _get_user_prompt_text(transcript[last_user_idx])
        if not session_id:
            session_id = f"codex-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        get_logger().log(CODEX_TRACING_LEVEL, "Creating MLflow trace for session: %s", session_id)

        conv_start_ns = parse_timestamp_to_ns(transcript[last_user_idx].get("timestamp"))

        parent_span = mlflow.start_span_no_context(
            name="codex_conversation",
            inputs={"prompt": user_prompt},
            start_time_ns=conv_start_ns,
            span_type=SpanType.AGENT,
        )

        final_response = _create_llm_and_tool_spans(parent_span, transcript, last_user_idx + 1)

        last_entry = transcript[-1]
        conv_end_ns = parse_timestamp_to_ns(last_entry.get("timestamp"))
        if not conv_end_ns or (conv_start_ns and conv_end_ns <= conv_start_ns):
            conv_end_ns = (conv_start_ns or 0) + int(10 * NANOSECONDS_PER_S)

        return _finalize_trace(
            parent_span,
            user_prompt,
            final_response,
            session_id,
            conv_end_ns,
        )

    except Exception as e:
        get_logger().error("Error processing transcript: %s", e, exc_info=True)
        return None
