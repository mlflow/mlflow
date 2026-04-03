"""MLflow tracing integration for Codex CLI interactions.

Codex CLI transcripts are JSONL "rollout" files where each line is a RolloutLine:
    {"timestamp": "...", "type": "<variant>", "payload": {...}}

Defined in codex-rs/protocol/src/protocol.rs (tagged enum ``RolloutItem``).
Stored at ~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<session_id>.jsonl.

Record types (``RolloutItem`` variants):
- ``session_meta``: session identity, model, cwd, git info (first line)
- ``response_item``: API response items (messages, function_call, function_call_output, reasoning)
- ``event_msg``: state transitions (task_started, task_complete, token_count, exec_command_end)
- ``turn_context``: per-turn metadata (model, sandbox policy, approval policy)
- ``compacted``: context-window compaction checkpoint

Turns are delimited by event_msg task_started / task_complete pairs. Within a turn:
- ``response_item`` type=message role=user → user input (content[].type=input_text)
- ``response_item`` type=message role=assistant → assistant text (content[].type=output_text)
- ``response_item`` type=function_call → tool invocation (name, call_id, arguments)
- ``response_item`` type=function_call_output → tool result (call_id, output)
- ``event_msg`` type=token_count → token usage (info.last_token_usage)

References:
- Protocol types: github.com/openai/codex codex-rs/protocol/src/protocol.rs
- Rollout recorder: github.com/openai/codex codex-rs/rollout/src/recorder.rs
- Hook payloads: github.com/openai/codex codex-rs/tui/src/hooks.rs
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


# ---------------------------------------------------------------------------
# Transcript parsing helpers
# ---------------------------------------------------------------------------


def _find_last_user_prompt(records: list[dict[str, Any]]) -> tuple[str, int] | None:
    """Find the last user prompt text and its record index.

    User prompts are ``response_item`` records with ``payload.type=message``
    and ``payload.role=user`` whose content contains ``input_text`` blocks
    that are not system/developer injections.
    """
    for i in range(len(records) - 1, -1, -1):
        record = records[i]
        if record.get("type") != "response_item":
            continue
        payload = record.get("payload", {})
        if payload.get("type") != "message" or payload.get("role") != "user":
            continue

        text = _extract_text_from_content(payload.get("content", []))
        # Skip system/developer context injections (start with XML-like tags)
        if text and not text.startswith("<"):
            return text, i
    return None


def _extract_text_from_content(content: list[dict[str, Any]] | str) -> str:
    """Extract text from a response_item content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") in ("input_text", "output_text"):
                    parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _get_last_turn_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract the records belonging to the last turn.

    Turns are delimited by event_msg records with type=task_started and
    type=task_complete.
    """
    last_start = None
    last_end = None

    for i, record in enumerate(records):
        if record.get("type") != "event_msg":
            continue
        payload = record.get("payload", {})
        if payload.get("type") == "task_started":
            last_start = i
        elif payload.get("type") == "task_complete":
            last_end = i

    if last_start is not None:
        end = (last_end or len(records) - 1) + 1
        return records[last_start:end]
    return records


def _get_token_usage_from_records(records: list[dict[str, Any]]) -> dict[str, int]:
    """Extract cumulative token usage from the last token_count event in a turn."""
    usage: dict[str, int] = {}
    for record in records:
        if record.get("type") != "event_msg":
            continue
        payload = record.get("payload", {})
        if payload.get("type") != "token_count":
            continue
        if info := payload.get("info"):
            if last := info.get("last_token_usage"):
                usage = {
                    "input_tokens": last.get("input_tokens", 0),
                    "output_tokens": last.get("output_tokens", 0),
                    "total_tokens": last.get("total_tokens", 0),
                }
    return usage


def _get_model_from_records(records: list[dict[str, Any]]) -> str:
    """Extract model name from session_meta or turn_context records."""
    for record in records:
        if record.get("type") == "session_meta" or record.get("type") == "turn_context":
            payload = record.get("payload", {})
            if model := payload.get("model"):
                return model
    return "unknown"


def _get_session_id_from_records(records: list[dict[str, Any]]) -> str | None:
    """Extract session ID from the session_meta record."""
    for record in records:
        if record.get("type") == "session_meta":
            return record.get("payload", {}).get("id")
    return None


def _set_token_usage_attribute(span, usage: dict[str, int]) -> None:
    if not usage:
        return

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    span.set_attribute(
        SpanAttributeKey.CHAT_USAGE,
        {
            TokenUsageKey.INPUT_TOKENS: input_tokens,
            TokenUsageKey.OUTPUT_TOKENS: output_tokens,
            TokenUsageKey.TOTAL_TOKENS: total_tokens,
        },
    )


def _create_child_spans(
    parent_span,
    turn_records: list[dict[str, Any]],
    model: str,
) -> str | None:
    """Create LLM and TOOL child spans from a turn's records.

    Returns the final assistant text response for trace preview.
    """
    final_response = None
    # Map call_id → function_call_output payload for tool results
    tool_results: dict[str, str] = {}
    for record in turn_records:
        if record.get("type") != "response_item":
            continue
        payload = record.get("payload", {})
        if payload.get("type") == "function_call_output":
            tool_results[payload.get("call_id", "")] = payload.get("output", "")

    for record in turn_records:
        if record.get("type") != "response_item":
            continue

        payload = record.get("payload", {})
        timestamp_ns = parse_timestamp_to_ns(record.get("timestamp"))

        match payload.get("type"):
            case "message" if payload.get("role") == "assistant":
                text = _extract_text_from_content(payload.get("content", []))
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
                    llm_span.set_outputs({"content": text})
                    llm_span.end(end_time_ns=timestamp_ns)

            case "function_call":
                call_id = payload.get("call_id", "")
                func_name = payload.get("name", "unknown")

                try:
                    arguments = json.loads(payload.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    arguments = payload.get("arguments", {})

                tool_output = tool_results.get(call_id, "")

                tool_span = mlflow.start_span_no_context(
                    name=f"tool_{func_name}",
                    parent_span=parent_span,
                    span_type=SpanType.TOOL,
                    start_time_ns=timestamp_ns,
                    inputs=arguments,
                    attributes={
                        "tool_name": func_name,
                        "tool_id": call_id,
                    },
                )
                tool_span.set_outputs({"result": tool_output})
                tool_span.end(end_time_ns=timestamp_ns)

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
    """Process a Codex CLI conversation transcript and create an MLflow trace.

    Parses the RolloutLine JSONL format, extracts the last turn, and creates
    an AGENT span with LLM and TOOL child spans.
    """
    try:
        if not transcript_path:
            get_logger().warning("No transcript path provided, skipping")
            return None

        records = read_transcript(transcript_path)
        if not records:
            get_logger().warning("Empty transcript, skipping")
            return None

        result = _find_last_user_prompt(records)
        if result is None:
            get_logger().warning("No user message found in transcript")
            return None
        user_prompt, _ = result

        if not session_id:
            session_id = _get_session_id_from_records(records) or (
                f"codex-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        get_logger().log(CODEX_TRACING_LEVEL, "Creating MLflow trace for session: %s", session_id)

        model = _get_model_from_records(records)
        turn_records = _get_last_turn_records(records)

        # Use the turn's first timestamp as start, last as end
        turn_start_ns = parse_timestamp_to_ns(turn_records[0].get("timestamp"))
        turn_end_ns = parse_timestamp_to_ns(turn_records[-1].get("timestamp"))
        if not turn_end_ns or (turn_start_ns and turn_end_ns <= turn_start_ns):
            turn_end_ns = (turn_start_ns or 0) + int(10 * NANOSECONDS_PER_S)

        parent_span = mlflow.start_span_no_context(
            name="codex_conversation",
            inputs={"prompt": user_prompt},
            start_time_ns=turn_start_ns,
            span_type=SpanType.AGENT,
            attributes={"model": model},
        )

        final_response = _create_child_spans(parent_span, turn_records, model)

        # Set token usage on the root span
        token_usage = _get_token_usage_from_records(turn_records)
        _set_token_usage_attribute(parent_span, token_usage)

        return _finalize_trace(
            parent_span,
            user_prompt,
            final_response,
            session_id,
            turn_end_ns,
        )

    except Exception as e:
        get_logger().error("Error processing transcript: %s", e, exc_info=True)
        return None
