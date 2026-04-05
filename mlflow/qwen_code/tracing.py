"""MLflow tracing integration for Qwen Code interactions.

Qwen Code transcripts use tree-structured JSONL ChatRecords:
- Each record has ``uuid`` and ``parentUuid`` forming a parent-child tree
- Records have ``type`` (user/assistant/system), ``message``, ``timestamp``
- Tool results appear in ``toolCallResult`` field
- Token usage is in ``usageMetadata``

This contrasts with Claude Code (flat Anthropic content blocks) and Codex (flat OpenAI tool_calls).
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
from mlflow.qwen_code.config import (
    ENVIRONMENT_FIELD,
    MLFLOW_TRACING_ENABLED,
    load_qwen_config,
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

# ChatRecord type values
RECORD_TYPE_USER = "user"
RECORD_TYPE_ASSISTANT = "assistant"

# Custom logging level for Qwen tracing
QWEN_TRACING_LEVEL = logging.WARNING - 5


def setup_logging() -> logging.Logger:
    log_dir = Path(os.getcwd()) / ".qwen" / "mlflow"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.handlers.clear()

    log_file = log_dir / "qwen_tracing.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logging.addLevelName(QWEN_TRACING_LEVEL, "QWEN_TRACING")
    logger.setLevel(QWEN_TRACING_LEVEL)
    logger.propagate = False

    return logger


_MODULE_LOGGER: logging.Logger | None = None


def get_logger() -> logging.Logger:
    global _MODULE_LOGGER

    if _MODULE_LOGGER is None:
        _MODULE_LOGGER = setup_logging()
    return _MODULE_LOGGER


def get_env_var(var_name: str, default: str = "") -> str:
    """Get environment variable from Qwen settings or OS environment."""
    try:
        settings_path = Path(".qwen/settings.json")
        if settings_path.exists():
            config = load_qwen_config(settings_path)
            env_vars = config.get(ENVIRONMENT_FIELD, {})
            if (value := env_vars.get(var_name)) is not None:
                return value
    except Exception:
        pass

    if (value := os.environ.get(var_name)) is not None:
        return value

    return default


def setup_mlflow() -> None:
    """Configure MLflow tracking URI and experiment."""
    if not is_tracing_enabled():
        return

    if tracking_uri := get_env_var(MLFLOW_TRACKING_URI.name):
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

    _record_event(AutologgingEvent, {"flavor": "qwen_code"})


def is_tracing_enabled() -> bool:
    return get_env_var(MLFLOW_TRACING_ENABLED).lower() in ("true", "1", "yes")


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


def _build_record_tree(
    records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Build a lookup and children map from ChatRecords.

    Returns:
        (records_by_uuid, children_map) where children_map maps uuid → list of child uuids
    """
    records_by_uuid: dict[str, dict[str, Any]] = {}
    children_map: dict[str, list[str]] = {}

    for record in records:
        uuid = record.get("uuid", "")
        records_by_uuid[uuid] = record
        children_map.setdefault(uuid, [])

        if parent_uuid := record.get("parentUuid"):
            children_map.setdefault(parent_uuid, [])
            children_map[parent_uuid].append(uuid)

    return records_by_uuid, children_map


def _find_last_user_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find the last user record in the transcript."""
    for record in reversed(records):
        if record.get("type") == RECORD_TYPE_USER:
            msg = record.get("message", "")
            if isinstance(msg, str) and msg.strip():
                return record
            if isinstance(msg, dict) and msg.get("content", ""):
                return record
    return None


def _get_message_text(record: dict[str, Any]) -> str:
    """Extract text content from a ChatRecord's message field."""
    msg = record.get("message", "")
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict):
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            return "\n".join(parts)
    return str(msg)


def _set_token_usage_attribute(span, usage: dict[str, Any]) -> None:
    if not usage:
        return

    input_tokens = usage.get("promptTokenCount", 0) or usage.get("input_tokens", 0)
    output_tokens = usage.get("candidatesTokenCount", 0) or usage.get("output_tokens", 0)
    total_tokens = usage.get("totalTokenCount", input_tokens + output_tokens)

    usage_dict = {
        TokenUsageKey.INPUT_TOKENS: input_tokens,
        TokenUsageKey.OUTPUT_TOKENS: output_tokens,
        TokenUsageKey.TOTAL_TOKENS: total_tokens,
    }

    span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)


def _create_child_spans(
    parent_span,
    record_uuid: str,
    records_by_uuid: dict[str, dict[str, Any]],
    children_map: dict[str, list[str]],
) -> str | None:
    """Recursively create child spans from the ChatRecord tree.

    Returns the last assistant text response for trace preview.
    """
    final_response = None

    for child_uuid in children_map.get(record_uuid, []):
        child = records_by_uuid.get(child_uuid)
        if not child:
            continue

        record_type = child.get("type")
        timestamp_ns = parse_timestamp_to_ns(child.get("timestamp"))
        model = child.get("model", "unknown")

        # Tool result records → TOOL span
        if child.get("toolCallResult"):
            tool_result = child["toolCallResult"]
            tool_name = tool_result.get("name", "unknown")

            tool_span = mlflow.start_span_no_context(
                name=f"tool_{tool_name}",
                parent_span=parent_span,
                span_type=SpanType.TOOL,
                start_time_ns=timestamp_ns,
                inputs=tool_result.get("input", {}),
                attributes={"tool_name": tool_name},
            )
            tool_span.set_outputs({"result": tool_result.get("output", "")})
            tool_span.end(end_time_ns=timestamp_ns)

        # Assistant text records → LLM span
        elif record_type == RECORD_TYPE_ASSISTANT:
            text = _get_message_text(child)
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

                usage = child.get("usageMetadata", {})
                _set_token_usage_attribute(llm_span, usage)
                llm_span.set_outputs({"content": text})
                llm_span.end(end_time_ns=timestamp_ns)

        # Recurse into children
        if child_text := _create_child_spans(
            parent_span, child_uuid, records_by_uuid, children_map
        ):
            final_response = child_text

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
    get_logger().log(QWEN_TRACING_LEVEL, "Created MLflow trace: %s", parent_span.trace_id)
    return mlflow.get_trace(parent_span.trace_id)


def process_transcript(
    transcript_path: str | None, session_id: str | None = None
) -> mlflow.entities.Trace | None:
    """Process a Qwen Code conversation transcript and create an MLflow trace.

    Qwen Code transcripts are JSONL files where each line is a ChatRecord with
    ``uuid``/``parentUuid`` forming a tree. This function builds the tree and
    walks it to create AGENT→LLM/TOOL span hierarchies.
    """
    try:
        if not transcript_path:
            get_logger().warning("No transcript path provided, skipping")
            return None

        records = read_transcript(transcript_path)
        if not records:
            get_logger().warning("Empty transcript, skipping")
            return None

        user_record = _find_last_user_record(records)
        if user_record is None:
            get_logger().warning("No user message found in transcript")
            return None

        user_prompt = _get_message_text(user_record)
        user_uuid = user_record.get("uuid", "")

        if not session_id:
            session_id = (
                user_record.get("sessionId") or f"qwen-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        get_logger().log(QWEN_TRACING_LEVEL, "Creating MLflow trace for session: %s", session_id)

        records_by_uuid, children_map = _build_record_tree(records)

        conv_start_ns = parse_timestamp_to_ns(user_record.get("timestamp"))

        parent_span = mlflow.start_span_no_context(
            name="qwen_code_conversation",
            inputs={"prompt": user_prompt},
            start_time_ns=conv_start_ns,
            span_type=SpanType.AGENT,
        )

        final_response = _create_child_spans(parent_span, user_uuid, records_by_uuid, children_map)

        last_record = records[-1]
        conv_end_ns = parse_timestamp_to_ns(last_record.get("timestamp"))
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
