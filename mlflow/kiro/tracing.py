"""MLflow tracing integration for Kiro CLI agent sessions.

Kiro passes session data via stdin as JSON when the *Agent Stop* hook fires.
The payload contains at minimum:

.. code-block:: json

    {
        "session_id": "abc123",
        "conversation": [
            {
                "role": "user",
                "content": "Help me refactor this function",
                "timestamp": "2024-01-15T10:00:00Z"
            },
            {
                "role": "assistant",
                "content": "Sure! Here's the refactored version...",
                "timestamp": "2024-01-15T10:00:05Z",
                "model": "claude-3-5-sonnet",
                "usage": {
                    "input_tokens": 150,
                    "output_tokens": 300
                },
                "tool_calls": [
                    {
                        "id": "tool_abc",
                        "name": "read_file",
                        "input": {"path": "src/utils.py"},
                        "result": "def helper(): ..."
                    }
                ]
            }
        ],
        "kiro_version": "1.2.3"
    }

When the payload format is minimal or unknown we fall back to logging the raw
session data as a single AGENT span so that *something* is always captured.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.kiro.config import (
    KIRO_ENV_FILE,
    MLFLOW_TRACING_ENABLED,
    get_env_var,
)
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NANOSECONDS_PER_MS = 1e6
NANOSECONDS_PER_S = 1e9
MAX_PREVIEW_LENGTH = 1000

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

METADATA_KEY_KIRO_VERSION = "mlflow.kiro_version"

KIRO_TRACING_LEVEL = logging.WARNING - 5


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    """Set up logging directory and return a configured logger.

    Creates a ``.kiro/mlflow/`` directory structure and attaches a file
    handler.  Log propagation is disabled so messages never reach the root
    logger.
    """
    log_dir = Path(os.getcwd()) / ".kiro" / "mlflow"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.handlers.clear()

    log_file = log_dir / "kiro_tracing.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logging.addLevelName(KIRO_TRACING_LEVEL, "KIRO_TRACING")
    logger.setLevel(KIRO_TRACING_LEVEL)
    logger.propagate = False
    return logger


_MODULE_LOGGER: logging.Logger | None = None


def get_logger() -> logging.Logger:
    """Return the configured module-level logger (lazily initialised)."""
    global _MODULE_LOGGER
    if _MODULE_LOGGER is None:
        _MODULE_LOGGER = setup_logging()
    return _MODULE_LOGGER


# ---------------------------------------------------------------------------
# MLflow bootstrap
# ---------------------------------------------------------------------------


def is_tracing_enabled() -> bool:
    """Return True when Kiro MLflow tracing is enabled via env-var/config."""
    return get_env_var(MLFLOW_TRACING_ENABLED).lower() in ("true", "1", "yes")


def setup_mlflow() -> None:
    """Configure MLflow tracking URI and experiment from env/config."""
    if not is_tracing_enabled():
        return

    mlflow.set_tracking_uri(get_env_var(MLFLOW_TRACKING_URI.name))

    experiment_id = get_env_var(MLFLOW_EXPERIMENT_ID.name)
    experiment_name = get_env_var(MLFLOW_EXPERIMENT_NAME.name)

    try:
        if experiment_id:
            mlflow.set_experiment(experiment_id=experiment_id)
        elif experiment_name:
            mlflow.set_experiment(experiment_name)
    except Exception as exc:
        get_logger().warning("Failed to set experiment: %s", exc)

    _record_event(AutologgingEvent, {"flavor": "kiro"})


# ---------------------------------------------------------------------------
# stdin / hook I/O helpers
# ---------------------------------------------------------------------------


def read_hook_input() -> dict[str, Any]:
    """Read JSON session data from stdin (sent by the Kiro Agent Stop hook)."""
    try:
        raw = sys.stdin.read()
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(f"Failed to parse Kiro hook input: {exc}", raw, 0) from exc


def get_hook_response(error: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Build the JSON hook response expected by Kiro.

    A response with ``"continue": false`` and a ``stopReason`` causes Kiro to
    surface the error message in the IDE.
    """
    if error is not None:
        return {"continue": False, "stopReason": error, **kwargs}
    return {"continue": True, **kwargs}


# ---------------------------------------------------------------------------
# Timestamp utilities
# ---------------------------------------------------------------------------


def parse_timestamp_to_ns(timestamp: str | int | float | None) -> int | None:
    """Convert an ISO-8601 string or numeric Unix timestamp to nanoseconds.

    Args:
        timestamp: ISO string, Unix seconds/milliseconds/nanoseconds, or None.

    Returns:
        Integer nanoseconds since epoch, or None if conversion fails.
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


# ---------------------------------------------------------------------------
# Token-usage utilities
# ---------------------------------------------------------------------------


def _build_usage_dict(usage: dict[str, Any]) -> dict[str, int]:
    """Normalise a Kiro usage payload into the standard MLflow CHAT_USAGE schema."""
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    result: dict[str, int] = {
        TokenUsageKey.INPUT_TOKENS: input_tokens,
        TokenUsageKey.OUTPUT_TOKENS: output_tokens,
        TokenUsageKey.TOTAL_TOKENS: input_tokens + output_tokens,
    }
    if (cached := usage.get("cache_read_input_tokens")) is not None:
        result[TokenUsageKey.CACHE_READ_INPUT_TOKENS] = cached
    if (created := usage.get("cache_creation_input_tokens")) is not None:
        result[TokenUsageKey.CACHE_CREATION_INPUT_TOKENS] = created
    return result


def _set_token_usage(span: Any, usage: dict[str, Any]) -> None:
    """Attach token-usage data to *span* using the standard CHAT_USAGE attribute."""
    if not usage:
        return
    span.set_attribute(SpanAttributeKey.CHAT_USAGE, _build_usage_dict(usage))


# ---------------------------------------------------------------------------
# Conversation-parsing helpers
# ---------------------------------------------------------------------------


def _find_last_user_message(conversation: list[dict[str, Any]]) -> int | None:
    """Return the index of the last genuine user turn in *conversation*."""
    for idx in range(len(conversation) - 1, -1, -1):
        entry = conversation[idx]
        if entry.get("role") == ROLE_USER:
            content = entry.get("content", "")
            if content and (isinstance(content, str) and content.strip()):
                return idx
            if isinstance(content, list) and content:
                return idx
    return None


def _extract_text(content: str | list[Any] | Any) -> str:
    """Flatten content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p)
    return str(content) if content else ""


def _find_final_assistant_response(
    conversation: list[dict[str, Any]], start_idx: int
) -> str | None:
    """Return the text of the last assistant turn after *start_idx*."""
    final: str | None = None
    for entry in conversation[start_idx:]:
        if entry.get("role") == ROLE_ASSISTANT:
            text = _extract_text(entry.get("content", ""))
            if text.strip():
                final = text
    return final


# ---------------------------------------------------------------------------
# Span builders
# ---------------------------------------------------------------------------


def _create_tool_spans(
    parent_span: Any,
    tool_calls: list[dict[str, Any]],
    start_ns: int,
    total_duration_ns: int,
) -> None:
    """Create one TOOL child span per tool call with proportional timing."""
    if not tool_calls:
        return

    per_tool_ns = total_duration_ns // len(tool_calls)
    for idx, call in enumerate(tool_calls):
        tool_start = start_ns + idx * per_tool_ns
        span = mlflow.start_span_no_context(
            name=f"tool_{call.get('name', 'unknown')}",
            parent_span=parent_span,
            span_type=SpanType.TOOL,
            start_time_ns=tool_start,
            inputs=call.get("input", {}),
            attributes={
                "tool_name": call.get("name", "unknown"),
                "tool_id": call.get("id", ""),
            },
        )
        span.set_outputs({"result": call.get("result", "")})
        span.end(end_time_ns=tool_start + per_tool_ns)


def _create_llm_span(
    parent_span: Any,
    entry: dict[str, Any],
    messages: list[dict[str, Any]],
    start_ns: int | None,
    end_ns: int | None,
) -> None:
    """Create one LLM child span for an assistant response that has text."""
    model = entry.get("model", "unknown")
    content = entry.get("content", "")
    text = _extract_text(content)

    span = mlflow.start_span_no_context(
        name="llm",
        parent_span=parent_span,
        span_type=SpanType.LLM,
        start_time_ns=start_ns,
        inputs={"model": model, "messages": messages},
        attributes={"model": model, SpanAttributeKey.MESSAGE_FORMAT: "openai"},
    )

    _set_token_usage(span, entry.get("usage", {}))

    span.set_outputs({
        "role": "assistant",
        "content": text,
    })
    span.end(end_time_ns=end_ns)


def _build_child_spans(
    parent_span: Any,
    conversation: list[dict[str, Any]],
    start_idx: int,
) -> None:
    """Create LLM + TOOL child spans for every assistant turn after *start_idx*."""
    # Build a running history of prior messages for LLM span inputs
    history: list[dict[str, Any]] = [
        {"role": e.get("role", ""), "content": _extract_text(e.get("content", ""))}
        for e in conversation[:start_idx + 1]
    ]

    for i in range(start_idx + 1, len(conversation)):
        entry = conversation[i]
        if entry.get("role") != ROLE_ASSISTANT:
            # Accumulate user messages into history
            history.append({
                "role": entry.get("role", ""),
                "content": _extract_text(entry.get("content", "")),
            })
            continue

        ts_start = parse_timestamp_to_ns(entry.get("timestamp"))
        # Duration: gap to the next entry or default 1 s
        next_ts = None
        for j in range(i + 1, len(conversation)):
            if ts := conversation[j].get("timestamp"):
                next_ts = parse_timestamp_to_ns(ts)
                break
        duration_ns = int(NANOSECONDS_PER_S)  # 1-second default
        if ts_start and next_ts and next_ts > ts_start:
            duration_ns = next_ts - ts_start
        ts_end = (ts_start + duration_ns) if ts_start else None

        tool_calls: list[dict[str, Any]] = entry.get("tool_calls", [])
        content = _extract_text(entry.get("content", ""))

        if content and not tool_calls:
            _create_llm_span(parent_span, entry, list(history), ts_start, ts_end)

        if tool_calls:
            _create_tool_spans(
                parent_span,
                tool_calls,
                ts_start or 0,
                duration_ns,
            )

        # Add this assistant turn to the running history
        history.append({"role": ROLE_ASSISTANT, "content": content})


# ---------------------------------------------------------------------------
# Trace finalisation
# ---------------------------------------------------------------------------


def _flush_async_logging() -> None:
    try:
        if hasattr(_get_trace_exporter(), "_async_queue"):
            mlflow.flush_trace_async_logging()
    except Exception as exc:
        get_logger().debug("Failed to flush async trace logging: %s", exc)


def _finalize_trace(
    parent_span: Any,
    user_prompt: str,
    final_response: str | None,
    session_id: str | None,
    end_time_ns: int | None = None,
    usage: dict[str, Any] | None = None,
    kiro_version: str | None = None,
) -> "mlflow.entities.Trace":
    """Set trace metadata, end the root span, and flush the trace."""
    try:
        with InMemoryTraceManager.get_instance().get_trace(
            parent_span.trace_id
        ) as in_memory_trace:
            if user_prompt:
                in_memory_trace.info.request_preview = user_prompt[:MAX_PREVIEW_LENGTH]
            if final_response:
                in_memory_trace.info.response_preview = final_response[:MAX_PREVIEW_LENGTH]

            metadata: dict[str, str] = {
                TraceMetadataKey.TRACE_USER: os.environ.get("USER", ""),
                "mlflow.trace.working_directory": os.getcwd(),
            }
            if session_id:
                metadata[TraceMetadataKey.TRACE_SESSION] = session_id
            if kiro_version:
                metadata[METADATA_KEY_KIRO_VERSION] = kiro_version
            if usage:
                metadata[TraceMetadataKey.TOKEN_USAGE] = json.dumps(_build_usage_dict(usage))

            in_memory_trace.info.trace_metadata = {
                **in_memory_trace.info.trace_metadata,
                **metadata,
            }
    except Exception as exc:
        get_logger().warning("Failed to update trace metadata: %s", exc)

    outputs: dict[str, Any] = {"status": "completed"}
    if final_response:
        outputs["response"] = final_response
    parent_span.set_outputs(outputs)
    parent_span.end(end_time_ns=end_time_ns)
    _flush_async_logging()
    get_logger().log(KIRO_TRACING_LEVEL, "Created MLflow trace: %s", parent_span.trace_id)
    return mlflow.get_trace(parent_span.trace_id)


# ---------------------------------------------------------------------------
# Main session processor
# ---------------------------------------------------------------------------


def process_session(
    session_data: dict[str, Any],
) -> "mlflow.entities.Trace | None":
    """Process a Kiro session payload and create an MLflow trace.

    The ``session_data`` dict is the JSON object Kiro passes to the Agent Stop
    hook via stdin.  We handle both the full structured format and a minimal
    fallback where the payload may only contain a ``session_id``.

    Args:
        session_data: Parsed JSON hook payload from Kiro.

    Returns:
        An :class:`mlflow.entities.Trace` on success, or ``None`` on failure.
    """
    try:
        session_id: str | None = session_data.get("session_id")
        conversation: list[dict[str, Any]] = session_data.get("conversation", [])
        kiro_version: str | None = session_data.get("kiro_version")

        if not session_id:
            session_id = f"kiro-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        get_logger().log(
            KIRO_TRACING_LEVEL,
            "Creating MLflow trace for Kiro session: %s",
            session_id,
        )

        # ------------------------------------------------------------------
        # Minimal fallback: no conversation data in payload
        # ------------------------------------------------------------------
        if not conversation:
            get_logger().warning(
                "No conversation data in Kiro hook payload for session %s; "
                "creating minimal trace.",
                session_id,
            )
            parent_span = mlflow.start_span_no_context(
                name="kiro_session",
                inputs={"session_id": session_id},
                span_type=SpanType.AGENT,
            )
            return _finalize_trace(
                parent_span,
                user_prompt="",
                final_response=None,
                session_id=session_id,
                kiro_version=kiro_version,
            )

        # ------------------------------------------------------------------
        # Full conversation processing
        # ------------------------------------------------------------------
        last_user_idx = _find_last_user_message(conversation)
        if last_user_idx is None:
            get_logger().warning("No user message found in Kiro session %s", session_id)
            return None

        last_user_entry = conversation[last_user_idx]
        user_prompt = _extract_text(last_user_entry.get("content", ""))

        conv_start_ns = parse_timestamp_to_ns(last_user_entry.get("timestamp"))

        parent_span = mlflow.start_span_no_context(
            name="kiro_session",
            inputs={"prompt": user_prompt},
            span_type=SpanType.AGENT,
            start_time_ns=conv_start_ns,
        )

        # Create child spans for each assistant turn after the last user prompt
        _build_child_spans(parent_span, conversation, last_user_idx)

        final_response = _find_final_assistant_response(conversation, last_user_idx + 1)

        # Determine conversation end time
        last_entry = conversation[-1]
        conv_end_ns = parse_timestamp_to_ns(last_entry.get("timestamp"))
        if not conv_end_ns or (conv_start_ns and conv_end_ns <= conv_start_ns):
            conv_end_ns = (conv_start_ns or 0) + int(10 * NANOSECONDS_PER_S)

        # Accumulate token usage across all assistant turns
        total_usage: dict[str, int] = {}
        for entry in conversation:
            if entry.get("role") == ROLE_ASSISTANT:
                usage = entry.get("usage", {})
                for key, val in usage.items():
                    total_usage[key] = total_usage.get(key, 0) + int(val or 0)

        return _finalize_trace(
            parent_span,
            user_prompt,
            final_response,
            session_id,
            end_time_ns=conv_end_ns,
            usage=total_usage or None,
            kiro_version=kiro_version,
        )

    except Exception as exc:
        get_logger().error("Error processing Kiro session: %s", exc, exc_info=True)
        return None
