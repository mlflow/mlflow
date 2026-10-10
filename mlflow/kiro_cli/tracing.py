"""Transcript parsing and span-tree construction for Kiro CLI sessions.

Reads the paired ``.jsonl`` + ``.json`` session files that Kiro CLI writes
under ``~/.kiro/sessions/cli/``, reconstructs the most recent completed turn,
and emits a single MLflow trace with an AGENT root span, a CHAIN turn child,
and per-tool-call TOOL / per-text-response LLM grandchildren.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
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
from mlflow.kiro_cli.config import get_env_var
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey, TraceMetadataKey
from mlflow.tracing.provider import _get_trace_exporter
from mlflow.tracing.trace_manager import InMemoryTraceManager

# ============================================================================
# CONSTANTS (Sub-task 6.2)
# ============================================================================

# Custom logging level for Kiro CLI tracing (logging.WARNING - 5 = 25)
KIRO_TRACING_LEVEL = logging.WARNING - 5

NANOSECONDS_PER_S = 1_000_000_000
MAX_PREVIEW_LENGTH = 1000
DEFAULT_TURN_DURATION_S = 10

# Span name constants
AGENT_SPAN_NAME = "kiro_cli_conversation"
TURN_SPAN_NAME = "turn"


# ============================================================================
# TRANSCRIPT DATACLASSES (Sub-task 6.3)
# ============================================================================


@dataclass
class ToolUseBlock:
    tool_use_id: str
    name: str
    input: Any


@dataclass
class ToolResult:
    tool_use_id: str
    content: str
    status: str


@dataclass
class PromptRecord:
    kind: str  # "Prompt"
    message_id: str
    text: str
    timestamp_epoch_s: float | None


@dataclass
class AssistantMessageRecord:
    kind: str  # "AssistantMessage"
    message_id: str
    text: str
    tool_uses: list[ToolUseBlock]


@dataclass
class ToolResultsRecord:
    kind: str  # "ToolResults"
    message_id: str
    results: dict[str, ToolResult]


@dataclass
class TurnMetadata:
    loop_id: str | None
    message_ids: list[str]
    turn_duration_secs: float | None
    turn_duration_nanos: int | None
    end_timestamp: str | None
    input_token_count: int
    output_token_count: int
    context_usage_percentage: float | None
    metering_usage: list[dict]
    end_reason: str | None


@dataclass
class Turn:
    prompt: PromptRecord
    assistant_messages: list[AssistantMessageRecord] = field(default_factory=list)
    tool_results: list[ToolResultsRecord] = field(default_factory=list)
    metadata: TurnMetadata | None = None


# ============================================================================
# LOGGING (Sub-task 6.1)
# ============================================================================


def setup_logging() -> logging.Logger:
    """Set up logging directory and return configured logger.

    Creates .kiro/mlflow directory structure and configures file-based logging.
    Falls back to stderr handler when the log directory cannot be created.
    """
    logger = logging.getLogger("mlflow.kiro_cli.tracing")
    logger.handlers.clear()

    log_dir = Path(os.getcwd()) / ".kiro" / "mlflow"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "kiro_tracing.log"
        handler: logging.Handler = logging.FileHandler(log_file)
    except (OSError, PermissionError):
        handler = logging.StreamHandler(sys.stderr)

    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logging.addLevelName(KIRO_TRACING_LEVEL, "KIRO_TRACING")
    logger.setLevel(KIRO_TRACING_LEVEL)
    logger.propagate = False

    return logger


_MODULE_LOGGER: logging.Logger | None = None


def get_logger() -> logging.Logger:
    """Get the configured module logger, initializing on first call."""
    global _MODULE_LOGGER

    if _MODULE_LOGGER is None:
        _MODULE_LOGGER = setup_logging()
    return _MODULE_LOGGER


# ============================================================================
# INPUT / OUTPUT UTILITIES (Sub-tasks 6.4, 6.5)
# ============================================================================


def read_hook_input() -> dict[str, Any]:
    """Read JSON input from stdin for Kiro CLI hook processing.

    Returns an empty dict on empty stdin; raises on malformed JSON.
    """
    input_data = sys.stdin.read()
    if not input_data.strip():
        return {}
    return json.loads(input_data)


def get_hook_response(error: str | None = None, **kwargs) -> dict[str, Any]:
    """Build hook response dictionary for Kiro CLI hook protocol.

    Args:
        error: Error message if hook failed, None if successful.
        kwargs: Additional fields to include in response.

    Returns:
        Hook response dictionary.
    """
    if error is not None:
        return {"continue": False, "stopReason": error, **kwargs}
    return {"continue": True, **kwargs}


# ============================================================================
# PREVIEW TRUNCATION (Sub-task 6.10)
# ============================================================================


def truncate_preview(text: Any, max_length: int = MAX_PREVIEW_LENGTH) -> str:
    """Truncate text to max_length characters for span previews.

    Non-string inputs are serialized via ``json.dumps`` before truncation.

    Args:
        text: The text (or object) to truncate.
        max_length: Maximum character length (default 1000).

    Returns:
        Truncated string.
    """
    if not isinstance(text, str):
        try:
            text = json.dumps(text)
        except (TypeError, ValueError):
            text = str(text)
    return text[:max_length]


# ============================================================================
# TRANSCRIPT PARSING (Sub-task 6.6)
# ============================================================================


def _extract_text_from_content(content: list[dict[str, Any]]) -> str:
    """Extract concatenated text from a Kiro content array.

    Kiro content items have shape ``{"kind": "text", "data": "..."}``
    """
    parts = []
    if not isinstance(content, list):
        return ""
    for item in content:
        if isinstance(item, dict) and item.get("kind") == "text":
            data = item.get("data", "")
            if isinstance(data, str):
                parts.append(data)
    return "\n".join(parts) if parts else ""


def _extract_tool_uses(content: list[dict[str, Any]]) -> list[ToolUseBlock]:
    """Extract tool-use blocks from a Kiro content array."""
    tool_uses = []
    if not isinstance(content, list):
        return tool_uses
    for item in content:
        if isinstance(item, dict) and item.get("kind") == "toolUse":
            data = item.get("data", {})
            if isinstance(data, dict):
                tool_use_id = data.get("toolUseId", "")
                name = data.get("name", "")
                tool_input = data.get("input", {})
                if tool_use_id:
                    tool_uses.append(
                        ToolUseBlock(
                            tool_use_id=tool_use_id,
                            name=name,
                            input=tool_input,
                        )
                    )
    return tool_uses


def _extract_tool_results(content: list[dict[str, Any]]) -> dict[str, ToolResult]:
    """Extract tool results from a Kiro ToolResults content array."""
    results: dict[str, ToolResult] = {}
    if not isinstance(content, list):
        return results
    for item in content:
        if isinstance(item, dict) and item.get("kind") == "toolResult":
            data = item.get("data", {})
            if isinstance(data, dict):
                tool_use_id = data.get("toolUseId", "")
                # Content may be a list of text items or a string
                raw_content = data.get("content", "")
                if isinstance(raw_content, list):
                    content_text = _extract_text_from_content(raw_content)
                elif isinstance(raw_content, str):
                    content_text = raw_content
                else:
                    content_text = str(raw_content)
                status = data.get("status", "")
                if tool_use_id:
                    results[tool_use_id] = ToolResult(
                        tool_use_id=tool_use_id,
                        content=content_text,
                        status=status,
                    )
    return results


def _parse_record(
    raw: dict[str, Any],
) -> PromptRecord | AssistantMessageRecord | ToolResultsRecord | None:
    """Parse a single JSONL record into a typed dataclass, or None if invalid.

    Handles both the wrapped format (``{"version": "v1", "kind": "...", "data": {...}}``)
    used by real Kiro CLI and the flat format (``{"kind": "...", "message_id": "...", ...}``)
    used in test fixtures.
    """
    kind = raw.get("kind")

    # Real Kiro CLI wraps the payload in a "data" envelope
    if "data" in raw and isinstance(raw["data"], dict):
        inner = raw["data"]
        message_id = inner.get("message_id", "")
        content = inner.get("content", [])
        meta = inner.get("meta", {})
    else:
        # Flat format (test fixtures)
        message_id = raw.get("message_id", "")
        content = raw.get("content", [])
        meta = raw.get("meta", {})

    if kind == "Prompt":
        if not message_id:
            return None
        text = _extract_text_from_content(content)
        timestamp = meta.get("timestamp") if isinstance(meta, dict) else None
        return PromptRecord(
            kind="Prompt",
            message_id=message_id,
            text=text,
            timestamp_epoch_s=float(timestamp) if timestamp is not None else None,
        )
    elif kind == "AssistantMessage":
        if not message_id:
            return None
        text = _extract_text_from_content(content)
        tool_uses = _extract_tool_uses(content)
        return AssistantMessageRecord(
            kind="AssistantMessage",
            message_id=message_id,
            text=text,
            tool_uses=tool_uses,
        )
    elif kind == "ToolResults":
        if not message_id:
            return None
        results = _extract_tool_results(content)
        return ToolResultsRecord(
            kind="ToolResults",
            message_id=message_id,
            results=results,
        )
    else:
        return None  # Unknown kind


def parse_transcript(
    jsonl_path: Path,
) -> list[PromptRecord | AssistantMessageRecord | ToolResultsRecord]:
    """Parse a Kiro CLI transcript JSONL file into typed records.

    Tolerant line-by-line parser: skips empty lines, lines that fail JSON
    parsing, and records with unknown ``kind`` or missing required fields.
    Logs a WARNING for each skipped line. Never raises on malformed trailing
    lines.

    Args:
        jsonl_path: Path to the ``.jsonl`` transcript file.

    Returns:
        List of parsed transcript records.
    """
    records: list[PromptRecord | AssistantMessageRecord | ToolResultsRecord] = []
    logger = get_logger()

    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    raw = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON at line %d in %s", line_num, jsonl_path)
                    continue

                if not isinstance(raw, dict):
                    logger.warning(
                        "Skipping non-object JSON at line %d in %s", line_num, jsonl_path
                    )
                    continue

                record = _parse_record(raw)
                if record is None:
                    kind = raw.get("kind", "<missing>")
                    if kind not in ("Prompt", "AssistantMessage", "ToolResults"):
                        logger.warning(
                            "Skipping unknown kind '%s' at line %d in %s",
                            kind,
                            line_num,
                            jsonl_path,
                        )
                    else:
                        logger.warning(
                            "Skipping record with missing required fields at line %d in %s",
                            line_num,
                            jsonl_path,
                        )
                    continue

                records.append(record)
    except OSError as e:
        logger.warning("Could not read transcript file %s: %s", jsonl_path, e)

    return records


# ============================================================================
# SESSION JSON PARSING (Sub-task 6.7)
# ============================================================================


def parse_session_json(json_path: Path) -> dict[str, Any]:
    """Parse the aggregate session JSON file and return its contents.

    Missing or unparseable file returns an empty dict; never raises.

    Args:
        json_path: Path to the ``.json`` session file.

    Returns:
        Parsed session data dictionary.
    """
    logger = get_logger()
    try:
        if not json_path.exists():
            return {}
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("Session JSON at %s is not a dict", json_path)
            return {}
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not parse session JSON %s: %s", json_path, e)
        return {}


def _get_turn_metadatas(session_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract user_turn_metadatas from session JSON."""
    session_state = session_json.get("session_state", {})
    if not isinstance(session_state, dict):
        return []
    conv_meta = session_state.get("conversation_metadata", {})
    if not isinstance(conv_meta, dict):
        return []
    metadatas = conv_meta.get("user_turn_metadatas", [])
    if not isinstance(metadatas, list):
        return []
    return metadatas


def _parse_turn_metadata(raw: dict[str, Any]) -> TurnMetadata:
    """Parse a single user_turn_metadatas entry into a TurnMetadata dataclass."""
    loop_id = raw.get("loop_id")
    message_ids = raw.get("message_ids", [])
    if not isinstance(message_ids, list):
        message_ids = []

    # Parse turn_duration which has {secs, nanos} shape
    turn_duration = raw.get("turn_duration")
    turn_duration_secs: float | None = None
    turn_duration_nanos: int | None = None
    if isinstance(turn_duration, dict):
        secs = turn_duration.get("secs")
        nanos = turn_duration.get("nanos")
        if secs is not None:
            turn_duration_secs = float(secs)
        if nanos is not None:
            turn_duration_nanos = int(nanos)

    end_timestamp = raw.get("end_timestamp")
    input_token_count = int(raw.get("input_token_count", 0))
    output_token_count = int(raw.get("output_token_count", 0))
    context_usage_percentage = raw.get("context_usage_percentage")
    if context_usage_percentage is not None:
        try:
            context_usage_percentage = float(context_usage_percentage)
        except (TypeError, ValueError):
            context_usage_percentage = None

    metering_usage = raw.get("metering_usage", [])
    if not isinstance(metering_usage, list):
        metering_usage = []

    end_reason = raw.get("end_reason")

    return TurnMetadata(
        loop_id=loop_id,
        message_ids=message_ids,
        turn_duration_secs=turn_duration_secs,
        turn_duration_nanos=turn_duration_nanos,
        end_timestamp=end_timestamp,
        input_token_count=input_token_count,
        output_token_count=output_token_count,
        context_usage_percentage=context_usage_percentage,
        metering_usage=metering_usage,
        end_reason=end_reason,
    )


def _get_session_metadata(session_json: dict[str, Any]) -> dict[str, Any]:
    """Extract agent_name, model_id, context_window_tokens from session JSON."""
    result: dict[str, Any] = {}
    session_state = session_json.get("session_state", {})
    if not isinstance(session_state, dict):
        return result

    agent_name = session_state.get("agent_name")
    if agent_name:
        result["agent_name"] = agent_name

    rts = session_state.get("rts_model_state", {})
    if isinstance(rts, dict):
        model_info = rts.get("model_info", {})
        if isinstance(model_info, dict):
            model_id = model_info.get("model_id")
            if model_id:
                result["model_id"] = model_id
            ctx_window = model_info.get("context_window_tokens")
            if ctx_window is not None:
                result["context_window_tokens"] = ctx_window

    return result


# ============================================================================
# TURN GROUPING (Sub-task 6.8)
# ============================================================================


def group_turns(
    records: list[PromptRecord | AssistantMessageRecord | ToolResultsRecord],
) -> list[Turn]:
    """Group transcript records into turns.

    A turn starts at each ``Prompt`` record and ends immediately before the
    next ``Prompt`` record (or at end-of-file).

    Args:
        records: Parsed transcript records.

    Returns:
        List of Turn objects.
    """
    turns: list[Turn] = []
    current_turn: Turn | None = None

    for record in records:
        if isinstance(record, PromptRecord):
            if current_turn is not None:
                turns.append(current_turn)
            current_turn = Turn(prompt=record)
        elif current_turn is not None:
            if isinstance(record, AssistantMessageRecord):
                current_turn.assistant_messages.append(record)
            elif isinstance(record, ToolResultsRecord):
                current_turn.tool_results.append(record)

    if current_turn is not None:
        turns.append(current_turn)

    return turns


# ============================================================================
# FIND LAST TURN (Sub-task 6.9)
# ============================================================================


def find_last_turn(
    records: list[PromptRecord | AssistantMessageRecord | ToolResultsRecord],
    session_json: dict[str, Any],
) -> Turn | None:
    """Select the final turn and attach its TurnMetadata by message_id membership.

    Args:
        records: Parsed transcript records.
        session_json: Parsed session JSON data.

    Returns:
        The last Turn with metadata attached, or None if no turns exist.
    """
    turns = group_turns(records)
    if not turns:
        return None

    last_turn = turns[-1]

    # Try to match metadata by message_id membership
    turn_metadatas = _get_turn_metadatas(session_json)
    if turn_metadatas:
        # Collect all message_ids in the last turn
        turn_message_ids = {last_turn.prompt.message_id}
        turn_message_ids.update(am.message_id for am in last_turn.assistant_messages)
        turn_message_ids.update(tr.message_id for tr in last_turn.tool_results)

        # Find the metadata entry whose message_ids overlap with the turn's
        for raw_meta in reversed(turn_metadatas):
            meta_msg_ids = raw_meta.get("message_ids", [])
            if isinstance(meta_msg_ids, list) and any(
                mid in turn_message_ids for mid in meta_msg_ids
            ):
                last_turn.metadata = _parse_turn_metadata(raw_meta)
                break

    return last_turn


# ============================================================================
# TOKEN USAGE (Sub-task 6.11 helper)
# ============================================================================


def _build_usage_dict(metadata: TurnMetadata) -> dict[str, int]:
    """Build a CHAT_USAGE dict from TurnMetadata token counts."""
    input_tokens = metadata.input_token_count
    output_tokens = metadata.output_token_count
    return {
        TokenUsageKey.INPUT_TOKENS: input_tokens,
        TokenUsageKey.OUTPUT_TOKENS: output_tokens,
        TokenUsageKey.TOTAL_TOKENS: input_tokens + output_tokens,
    }


# ============================================================================
# TIMESTAMP RECONSTRUCTION (Sub-task 6.12)
# ============================================================================


def _compute_turn_timestamps(
    turn: Turn,
) -> tuple[int, int]:
    """Compute turn start and end timestamps in nanoseconds.

    Args:
        turn: The Turn to compute timestamps for.

    Returns:
        Tuple of (turn_start_ns, turn_end_ns).
    """
    # Start time from prompt timestamp
    if turn.prompt.timestamp_epoch_s is not None:
        turn_start_ns = int(turn.prompt.timestamp_epoch_s * NANOSECONDS_PER_S)
    else:
        # Fallback: use current time
        import time

        turn_start_ns = int(time.time() * NANOSECONDS_PER_S)

    # End time: prefer end_timestamp, then turn_duration, then default +10s
    turn_end_ns = None
    if turn.metadata is not None:
        # Prefer parsed end_timestamp
        if turn.metadata.end_timestamp:
            try:
                dt = dateutil.parser.parse(turn.metadata.end_timestamp)
                turn_end_ns = int(dt.timestamp() * NANOSECONDS_PER_S)
            except Exception:
                pass

        # Fallback to turn_duration
        if turn_end_ns is None:
            duration_ns = 0
            if turn.metadata.turn_duration_secs is not None:
                duration_ns += int(turn.metadata.turn_duration_secs * NANOSECONDS_PER_S)
            if turn.metadata.turn_duration_nanos is not None:
                duration_ns += turn.metadata.turn_duration_nanos
            if duration_ns > 0:
                turn_end_ns = turn_start_ns + duration_ns

    # Final fallback: default +10s
    if turn_end_ns is None:
        turn_end_ns = turn_start_ns + int(DEFAULT_TURN_DURATION_S * NANOSECONDS_PER_S)

    # Ensure end > start
    if turn_end_ns <= turn_start_ns:
        turn_end_ns = turn_start_ns + int(DEFAULT_TURN_DURATION_S * NANOSECONDS_PER_S)

    return turn_start_ns, turn_end_ns


def _allocate_grandchild_slices(
    turn_start_ns: int, turn_end_ns: int, count: int
) -> list[tuple[int, int]]:
    """Allocate equal proportional time slices to N grandchildren.

    Args:
        turn_start_ns: Turn start in nanoseconds.
        turn_end_ns: Turn end in nanoseconds.
        count: Number of grandchildren.

    Returns:
        List of (start_ns, end_ns) tuples.
    """
    if count <= 0:
        return []
    total_duration = turn_end_ns - turn_start_ns
    slice_duration = total_duration // count
    slices = []
    for i in range(count):
        start = turn_start_ns + i * slice_duration
        end = start + slice_duration
        slices.append((start, end))
    return slices


# ============================================================================
# SPAN BUILDERS (Sub-task 6.11)
# ============================================================================


def _find_final_assistant_text(turn: Turn) -> str | None:
    """Find the final text response from assistant messages in a turn."""
    final_text = None
    for am in turn.assistant_messages:
        if am.text and am.text.strip():
            final_text = am.text
    return final_text


def _collect_grandchild_specs(
    turn: Turn,
) -> list[dict[str, Any]]:
    """Collect specifications for grandchild spans (LLM and TOOL).

    Returns a list of dicts with keys: type ("llm" or "tool"), and relevant data.
    """
    specs: list[dict[str, Any]] = []

    # Build a tool_use_id -> ToolResult lookup from all ToolResults records
    tool_result_map: dict[str, ToolResult] = {}
    for tr in turn.tool_results:
        for tool_use_id, result in tr.results.items():
            tool_result_map[tool_use_id] = result

    for am in turn.assistant_messages:
        has_text = bool(am.text and am.text.strip())
        has_tools = bool(am.tool_uses)

        # If message has text content, create an LLM span
        if has_text:
            specs.append({
                "type": "llm",
                "text": am.text,
                "message_id": am.message_id,
            })

        # If message has tool uses, create a TOOL span per tool use
        if has_tools:
            for tu in am.tool_uses:
                tool_result = tool_result_map.get(tu.tool_use_id)
                result_content = tool_result.content if tool_result else "No result found"
                specs.append({
                    "type": "tool",
                    "tool_use": tu,
                    "result_content": result_content,
                })

    return specs


def _create_grandchild_spans(
    turn_span,
    turn: Turn,
    time_slices: list[tuple[int, int]],
    model_id: str,
) -> None:
    """Create LLM and TOOL grandchild spans under the turn CHAIN span."""
    specs = _collect_grandchild_specs(turn)

    for i, spec in enumerate(specs):
        if i < len(time_slices):
            start_ns, end_ns = time_slices[i]
        else:
            # Shouldn't happen, but be safe
            start_ns, end_ns = None, None

        if spec["type"] == "llm":
            llm_span = mlflow.start_span_no_context(
                name="llm",
                parent_span=turn_span,
                span_type=SpanType.LLM,
                start_time_ns=start_ns,
                inputs={
                    "model": model_id,
                    "messages": [{"role": "user", "content": truncate_preview(turn.prompt.text)}],
                },
                attributes={
                    "model": model_id,
                },
            )
            llm_span.set_outputs({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": truncate_preview(spec["text"])}],
            })
            llm_span.end(end_time_ns=end_ns)

        elif spec["type"] == "tool":
            tu: ToolUseBlock = spec["tool_use"]
            tool_span = mlflow.start_span_no_context(
                name=f"tool_{tu.name}",
                parent_span=turn_span,
                span_type=SpanType.TOOL,
                start_time_ns=start_ns,
                inputs=truncate_preview(tu.input),
                attributes={
                    "tool_name": tu.name,
                    "tool_id": tu.tool_use_id,
                },
            )
            tool_span.set_outputs({"result": truncate_preview(spec["result_content"])})
            tool_span.end(end_time_ns=end_ns)


# ============================================================================
# TRACE FINALIZATION (Sub-task 6.13)
# ============================================================================


def _finalize_trace(
    parent_span,
    user_prompt: str,
    final_response: str | None,
    session_id: str,
    cwd: str,
    session_metadata: dict[str, Any],
    turn_metadata: TurnMetadata | None,
    end_time_ns: int | None = None,
) -> mlflow.entities.Trace:
    """Set trace previews, metadata, and end the root span."""
    try:
        with InMemoryTraceManager.get_instance().get_trace(parent_span.trace_id) as in_memory_trace:
            if user_prompt:
                in_memory_trace.info.request_preview = truncate_preview(user_prompt)
            if final_response:
                in_memory_trace.info.response_preview = truncate_preview(final_response)

            metadata = {
                TraceMetadataKey.TRACE_USER: os.environ.get("USER", ""),
                TraceMetadataKey.TRACE_SESSION: session_id,
                "mlflow.trace.working_directory": cwd,
            }

            # Session-level metadata
            if "agent_name" in session_metadata:
                metadata["mlflow.kiro_cli.agent_name"] = session_metadata["agent_name"]
            if "model_id" in session_metadata:
                metadata["mlflow.kiro_cli.model_id"] = session_metadata["model_id"]
            if "context_window_tokens" in session_metadata:
                metadata["mlflow.kiro_cli.context_window_tokens"] = str(
                    session_metadata["context_window_tokens"]
                )

            # Token usage at trace level
            if turn_metadata and (
                turn_metadata.input_token_count > 0 or turn_metadata.output_token_count > 0
            ):
                usage_dict = _build_usage_dict(turn_metadata)
                metadata[TraceMetadataKey.TOKEN_USAGE] = json.dumps(usage_dict)

            in_memory_trace.info.trace_metadata = {
                **in_memory_trace.info.trace_metadata,
                **metadata,
            }
    except Exception as e:
        get_logger().warning("Failed to update trace metadata and previews: %s", e)

    outputs = {"status": "completed"}
    if final_response:
        outputs["response"] = truncate_preview(final_response)
    parent_span.set_outputs(outputs)
    parent_span.end(end_time_ns=end_time_ns)
    _flush_trace_async_logging()
    get_logger().log(KIRO_TRACING_LEVEL, "Created MLflow trace: %s", parent_span.trace_id)
    return mlflow.get_trace(parent_span.trace_id)


def _flush_trace_async_logging() -> None:
    """Flush any pending async trace logging."""
    try:
        if hasattr(_get_trace_exporter(), "_async_queue"):
            mlflow.flush_trace_async_logging()
    except Exception as e:
        get_logger().debug("Failed to flush trace async logging: %s", e)


# ============================================================================
# MLFLOW SETUP (Sub-task 6.14)
# ============================================================================


def setup_mlflow() -> None:
    """Configure MLflow tracking URI and experiment from env-var precedence.

    Reads configuration from ``.kiro/settings.json`` (project-scoped) with
    fallback to OS environment variables. Experiment ID takes precedence
    over experiment name when both resolve to values.
    """
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


# ============================================================================
# PROCESS TURN (Sub-task 6.15)
# ============================================================================


def process_turn(
    transcript_jsonl: Path,
    transcript_json: Path,
    session_id: str,
    cwd: str,
) -> mlflow.entities.Trace | None:
    """Parse the transcript and emit an MLflow trace for the most recent turn.

    Reads the paired ``.jsonl`` and ``.json`` session files, identifies the
    most recent completed turn, and builds a span tree with an AGENT root,
    a CHAIN turn child, and per-tool/per-text grandchildren.

    Args:
        transcript_jsonl: Path to the ``.jsonl`` transcript file.
        transcript_json: Path to the ``.json`` session file.
        session_id: Session identifier from the hook payload.
        cwd: Working directory from the hook payload.

    Returns:
        MLflow Trace if successful, None if processing fails or no records found.
    """
    logger = get_logger()

    # Parse transcript records
    records = parse_transcript(transcript_jsonl)
    if not records:
        logger.warning("Empty or unparseable transcript at %s, skipping", transcript_jsonl)
        return None

    # Parse session JSON for metadata
    session_json = parse_session_json(transcript_json)
    session_metadata = _get_session_metadata(session_json)

    # Find the last turn
    last_turn = find_last_turn(records, session_json)
    if last_turn is None:
        logger.warning("No turns found in transcript %s, skipping", transcript_jsonl)
        return None

    logger.log(KIRO_TRACING_LEVEL, "Creating MLflow trace for session: %s", session_id)

    # Compute timestamps
    turn_start_ns, turn_end_ns = _compute_turn_timestamps(last_turn)

    # Model ID for LLM spans
    model_id = session_metadata.get("model_id", "unknown")

    # Create root AGENT span
    parent_span = mlflow.start_span_no_context(
        name=AGENT_SPAN_NAME,
        inputs={"prompt": truncate_preview(last_turn.prompt.text)},
        start_time_ns=turn_start_ns,
        span_type=SpanType.AGENT,
    )

    # Create CHAIN turn child span
    turn_span = mlflow.start_span_no_context(
        name=TURN_SPAN_NAME,
        parent_span=parent_span,
        span_type=SpanType.CHAIN,
        start_time_ns=turn_start_ns,
    )

    # Set turn-level attributes from metadata
    if last_turn.metadata is not None:
        meta = last_turn.metadata

        # Token usage (CHAT_USAGE) when non-zero
        if meta.input_token_count > 0 or meta.output_token_count > 0:
            turn_span.set_attribute(SpanAttributeKey.CHAT_USAGE, _build_usage_dict(meta))

        # Metering usage attributes
        for entry in meta.metering_usage:
            if isinstance(entry, dict):
                unit = entry.get("unit", "unknown")
                value = entry.get("value")
                if value is not None:
                    turn_span.set_attribute(f"mlflow.kiro_cli.metering_usage.{unit}", value)

        # Context usage percentage
        if meta.context_usage_percentage is not None:
            turn_span.set_attribute(
                "mlflow.kiro_cli.context_usage_percentage", meta.context_usage_percentage
            )

        # End reason
        if meta.end_reason is not None:
            turn_span.set_attribute("mlflow.kiro_cli.end_reason", meta.end_reason)

    # Create grandchild spans (LLM and TOOL)
    grandchild_specs = _collect_grandchild_specs(last_turn)
    time_slices = _allocate_grandchild_slices(turn_start_ns, turn_end_ns, len(grandchild_specs))
    _create_grandchild_spans(turn_span, last_turn, time_slices, model_id)

    # End the turn span
    turn_span.end(end_time_ns=turn_end_ns)

    # Finalize the trace
    final_response = _find_final_assistant_text(last_turn)

    return _finalize_trace(
        parent_span,
        last_turn.prompt.text,
        final_response,
        session_id,
        cwd,
        session_metadata,
        last_turn.metadata,
        end_time_ns=turn_end_ns,
    )
