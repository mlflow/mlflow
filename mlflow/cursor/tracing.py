"""MLflow tracing integration for Cursor AI interactions."""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.cursor.config import (
    CURSOR_DIR,
    get_env_var,
    is_tracing_enabled,
)
from mlflow.entities import SpanType
from mlflow.environment_variables import (
    MLFLOW_EXPERIMENT_ID,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracking.fluent import _get_trace_exporter

NANOSECONDS_PER_MS = 1e6
NANOSECONDS_PER_S = 1e9
MAX_PREVIEW_LENGTH = 1000
EVENTS_DIR = "events"

# Custom logging level for Cursor tracing
CURSOR_TRACING_LEVEL = logging.WARNING - 5


def setup_logging() -> logging.Logger:
    """Set up logging directory and return configured logger.

    Creates .cursor/mlflow directory structure and configures file-based logging.
    """
    log_dir = Path(os.getcwd()) / CURSOR_DIR / "mlflow"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.handlers.clear()

    log_file = log_dir / "cursor_tracing.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logging.addLevelName(CURSOR_TRACING_LEVEL, "CURSOR_TRACING")
    logger.setLevel(CURSOR_TRACING_LEVEL)
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


def read_hook_input() -> dict[str, Any]:
    """Read JSON input from stdin for Cursor hook processing."""
    try:
        input_data = sys.stdin.read()
        return json.loads(input_data)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Failed to parse hook input: {e}", input_data, 0) from e


def get_hook_response(
    continue_execution: bool = True,
    permission: str = "allow",
    error: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Build hook response dictionary for Cursor hook protocol.

    Args:
        continue_execution: Whether to continue the execution
        permission: Permission for the action ("allow" or "deny")
        error: Error message if hook failed
        kwargs: Additional fields to include in response

    Returns:
        Hook response dictionary
    """
    response = {"continue": continue_execution, "permission": permission, **kwargs}
    if error is not None:
        response["error"] = error
    return response


def get_current_time_ns() -> int:
    """Get current time in nanoseconds since Unix epoch."""
    return int(datetime.now().timestamp() * NANOSECONDS_PER_S)


def generate_session_id(workspace_roots: list[str] | None) -> str:
    """Generate a session ID based on workspace roots.

    Args:
        workspace_roots: List of workspace root paths

    Returns:
        Session ID string
    """
    if workspace_roots and len(workspace_roots) > 0:
        return Path(workspace_roots[0]).name
    return "cursor-session"


def generate_trace_name(prompt: str | None, model: str | None) -> str:
    """Generate a trace name from prompt and model.

    Args:
        prompt: User prompt text
        model: Model name

    Returns:
        Trace name string
    """
    if prompt:
        return prompt[:50] + ("..." if len(prompt) > 50 else "")
    return f"Cursor Agent - {model or 'unknown'}"


def _get_events_dir() -> Path:
    """Get the directory for storing conversation events."""
    return Path(os.getcwd()) / CURSOR_DIR / "mlflow" / EVENTS_DIR


def _get_events_file(conversation_id: str) -> Path:
    """Get the events file path for a conversation."""
    events_dir = _get_events_dir()
    events_dir.mkdir(parents=True, exist_ok=True)
    return events_dir / f"{conversation_id}.jsonl"


def append_event(conversation_id: str, event: dict[str, Any]) -> None:
    """Append an event to the conversation's event log.

    Events are stored in a JSONL file, one event per line.

    Args:
        conversation_id: Unique identifier for the conversation
        event: Event data to append
    """
    events_file = _get_events_file(conversation_id)

    # Add timestamp to event
    event["timestamp_ns"] = get_current_time_ns()
    event["timestamp"] = datetime.now().isoformat()

    try:
        with open(events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except IOError as e:
        get_logger().error("Failed to append event: %s", e)


def read_events(conversation_id: str) -> list[dict[str, Any]]:
    """Read all events for a conversation.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        List of events in chronological order
    """
    events_file = _get_events_file(conversation_id)
    if not events_file.exists():
        return []

    try:
        with open(events_file, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except (json.JSONDecodeError, IOError) as e:
        get_logger().error("Failed to read events: %s", e)
        return []


def cleanup_events(conversation_id: str) -> None:
    """Remove the events file for a conversation.

    Args:
        conversation_id: Unique identifier for the conversation
    """
    events_file = _get_events_file(conversation_id)
    try:
        if events_file.exists():
            events_file.unlink()
    except IOError as e:
        get_logger().warning("Failed to cleanup events file: %s", e)


def create_trace_from_events(
    conversation_id: str,
    events: list[dict[str, Any]],
    final_status: str = "completed",
) -> mlflow.entities.Trace | None:
    """Create a single MLflow trace from collected events.

    This is called when the stop hook fires, creating a proper trace
    with nested spans for all events in the conversation.

    Args:
        conversation_id: Unique identifier for the conversation
        events: List of events collected during the conversation
        final_status: Final status of the agent

    Returns:
        MLflow trace object if successful, None if processing fails
    """
    if not events:
        get_logger().warning("No events to create trace from")
        return None

    get_logger().log(
        CURSOR_TRACING_LEVEL,
        "Creating trace from %d events for conversation %s",
        len(events),
        conversation_id,
    )

    # Find the first event to get initial data
    first_event = events[0]
    last_event = events[-1]

    # Extract metadata from first event
    model = first_event.get("model", "unknown")
    workspace_roots = first_event.get("workspace_roots", [])
    user_email = first_event.get("user_email")
    cursor_version = first_event.get("cursor_version")

    # Find user prompt from beforeSubmitPrompt events
    user_prompt = ""
    for event in events:
        if event.get("event_type") == "beforeSubmitPrompt":
            user_prompt = event.get("prompt", "")
            if user_prompt:
                break

    # Find final response from afterAgentResponse events
    final_response = ""
    for event in reversed(events):
        if event.get("event_type") == "afterAgentResponse":
            final_response = event.get("text", "")
            if final_response:
                break

    # Calculate timing
    start_time_ns = first_event.get("timestamp_ns", get_current_time_ns())
    end_time_ns = last_event.get("timestamp_ns", get_current_time_ns())

    # Create the parent AGENT span
    parent_span = mlflow.start_span_no_context(
        name=generate_trace_name(user_prompt, model),
        inputs={"prompt": user_prompt} if user_prompt else {},
        start_time_ns=start_time_ns,
        span_type=SpanType.AGENT,
        attributes={
            "cursor.conversation_id": conversation_id,
            "cursor.model": model,
            "cursor.version": cursor_version or "unknown",
        },
    )

    # Create child spans for each event
    _create_spans_from_events(parent_span, events)

    # Note: Token usage is not available from Cursor hooks
    # Cursor tracks tokens internally for billing but doesn't expose them via hooks

    # Set trace metadata
    try:
        with InMemoryTraceManager.get_instance().get_trace(parent_span.trace_id) as in_memory_trace:
            if user_prompt:
                in_memory_trace.info.request_preview = user_prompt[:MAX_PREVIEW_LENGTH]
            if final_response:
                in_memory_trace.info.response_preview = final_response[:MAX_PREVIEW_LENGTH]
            in_memory_trace.info.trace_metadata = {
                **in_memory_trace.info.trace_metadata,
                TraceMetadataKey.TRACE_SESSION: generate_session_id(workspace_roots),
                TraceMetadataKey.TRACE_USER: user_email or os.environ.get("USER", ""),
                "mlflow.trace.working_directory": os.getcwd(),
                "cursor.conversation_id": conversation_id,
            }
    except Exception as e:
        get_logger().warning("Failed to update trace metadata: %s", e)

    # End the parent span
    parent_span.set_outputs(
        {
            "response": final_response or "Conversation completed",
            "status": final_status,
        }
    )
    parent_span.end(end_time_ns=end_time_ns)

    # Flush traces
    try:
        if hasattr(_get_trace_exporter(), "_async_queue"):
            mlflow.flush_trace_async_logging()
    except Exception as e:
        get_logger().debug("Failed to flush trace async logging: %s", e)

    get_logger().log(
        CURSOR_TRACING_LEVEL,
        "Created trace %s for conversation %s with %d events",
        parent_span.trace_id,
        conversation_id,
        len(events),
    )

    return mlflow.get_trace(parent_span.trace_id)


def _create_spans_from_events(parent_span: Any, events: list[dict[str, Any]]) -> None:
    """Create child spans from collected events.

    Args:
        parent_span: The parent AGENT span
        events: List of events to create spans from
    """
    # Build a map of preToolUse events keyed by tool_name for merging with postToolUse
    pre_tool_events: dict[str, dict[str, Any]] = {}

    for event in events:
        event_type = event.get("event_type", "unknown")
        timestamp_ns = event.get("timestamp_ns", get_current_time_ns())

        # Calculate span duration (use 100ms default if no duration provided)
        duration_ms = event.get("duration_ms", 100) or event.get("duration", 100)
        duration_ns = int(duration_ms * NANOSECONDS_PER_MS)
        end_time_ns = timestamp_ns + duration_ns

        if event_type == "beforeSubmitPrompt":
            _create_prompt_span(parent_span, event, timestamp_ns, end_time_ns)
        elif event_type == "afterAgentResponse":
            _create_response_span(parent_span, event, timestamp_ns, end_time_ns)
        elif event_type == "afterAgentThought":
            _create_thought_span(parent_span, event, timestamp_ns, end_time_ns)
        elif event_type in ("beforeShellExecution", "afterShellExecution"):
            _create_shell_span(parent_span, event, timestamp_ns, end_time_ns)
        elif event_type in ("beforeMCPExecution", "afterMCPExecution"):
            _create_mcp_span(parent_span, event, timestamp_ns, end_time_ns)
        elif event_type in ("beforeReadFile", "afterFileEdit"):
            _create_file_span(parent_span, event, timestamp_ns, end_time_ns)
        elif event_type in ("beforeTabFileRead", "afterTabFileEdit"):
            _create_tab_span(parent_span, event, timestamp_ns, end_time_ns)
        elif event_type == "preToolUse":
            # Store preToolUse for merging with postToolUse
            tool_name = event.get("tool_name", "unknown")
            pre_tool_events[tool_name] = event
        elif event_type == "postToolUse":
            # Merge with preToolUse to get inputs, then create span
            tool_name = event.get("tool_name", "unknown")
            pre_event = pre_tool_events.pop(tool_name, {})
            merged_event = _merge_tool_events(pre_event, event)
            _create_tool_use_span(parent_span, merged_event, timestamp_ns, end_time_ns)
        # Skip stop event - it's handled at the parent level


def _create_prompt_span(
    parent_span: Any, event: dict[str, Any], start_ns: int, end_ns: int
) -> None:
    """Create a span for user prompt."""
    prompt = event.get("prompt", "")
    model = event.get("model", "unknown")

    span = mlflow.start_span_no_context(
        name="user_prompt",
        parent_span=parent_span,
        span_type=SpanType.CHAIN,  # CHAIN for input processing
        start_time_ns=start_ns,
        inputs={"prompt": prompt, "model": model},
        attributes={
            "generation_id": event.get("generation_id", ""),
            "attachment_count": len(event.get("attachments", [])),
        },
    )
    span.end(end_time_ns=end_ns)


def _create_response_span(
    parent_span: Any, event: dict[str, Any], start_ns: int, end_ns: int
) -> None:
    """Create a span for agent response."""
    text = event.get("text", "")
    model = event.get("model", "unknown")

    span = mlflow.start_span_no_context(
        name="agent_response",
        parent_span=parent_span,
        span_type=SpanType.LLM,
        start_time_ns=start_ns,
        inputs={"model": model},
        attributes={
            "generation_id": event.get("generation_id", ""),
            "response_length": len(text),
            SpanAttributeKey.MODEL: model,
        },
    )
    span.set_outputs({"response": text})
    span.end(end_time_ns=end_ns)


def _create_thought_span(
    parent_span: Any, event: dict[str, Any], start_ns: int, end_ns: int
) -> None:
    """Create a span for agent thinking/reasoning."""
    text = event.get("text", "")
    duration_ms = event.get("duration_ms", 0)
    model = event.get("model", "unknown")

    span = mlflow.start_span_no_context(
        name="agent_thinking",
        parent_span=parent_span,
        span_type=SpanType.LLM,  # LLM for thinking/reasoning steps
        start_time_ns=start_ns,
        inputs={"type": "thinking", "model": model},
        attributes={
            "generation_id": event.get("generation_id", ""),
            "duration_ms": duration_ms,
            "thinking_length": len(text),
            SpanAttributeKey.MODEL: model,
        },
    )
    span.set_outputs({"thought": text})
    span.end(end_time_ns=end_ns)


def _create_shell_span(parent_span: Any, event: dict[str, Any], start_ns: int, end_ns: int) -> None:
    """Create a span for shell command execution."""
    event_type = event.get("event_type", "")
    command = event.get("command", "")
    cwd = event.get("cwd", "")
    output = event.get("output", "")

    # Truncate command for name
    cmd_preview = command[:30] + "..." if len(command) > 30 else command
    name = f"shell: {cmd_preview}"

    inputs = {"command": command}
    if cwd:
        inputs["cwd"] = cwd

    outputs = {}
    if output:
        outputs["output"] = output

    attributes = {
        "generation_id": event.get("generation_id", ""),
        "tool_name": "shell",
    }

    if event_type == "afterShellExecution":
        attributes["duration_ms"] = event.get("duration", 0)
        # Check for potential errors
        output_lower = output.lower() if output else ""
        attributes["might_have_failed"] = any(
            kw in output_lower for kw in ["error", "failed", "not found"]
        )

    span = mlflow.start_span_no_context(
        name=name,
        parent_span=parent_span,
        span_type=SpanType.TOOL,
        start_time_ns=start_ns,
        inputs=inputs,
        attributes=attributes,
    )
    if outputs:
        span.set_outputs(outputs)
    span.end(end_time_ns=end_ns)


def _create_mcp_span(parent_span: Any, event: dict[str, Any], start_ns: int, end_ns: int) -> None:
    """Create a span for MCP tool execution."""
    event_type = event.get("event_type", "")
    tool_name = event.get("tool_name", "unknown")
    tool_input = event.get("tool_input", {})
    result = event.get("result_json", {})

    name = f"mcp: {tool_name}"

    outputs = {}
    if result:
        outputs["result"] = result

    attributes = {
        "generation_id": event.get("generation_id", ""),
        "tool_name": tool_name,
        "server_url": event.get("url", ""),
    }

    if event_type == "afterMCPExecution":
        attributes["duration_ms"] = event.get("duration", 0)

    span = mlflow.start_span_no_context(
        name=name,
        parent_span=parent_span,
        span_type=SpanType.TOOL,
        start_time_ns=start_ns,
        inputs={"tool_name": tool_name, "tool_input": tool_input},
        attributes=attributes,
    )
    if outputs:
        span.set_outputs(outputs)
    span.end(end_time_ns=end_ns)


def _create_file_span(parent_span: Any, event: dict[str, Any], start_ns: int, end_ns: int) -> None:
    """Create a span for file operations."""
    event_type = event.get("event_type", "")
    file_path = event.get("file_path", "")
    file_name = Path(file_path).name if file_path else "file"
    extension = Path(file_path).suffix if file_path else ""

    if event_type == "beforeReadFile":
        name = f"read: {file_name}"
        outputs = {}
    else:  # afterFileEdit
        name = f"edit: {file_name}"
        edits = event.get("edits", [])
        edit_stats = _calculate_edit_stats(edits)
        outputs = edit_stats

    span = mlflow.start_span_no_context(
        name=name,
        parent_span=parent_span,
        span_type=SpanType.TOOL,
        start_time_ns=start_ns,
        inputs={"file_path": file_path, "extension": extension},
        attributes={
            "generation_id": event.get("generation_id", ""),
            "file_extension": extension,
            "tool_name": "read_file" if event_type == "beforeReadFile" else "edit_file",
        },
    )
    if outputs:
        span.set_outputs(outputs)
    span.end(end_time_ns=end_ns)


def _create_tab_span(parent_span: Any, event: dict[str, Any], start_ns: int, end_ns: int) -> None:
    """Create a span for tab operations."""
    event_type = event.get("event_type", "")
    file_path = event.get("file_path", "")
    file_name = Path(file_path).name if file_path else "file"
    extension = Path(file_path).suffix if file_path else ""

    if event_type == "beforeTabFileRead":
        name = f"tab_read: {file_name}"
        outputs = {}
    else:  # afterTabFileEdit
        name = f"tab_edit: {file_name}"
        edits = event.get("edits", [])
        edit_stats = _calculate_edit_stats(edits)
        outputs = edit_stats

    span = mlflow.start_span_no_context(
        name=name,
        parent_span=parent_span,
        span_type=SpanType.TOOL,
        start_time_ns=start_ns,
        inputs={"file_path": file_path, "extension": extension},
        attributes={
            "generation_id": event.get("generation_id", ""),
            "file_extension": extension,
            "source": "tab",
            "tool_name": "tab_read" if event_type == "beforeTabFileRead" else "tab_edit",
        },
    )
    if outputs:
        span.set_outputs(outputs)
    span.end(end_time_ns=end_ns)


def _merge_tool_events(pre_event: dict[str, Any], post_event: dict[str, Any]) -> dict[str, Any]:
    """Merge preToolUse and postToolUse events.

    Args:
        pre_event: The preToolUse event with inputs
        post_event: The postToolUse event with outputs

    Returns:
        Merged event with both inputs and outputs
    """
    merged = {**post_event}

    # Get tool_input from pre_event if not in post_event
    if not merged.get("tool_input") and pre_event.get("tool_input"):
        merged["tool_input"] = pre_event["tool_input"]

    # Use start time from pre_event if available
    if pre_event.get("timestamp_ns"):
        merged["start_timestamp_ns"] = pre_event["timestamp_ns"]

    return merged


def _create_tool_use_span(
    parent_span: Any, event: dict[str, Any], start_ns: int, end_ns: int
) -> None:
    """Create a span for tool use events (Grep, Glob, Read, Write, Task, etc.)."""
    tool_name = event.get("tool_name", "unknown")
    tool_input = event.get("tool_input", {})
    tool_output = event.get("tool_output", {})
    duration = event.get("duration", 0)

    # Normalize tool name for display
    tool_name_lower = tool_name.lower()

    # Create descriptive span name based on tool type
    if tool_name_lower in ("grep", "rg", "ripgrep"):
        pattern = tool_input.get("pattern", "") if isinstance(tool_input, dict) else ""
        name = f"grep: {pattern[:30]}..." if len(pattern) > 30 else f"grep: {pattern}"
    elif tool_name_lower == "glob":
        pattern = tool_input.get("glob_pattern", "") if isinstance(tool_input, dict) else ""
        name = f"glob: {pattern[:30]}..." if len(pattern) > 30 else f"glob: {pattern}"
    elif tool_name_lower in ("semanticsearch", "semantic_search"):
        query = tool_input.get("query", "") if isinstance(tool_input, dict) else ""
        name = f"search: {query[:30]}..." if len(query) > 30 else f"search: {query}"
    elif tool_name_lower == "read":
        path = tool_input.get("path", "") if isinstance(tool_input, dict) else ""
        file_name = Path(path).name if path else "file"
        name = f"read: {file_name}"
    elif tool_name_lower == "write":
        path = tool_input.get("path", "") if isinstance(tool_input, dict) else ""
        file_name = Path(path).name if path else "file"
        name = f"write: {file_name}"
    elif tool_name_lower == "task":
        description = tool_input.get("description", "") if isinstance(tool_input, dict) else ""
        name = f"task: {description[:30]}..." if len(description) > 30 else f"task: {description}"
    elif tool_name_lower == "ls":
        path = tool_input.get("target_directory", "") if isinstance(tool_input, dict) else ""
        name = f"ls: {Path(path).name if path else 'dir'}"
    else:
        name = f"tool: {tool_name}"

    # Prepare inputs
    inputs = {"tool_name": tool_name}
    if tool_input:
        inputs["input"] = tool_input

    # Prepare outputs
    outputs = {}
    if tool_output:
        outputs["output"] = tool_output

    attributes = {
        "generation_id": event.get("generation_id", ""),
        "tool_name": tool_name,
    }

    if duration:
        attributes["duration_ms"] = duration

    span = mlflow.start_span_no_context(
        name=name,
        parent_span=parent_span,
        span_type=SpanType.TOOL,
        start_time_ns=start_ns,
        inputs=inputs,
        attributes=attributes,
    )
    if outputs:
        span.set_outputs(outputs)
    span.end(end_time_ns=end_ns)


def _calculate_edit_stats(edits: list[dict[str, Any]]) -> dict[str, int]:
    """Calculate edit statistics from edit operations."""
    if not edits:
        return {"edit_count": 0, "lines_added": 0, "lines_removed": 0, "net_change": 0}

    lines_added = 0
    lines_removed = 0

    for edit in edits:
        old_string = edit.get("old_string", "")
        new_string = edit.get("new_string", "")

        old_lines = old_string.count("\n") + 1 if old_string else 0
        new_lines = new_string.count("\n") + 1 if new_string else 0

        if new_lines > old_lines:
            lines_added += new_lines - old_lines
        else:
            lines_removed += old_lines - new_lines

    return {
        "edit_count": len(edits),
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "net_change": lines_added - lines_removed,
    }


class CursorTraceManager:
    """Manages trace creation for Cursor conversations.

    This manager collects events during the conversation and creates
    a single trace when the stop hook fires.
    """

    _instance: "CursorTraceManager | None" = None

    def __new__(cls) -> "CursorTraceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "CursorTraceManager":
        """Get the singleton instance of CursorTraceManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record_event(
        self,
        conversation_id: str,
        event_type: str,
        input_data: dict[str, Any],
    ) -> None:
        """Record an event for a conversation.

        Args:
            conversation_id: Unique identifier for the conversation
            event_type: Type of hook event
            input_data: Full hook input data
        """
        event = {
            "event_type": event_type,
            "conversation_id": conversation_id,
            **input_data,
        }
        append_event(conversation_id, event)

        get_logger().log(
            CURSOR_TRACING_LEVEL,
            "Recorded event %s for conversation %s",
            event_type,
            conversation_id,
        )

    def finalize_trace(
        self,
        conversation_id: str,
        status: str = "completed",
    ) -> mlflow.entities.Trace | None:
        """Finalize and create the trace for a conversation.

        Called when the stop hook fires. Reads all collected events
        and creates a single trace with nested spans.

        Args:
            conversation_id: Unique identifier for the conversation
            status: Final status of the agent

        Returns:
            The created trace object, or None if failed
        """
        # Read all events
        events = read_events(conversation_id)

        if not events:
            get_logger().warning(
                "No events found for conversation %s, cannot create trace",
                conversation_id,
            )
            return None

        # Create the trace
        trace = create_trace_from_events(conversation_id, events, status)

        # Cleanup events file
        cleanup_events(conversation_id)

        return trace
