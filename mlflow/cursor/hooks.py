"""Hook management for Cursor integration with MLflow."""

import json
from pathlib import Path
from typing import Any

from mlflow.cursor.config import (
    HOOK_FIELD_COMMAND,
    HOOK_FIELD_HOOKS,
    HOOK_FIELD_VERSION,
    MLFLOW_HOOK_IDENTIFIER,
    is_tracing_enabled,
    load_cursor_config,
    save_cursor_config,
)
from mlflow.cursor.tracing import (
    CURSOR_TRACING_LEVEL,
    CursorTraceManager,
    get_hook_response,
    get_logger,
    read_hook_input,
    setup_mlflow,
)


def get_hook_command(handler_name: str) -> str:
    """Generate the command string for a hook handler.

    Args:
        handler_name: The handler function name

    Returns:
        Command string to execute the handler
    """
    # Use python3 for better cross-platform compatibility (python may not exist on macOS)
    return f'python3 -c "from mlflow.cursor.hooks import {handler_name}; {handler_name}()"'


def upsert_hook(config: dict[str, Any], hook_type: str, handler_name: str) -> None:
    """Insert or update a single MLflow hook in the configuration.

    Args:
        config: The hooks configuration dictionary to modify
        hook_type: The hook type (e.g., 'beforeSubmitPrompt', 'stop')
        handler_name: The handler function name
    """
    if HOOK_FIELD_HOOKS not in config:
        config[HOOK_FIELD_HOOKS] = {}

    if hook_type not in config[HOOK_FIELD_HOOKS]:
        config[HOOK_FIELD_HOOKS][hook_type] = []

    hook_command = get_hook_command(handler_name)
    mlflow_hook = {HOOK_FIELD_COMMAND: hook_command}

    # Check if MLflow hook already exists and update it
    hook_exists = False
    for i, hook_entry in enumerate(config[HOOK_FIELD_HOOKS][hook_type]):
        if isinstance(hook_entry, dict):
            command = hook_entry.get(HOOK_FIELD_COMMAND, "")
            if MLFLOW_HOOK_IDENTIFIER in command:
                config[HOOK_FIELD_HOOKS][hook_type][i] = mlflow_hook
                hook_exists = True
                break

    # Add new hook if it doesn't exist
    if not hook_exists:
        config[HOOK_FIELD_HOOKS][hook_type].append(mlflow_hook)


def setup_hooks_config(hooks_path: Path) -> None:
    """Set up Cursor hooks for MLflow tracing.

    Creates or updates all hook configurations that call MLflow tracing handlers.

    Args:
        hooks_path: Path to Cursor hooks.json file
    """
    config = load_cursor_config(hooks_path)

    if HOOK_FIELD_VERSION not in config:
        config[HOOK_FIELD_VERSION] = 1

    if HOOK_FIELD_HOOKS not in config:
        config[HOOK_FIELD_HOOKS] = {}

    # Map hook types to their handlers
    # Note: Tab hooks (beforeTabFileRead, afterTabFileEdit) are intentionally excluded
    # as they fire too frequently and can interfere with Tab completion performance
    hook_handlers = {
        "beforeSubmitPrompt": "handle_before_submit_prompt",
        "afterAgentResponse": "handle_after_agent_response",
        "afterAgentThought": "handle_after_agent_thought",
        "beforeShellExecution": "handle_before_shell_execution",
        "afterShellExecution": "handle_after_shell_execution",
        "beforeMCPExecution": "handle_before_mcp_execution",
        "afterMCPExecution": "handle_after_mcp_execution",
        "beforeReadFile": "handle_before_read_file",
        "afterFileEdit": "handle_after_file_edit",
        "stop": "handle_stop",
        # Tool use hooks - capture Grep, Read, Write, Task, Glob, etc.
        "preToolUse": "handle_pre_tool_use",
        "postToolUse": "handle_post_tool_use",
    }

    for hook_type, handler_name in hook_handlers.items():
        upsert_hook(config, hook_type, handler_name)

    save_cursor_config(hooks_path, config)


def disable_tracing_hooks(hooks_path: Path, env_file: Path) -> bool:
    """Remove MLflow hooks from Cursor configuration.

    Args:
        hooks_path: Path to Cursor hooks.json file
        env_file: Path to .env file

    Returns:
        True if hooks/config were removed, False if no configuration was found
    """
    hooks_removed = False
    env_removed = False

    # Remove MLflow hooks from hooks.json
    if hooks_path.exists():
        config = load_cursor_config(hooks_path)
        hooks = config.get(HOOK_FIELD_HOOKS, {})

        for hook_type in list(hooks.keys()):
            filtered_hooks = [
                hook
                for hook in hooks[hook_type]
                if not (
                    isinstance(hook, dict)
                    and MLFLOW_HOOK_IDENTIFIER in hook.get(HOOK_FIELD_COMMAND, "")
                )
            ]

            if len(filtered_hooks) != len(hooks[hook_type]):
                hooks_removed = True

            if filtered_hooks:
                hooks[hook_type] = filtered_hooks
            else:
                del hooks[hook_type]

        # Clean up empty hooks section
        if not hooks:
            config.pop(HOOK_FIELD_HOOKS, None)

        # Save updated config or remove file if empty
        if config and config != {HOOK_FIELD_VERSION: 1}:
            save_cursor_config(hooks_path, config)
        elif hooks_path.exists():
            hooks_path.unlink()

    # Remove MLflow environment variables from .env
    if env_file.exists():
        try:
            lines_to_keep = []
            mlflow_vars = [
                "MLFLOW_CURSOR_TRACING_ENABLED",
                "MLFLOW_TRACKING_URI",
                "MLFLOW_EXPERIMENT_ID",
                "MLFLOW_EXPERIMENT_NAME",
            ]

            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("#") and "MLflow" in stripped:
                        env_removed = True
                        continue
                    if stripped and "=" in stripped:
                        key = stripped.split("=", 1)[0].strip()
                        if key in mlflow_vars:
                            env_removed = True
                            continue
                    lines_to_keep.append(line)

            # Remove empty or comments-only .env file
            non_empty_lines = [
                line for line in lines_to_keep if line.strip() and not line.strip().startswith("#")
            ]
            if non_empty_lines:
                with open(env_file, "w", encoding="utf-8") as f:
                    f.writelines(lines_to_keep)
            else:
                env_file.unlink()
        except IOError:
            pass

    return hooks_removed or env_removed


def _run_hook_handler(handler_func) -> None:
    """Common wrapper for running hook handlers.

    Args:
        handler_func: The handler function to execute
    """
    if not is_tracing_enabled():
        response = get_hook_response()
        print(json.dumps(response))  # noqa: T201
        return

    try:
        hook_data = read_hook_input()
        setup_mlflow()
        response = handler_func(hook_data)
        print(json.dumps(response))  # noqa: T201

    except Exception as e:
        get_logger().error("Error in hook handler: %s", e, exc_info=True)
        # Return permissive response so Cursor can continue
        response = get_hook_response(error=str(e))
        print(json.dumps(response))  # noqa: T201


def _process_before_submit_prompt(input_data: dict[str, Any]) -> dict[str, Any]:
    """Process beforeSubmitPrompt hook - captures user prompts."""
    conversation_id = input_data.get("conversation_id", "")
    prompt = input_data.get("prompt", "")
    model = input_data.get("model", "unknown")

    get_logger().log(
        CURSOR_TRACING_LEVEL,
        "beforeSubmitPrompt: conversation=%s, model=%s, prompt_length=%d",
        conversation_id,
        model,
        len(prompt),
    )

    trace_manager = CursorTraceManager.get_instance()
    trace_manager.record_event(
        conversation_id=conversation_id,
        event_type="beforeSubmitPrompt",
        input_data=input_data,
    )

    return get_hook_response()


def handle_before_submit_prompt() -> None:
    """CLI hook handler for beforeSubmitPrompt."""
    _run_hook_handler(_process_before_submit_prompt)


def _process_after_agent_response(input_data: dict[str, Any]) -> dict[str, Any]:
    """Process afterAgentResponse hook - records agent responses."""
    conversation_id = input_data.get("conversation_id", "")
    text = input_data.get("text", "")

    get_logger().log(
        CURSOR_TRACING_LEVEL,
        "afterAgentResponse: conversation=%s, response_length=%d",
        conversation_id,
        len(text),
    )

    trace_manager = CursorTraceManager.get_instance()
    trace_manager.record_event(
        conversation_id=conversation_id,
        event_type="afterAgentResponse",
        input_data=input_data,
    )

    return get_hook_response()


def handle_after_agent_response() -> None:
    """CLI hook handler for afterAgentResponse."""
    _run_hook_handler(_process_after_agent_response)


def _process_after_agent_thought(input_data: dict[str, Any]) -> dict[str, Any]:
    """Process afterAgentThought hook - logs agent thinking."""
    conversation_id = input_data.get("conversation_id", "")
    text = input_data.get("text", "")
    duration_ms = input_data.get("duration_ms", 0)

    get_logger().log(
        CURSOR_TRACING_LEVEL,
        "afterAgentThought: conversation=%s, thinking_length=%d, duration=%dms",
        conversation_id,
        len(text),
        duration_ms,
    )

    trace_manager = CursorTraceManager.get_instance()
    trace_manager.record_event(
        conversation_id=conversation_id,
        event_type="afterAgentThought",
        input_data=input_data,
    )

    return get_hook_response()


def handle_after_agent_thought() -> None:
    """CLI hook handler for afterAgentThought."""
    _run_hook_handler(_process_after_agent_thought)


def _process_before_shell_execution(input_data: dict[str, Any]) -> dict[str, Any]:
    """Process beforeShellExecution hook - tracks shell commands before execution."""
    conversation_id = input_data.get("conversation_id", "")
    command = input_data.get("command", "")

    get_logger().log(
        CURSOR_TRACING_LEVEL,
        "beforeShellExecution: conversation=%s, command=%s",
        conversation_id,
        command[:50] if command else "",
    )

    trace_manager = CursorTraceManager.get_instance()
    trace_manager.record_event(
        conversation_id=conversation_id,
        event_type="beforeShellExecution",
        input_data=input_data,
    )

    return get_hook_response()


def handle_before_shell_execution() -> None:
    """CLI hook handler for beforeShellExecution."""
    _run_hook_handler(_process_before_shell_execution)


def _process_after_shell_execution(input_data: dict[str, Any]) -> dict[str, Any]:
    """Process afterShellExecution hook - captures shell command output."""
    conversation_id = input_data.get("conversation_id", "")
    command = input_data.get("command", "")
    duration = input_data.get("duration", 0)

    get_logger().log(
        CURSOR_TRACING_LEVEL,
        "afterShellExecution: conversation=%s, command=%s, duration=%dms",
        conversation_id,
        command[:50] if command else "",
        duration,
    )

    trace_manager = CursorTraceManager.get_instance()
    trace_manager.record_event(
        conversation_id=conversation_id,
        event_type="afterShellExecution",
        input_data=input_data,
    )

    return get_hook_response()


def handle_after_shell_execution() -> None:
    """CLI hook handler for afterShellExecution."""
    _run_hook_handler(_process_after_shell_execution)


def _process_before_mcp_execution(input_data: dict[str, Any]) -> dict[str, Any]:
    """Process beforeMCPExecution hook - logs MCP tool calls."""
    conversation_id = input_data.get("conversation_id", "")
    tool_name = input_data.get("tool_name", "")

    get_logger().log(
        CURSOR_TRACING_LEVEL,
        "beforeMCPExecution: conversation=%s, tool=%s",
        conversation_id,
        tool_name,
    )

    trace_manager = CursorTraceManager.get_instance()
    trace_manager.record_event(
        conversation_id=conversation_id,
        event_type="beforeMCPExecution",
        input_data=input_data,
    )

    return get_hook_response()


def handle_before_mcp_execution() -> None:
    """CLI hook handler for beforeMCPExecution."""
    _run_hook_handler(_process_before_mcp_execution)


def _process_after_mcp_execution(input_data: dict[str, Any]) -> dict[str, Any]:
    """Process afterMCPExecution hook - records MCP tool results."""
    conversation_id = input_data.get("conversation_id", "")
    tool_name = input_data.get("tool_name", "")
    duration = input_data.get("duration", 0)

    get_logger().log(
        CURSOR_TRACING_LEVEL,
        "afterMCPExecution: conversation=%s, tool=%s, duration=%dms",
        conversation_id,
        tool_name,
        duration,
    )

    trace_manager = CursorTraceManager.get_instance()
    trace_manager.record_event(
        conversation_id=conversation_id,
        event_type="afterMCPExecution",
        input_data=input_data,
    )

    return get_hook_response()


def handle_after_mcp_execution() -> None:
    """CLI hook handler for afterMCPExecution."""
    _run_hook_handler(_process_after_mcp_execution)


def _process_before_read_file(input_data: dict[str, Any]) -> dict[str, Any]:
    """Process beforeReadFile hook - tracks file read operations."""
    conversation_id = input_data.get("conversation_id", "")
    file_path = input_data.get("file_path", "")

    get_logger().log(
        CURSOR_TRACING_LEVEL,
        "beforeReadFile: conversation=%s, file=%s",
        conversation_id,
        file_path,
    )

    trace_manager = CursorTraceManager.get_instance()
    trace_manager.record_event(
        conversation_id=conversation_id,
        event_type="beforeReadFile",
        input_data=input_data,
    )

    return get_hook_response()


def handle_before_read_file() -> None:
    """CLI hook handler for beforeReadFile."""
    _run_hook_handler(_process_before_read_file)


def _process_after_file_edit(input_data: dict[str, Any]) -> dict[str, Any]:
    """Process afterFileEdit hook - captures file edits with statistics."""
    conversation_id = input_data.get("conversation_id", "")
    file_path = input_data.get("file_path", "")
    edits = input_data.get("edits", [])

    get_logger().log(
        CURSOR_TRACING_LEVEL,
        "afterFileEdit: conversation=%s, file=%s, edits=%d",
        conversation_id,
        file_path,
        len(edits),
    )

    trace_manager = CursorTraceManager.get_instance()
    trace_manager.record_event(
        conversation_id=conversation_id,
        event_type="afterFileEdit",
        input_data=input_data,
    )

    return get_hook_response()


def handle_after_file_edit() -> None:
    """CLI hook handler for afterFileEdit."""
    _run_hook_handler(_process_after_file_edit)


def _process_stop(input_data: dict[str, Any]) -> dict[str, Any]:
    """Process stop hook - creates the trace from all collected events.

    This is the main trace creation point. When the agent stops,
    we read all collected events and create a single trace with
    nested spans for the entire conversation.
    """
    conversation_id = input_data.get("conversation_id", "")
    status = input_data.get("status", "completed")
    loop_count = input_data.get("loop_count", 0)

    get_logger().log(
        CURSOR_TRACING_LEVEL,
        "stop: conversation=%s, status=%s, loop_count=%d",
        conversation_id,
        status,
        loop_count,
    )

    # Record the stop event
    trace_manager = CursorTraceManager.get_instance()
    trace_manager.record_event(
        conversation_id=conversation_id,
        event_type="stop",
        input_data=input_data,
    )

    # Create the trace from all collected events
    trace = trace_manager.finalize_trace(
        conversation_id=conversation_id,
        status=status,
    )

    if trace is not None:
        get_logger().log(
            CURSOR_TRACING_LEVEL,
            "Created trace %s for conversation %s",
            trace.info.request_id,
            conversation_id,
        )
        return get_hook_response()

    return get_hook_response(
        error="Failed to create trace, check .cursor/mlflow/cursor_tracing.log for details"
    )


def handle_stop() -> None:
    """CLI hook handler for stop."""
    _run_hook_handler(_process_stop)


def _process_before_tab_file_read(input_data: dict[str, Any]) -> dict[str, Any]:
    """Process beforeTabFileRead hook - Tab mode file reads."""
    conversation_id = input_data.get("conversation_id", "")
    file_path = input_data.get("file_path", "")

    get_logger().log(
        CURSOR_TRACING_LEVEL,
        "beforeTabFileRead: conversation=%s, file=%s",
        conversation_id,
        file_path,
    )

    trace_manager = CursorTraceManager.get_instance()
    trace_manager.record_event(
        conversation_id=conversation_id,
        event_type="beforeTabFileRead",
        input_data=input_data,
    )

    return get_hook_response()


def handle_before_tab_file_read() -> None:
    """CLI hook handler for beforeTabFileRead."""
    _run_hook_handler(_process_before_tab_file_read)


def _process_after_tab_file_edit(input_data: dict[str, Any]) -> dict[str, Any]:
    """Process afterTabFileEdit hook - Tab mode file edits."""
    conversation_id = input_data.get("conversation_id", "")
    file_path = input_data.get("file_path", "")
    edits = input_data.get("edits", [])

    get_logger().log(
        CURSOR_TRACING_LEVEL,
        "afterTabFileEdit: conversation=%s, file=%s, edits=%d",
        conversation_id,
        file_path,
        len(edits),
    )

    trace_manager = CursorTraceManager.get_instance()
    trace_manager.record_event(
        conversation_id=conversation_id,
        event_type="afterTabFileEdit",
        input_data=input_data,
    )

    return get_hook_response()


def handle_after_tab_file_edit() -> None:
    """CLI hook handler for afterTabFileEdit."""
    _run_hook_handler(_process_after_tab_file_edit)


def _process_pre_tool_use(input_data: dict[str, Any]) -> dict[str, Any]:
    """Process preToolUse hook - captures tool calls before execution.

    This hook captures ALL tool usage including:
    - Grep: Code search
    - Glob: File pattern matching
    - Read: File reading
    - Write: File writing
    - Task: Subagent tasks
    - Shell: Command execution
    - SemanticSearch: Semantic code search
    """
    conversation_id = input_data.get("conversation_id", "")
    tool_name = input_data.get("tool_name", "") or input_data.get("name", "unknown")
    tool_input = input_data.get("tool_input", {}) or input_data.get("input", {})

    get_logger().log(
        CURSOR_TRACING_LEVEL,
        "preToolUse: conversation=%s, tool=%s",
        conversation_id,
        tool_name,
    )

    trace_manager = CursorTraceManager.get_instance()
    trace_manager.record_event(
        conversation_id=conversation_id,
        event_type="preToolUse",
        input_data={
            **input_data,
            "tool_name": tool_name,
            "tool_input": tool_input,
        },
    )

    return get_hook_response()


def handle_pre_tool_use() -> None:
    """CLI hook handler for preToolUse."""
    _run_hook_handler(_process_pre_tool_use)


def _process_post_tool_use(input_data: dict[str, Any]) -> dict[str, Any]:
    """Process postToolUse hook - captures tool results after execution.

    This hook captures the results of ALL tool executions.
    """
    conversation_id = input_data.get("conversation_id", "")
    tool_name = input_data.get("tool_name", "") or input_data.get("name", "unknown")
    tool_output = input_data.get("tool_output", {}) or input_data.get("output", {})
    duration = input_data.get("duration", 0) or input_data.get("duration_ms", 0)

    get_logger().log(
        CURSOR_TRACING_LEVEL,
        "postToolUse: conversation=%s, tool=%s, duration=%dms",
        conversation_id,
        tool_name,
        duration,
    )

    trace_manager = CursorTraceManager.get_instance()
    trace_manager.record_event(
        conversation_id=conversation_id,
        event_type="postToolUse",
        input_data={
            **input_data,
            "tool_name": tool_name,
            "tool_output": tool_output,
            "duration": duration,
        },
    )

    return get_hook_response()


def handle_post_tool_use() -> None:
    """CLI hook handler for postToolUse."""
    _run_hook_handler(_process_post_tool_use)
