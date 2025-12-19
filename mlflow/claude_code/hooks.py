"""Hook management for Claude Code integration with MLflow."""

import json
import sys
from pathlib import Path
from typing import Any

from mlflow.claude_code.config import (
    ENVIRONMENT_FIELD,
    HOOK_FIELD_COMMAND,
    HOOK_FIELD_HOOKS,
    MLFLOW_EXPERIMENT_ID,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_HOOK_IDENTIFIER,
    MLFLOW_TRACING_ENABLED,
    MLFLOW_TRACKING_URI,
    load_claude_config,
    save_claude_config,
)
from mlflow.claude_code.tracing import (
    CLAUDE_TRACING_LEVEL,
    get_hook_response,
    get_logger,
    is_tracing_enabled,
    process_transcript,
    read_hook_input,
    setup_mlflow,
)

# ============================================================================
# HOOK CONFIGURATION UTILITIES
# ============================================================================


def upsert_hook(config: dict[str, Any], hook_type: str, handler_name: str) -> None:
    """Insert or update a single MLflow hook in the configuration.

    Args:
        config: The hooks configuration dictionary to modify
        hook_type: The hook type (e.g., 'PostToolUse', 'Stop')
        handler_name: The handler function name (e.g., 'post_tool_use_handler')
    """
    if hook_type not in config[HOOK_FIELD_HOOKS]:
        config[HOOK_FIELD_HOOKS][hook_type] = []

    hook_command = (
        f'python -c "from mlflow.claude_code.hooks import {handler_name}; {handler_name}()"'
    )

    mlflow_hook = {"type": "command", HOOK_FIELD_COMMAND: hook_command}

    # Check if MLflow hook already exists and update it
    hook_exists = False
    for hook_group in config[HOOK_FIELD_HOOKS][hook_type]:
        if HOOK_FIELD_HOOKS in hook_group:
            for hook in hook_group[HOOK_FIELD_HOOKS]:
                if MLFLOW_HOOK_IDENTIFIER in hook.get(HOOK_FIELD_COMMAND, ""):
                    hook.update(mlflow_hook)
                    hook_exists = True
                    break

    # Add new hook if it doesn't exist
    if not hook_exists:
        config[HOOK_FIELD_HOOKS][hook_type].append({HOOK_FIELD_HOOKS: [mlflow_hook]})


def setup_hooks_config(settings_path: Path) -> None:
    """Set up Claude Code hooks for MLflow tracing.

    Creates or updates Stop hook that calls MLflow tracing handler.
    Updates existing MLflow hooks if found, otherwise adds new ones.

    Args:
        settings_path: Path to Claude settings.json file
    """
    config = load_claude_config(settings_path)

    if HOOK_FIELD_HOOKS not in config:
        config[HOOK_FIELD_HOOKS] = {}

    upsert_hook(config, "Stop", "stop_hook_handler")

    save_claude_config(settings_path, config)


# ============================================================================
# HOOK REMOVAL AND CLEANUP
# ============================================================================


def disable_tracing_hooks(settings_path: Path) -> bool:
    """Remove MLflow hooks and environment variables from Claude settings.

    Args:
        settings_path: Path to Claude settings file

    Returns:
        True if hooks/config were removed, False if no configuration was found
    """
    if not settings_path.exists():
        return False

    config = load_claude_config(settings_path)
    hooks_removed = False
    env_removed = False

    # Remove MLflow hooks
    if "Stop" in config.get(HOOK_FIELD_HOOKS, {}):
        hook_groups = config[HOOK_FIELD_HOOKS]["Stop"]
        filtered_groups = []

        for group in hook_groups:
            if HOOK_FIELD_HOOKS in group:
                filtered_hooks = [
                    hook
                    for hook in group[HOOK_FIELD_HOOKS]
                    if MLFLOW_HOOK_IDENTIFIER not in hook.get(HOOK_FIELD_COMMAND, "")
                ]

                if filtered_hooks:
                    filtered_groups.append({HOOK_FIELD_HOOKS: filtered_hooks})
                else:
                    hooks_removed = True
            else:
                filtered_groups.append(group)

        if filtered_groups:
            config[HOOK_FIELD_HOOKS]["Stop"] = filtered_groups
        else:
            del config[HOOK_FIELD_HOOKS]["Stop"]
            hooks_removed = True

    # Remove config variables
    if ENVIRONMENT_FIELD in config:
        mlflow_vars = [
            MLFLOW_TRACING_ENABLED,
            MLFLOW_TRACKING_URI,
            MLFLOW_EXPERIMENT_ID,
            MLFLOW_EXPERIMENT_NAME,
        ]
        for var in mlflow_vars:
            if var in config[ENVIRONMENT_FIELD]:
                del config[ENVIRONMENT_FIELD][var]
                env_removed = True

        if not config[ENVIRONMENT_FIELD]:
            del config[ENVIRONMENT_FIELD]

    # Clean up empty hooks section
    if HOOK_FIELD_HOOKS in config and not config[HOOK_FIELD_HOOKS]:
        del config[HOOK_FIELD_HOOKS]

    # Save updated config or remove file if empty
    if config:
        save_claude_config(settings_path, config)
    else:
        settings_path.unlink()

    return hooks_removed or env_removed


# ============================================================================
# CLAUDE CODE HOOK HANDLERS
# ============================================================================


def _process_stop_hook(session_id: str | None, transcript_path: str | None) -> dict[str, Any]:
    """Common logic for processing stop hooks.

    Args:
        session_id: Session identifier
        transcript_path: Path to transcript file

    Returns:
        Hook response dictionary
    """
    get_logger().log(
        CLAUDE_TRACING_LEVEL, "Stop hook: session=%s, transcript=%s", session_id, transcript_path
    )

    # Process the transcript and create MLflow trace
    trace = process_transcript(transcript_path, session_id)

    if trace is not None:
        return get_hook_response()
    return get_hook_response(
        error=(
            "Failed to process transcript, please check .claude/mlflow/claude_tracing.log"
            " for more details"
        ),
    )


def stop_hook_handler() -> None:
    """CLI hook handler for conversation end - processes transcript and creates trace."""
    if not is_tracing_enabled():
        response = get_hook_response()
        print(json.dumps(response))  # noqa: T201
        return

    try:
        hook_data = read_hook_input()
        session_id = hook_data.get("session_id")
        transcript_path = hook_data.get("transcript_path")

        setup_mlflow()
        response = _process_stop_hook(session_id, transcript_path)
        print(json.dumps(response))  # noqa: T201

    except Exception as e:
        get_logger().error("Error in Stop hook: %s", e, exc_info=True)
        response = get_hook_response(error=str(e))
        print(json.dumps(response))  # noqa: T201
        sys.exit(1)


async def sdk_stop_hook_handler(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: Any,
) -> dict[str, Any]:
    """SDK hook handler for Stop event - processes transcript and creates trace.

    Args:
        input_data: Dictionary containing session_id and transcript_path
        tool_use_id: Tool use identifier
        context: HookContext from the SDK
    """
    from mlflow.utils.autologging_utils import autologging_is_disabled

    # Check if autologging is disabled
    if autologging_is_disabled("anthropic"):
        return get_hook_response()

    try:
        session_id = input_data.get("session_id")
        transcript_path = input_data.get("transcript_path")

        return _process_stop_hook(session_id, transcript_path)

    except Exception as e:
        get_logger().error("Error in SDK Stop hook: %s", e, exc_info=True)
        return get_hook_response(error=str(e))
