"""Hook management for Claude Code integration with MLflow."""

from pathlib import Path
from typing import Any

from mlflow.claude_code.config import (
    HOOK_FIELD_COMMAND,
    HOOK_FIELD_HOOKS,
    MLFLOW_HOOK_IDENTIFIER,
    load_claude_config,
    save_claude_config,
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
                filtered_hooks = []
                for hook in group[HOOK_FIELD_HOOKS]:
                    if MLFLOW_HOOK_IDENTIFIER not in hook.get(HOOK_FIELD_COMMAND, ""):
                        filtered_hooks.append(hook)

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
    from mlflow.claude_code.config import (
        ENVIRONMENT_FIELD,
        MLFLOW_EXPERIMENT_ID,
        MLFLOW_EXPERIMENT_NAME,
        MLFLOW_TRACING_ENABLED,
        MLFLOW_TRACKING_URI,
    )

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


def stop_hook_handler() -> None:
    """Hook handler for conversation end - processes transcript and creates trace."""
    from mlflow.claude_code.tracing import (
        get_logger,
        is_tracing_enabled,
        output_hook_response,
        process_transcript,
        read_hook_input,
    )

    if not is_tracing_enabled():
        output_hook_response()
        return

    try:
        hook_data = read_hook_input()
        session_id = hook_data.get("session_id")
        transcript_path = hook_data.get("transcript_path")

        get_logger().info("Stop hook: session=%s, transcript=%s", session_id, transcript_path)

        # Process the transcript and create MLflow trace
        trace = process_transcript(transcript_path, session_id)

        if trace is not None:
            output_hook_response()
        else:
            output_hook_response(
                error=(
                    "Failed to process transcript, please check .claude/mlflow/claude_tracing.log"
                    " for more details"
                ),
            )

    except Exception as e:
        import sys  # clint: disable=lazy-builtin-import

        get_logger().error("Error in Stop hook: %s", e, exc_info=True)
        output_hook_response(error=str(e))
        sys.exit(1)
