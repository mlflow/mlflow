"""Hook management for Gemini CLI integration with MLflow."""

import json
import sys
from pathlib import Path
from typing import Any

from mlflow.gemini_cli.config import (
    HOOK_FIELD_COMMAND,
    HOOK_FIELD_HOOKS,
    MLFLOW_HOOK_IDENTIFIER,
    load_gemini_config,
    save_gemini_config,
)
from mlflow.gemini_cli.tracing import (
    GEMINI_TRACING_LEVEL,
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
        hook_type: The hook type (e.g., 'SessionEnd')
        handler_name: The handler function name (e.g., 'session_end_hook_handler')
    """
    if hook_type not in config[HOOK_FIELD_HOOKS]:
        config[HOOK_FIELD_HOOKS][hook_type] = []

    hook_command = (
        f'python -c "from mlflow.gemini_cli.hooks import {handler_name}; {handler_name}()"'
    )

    mlflow_hook = {
        "type": "command",
        "name": "mlflow-tracing",
        HOOK_FIELD_COMMAND: hook_command,
        "description": "MLflow tracing hook for Gemini CLI",
    }

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
    """Set up Gemini CLI hooks for MLflow tracing.

    Creates or updates SessionEnd hook that calls MLflow tracing handler.
    Updates existing MLflow hooks if found, otherwise adds new ones.

    Args:
        settings_path: Path to Gemini settings.json file
    """
    config = load_gemini_config(settings_path)

    if HOOK_FIELD_HOOKS not in config:
        config[HOOK_FIELD_HOOKS] = {}

    upsert_hook(config, "SessionEnd", "session_end_hook_handler")

    save_gemini_config(settings_path, config)


# ============================================================================
# HOOK REMOVAL AND CLEANUP
# ============================================================================


def disable_tracing_hooks(settings_path: Path) -> bool:
    """Remove MLflow hooks from Gemini CLI settings.

    Args:
        settings_path: Path to Gemini settings file

    Returns:
        True if hooks were removed, False if no configuration was found
    """
    if not settings_path.exists():
        return False

    config = load_gemini_config(settings_path)
    hooks_removed = False

    # Remove MLflow hooks from SessionEnd
    if "SessionEnd" in config.get(HOOK_FIELD_HOOKS, {}):
        hook_groups = config[HOOK_FIELD_HOOKS]["SessionEnd"]
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
            config[HOOK_FIELD_HOOKS]["SessionEnd"] = filtered_groups
        else:
            del config[HOOK_FIELD_HOOKS]["SessionEnd"]
            hooks_removed = True

    # Clean up empty hooks section
    if HOOK_FIELD_HOOKS in config and not config[HOOK_FIELD_HOOKS]:
        del config[HOOK_FIELD_HOOKS]

    # Save updated config or remove file if empty
    if config:
        save_gemini_config(settings_path, config)
    else:
        settings_path.unlink()

    return hooks_removed


# ============================================================================
# GEMINI CLI HOOK HANDLERS
# ============================================================================


def _process_session_end_hook(
    session_id: str | None, transcript_path: str | None
) -> dict[str, Any]:
    """Common logic for processing session end hooks.

    Args:
        session_id: Session identifier
        transcript_path: Path to transcript file

    Returns:
        Hook response dictionary
    """
    get_logger().log(
        GEMINI_TRACING_LEVEL,
        "SessionEnd hook: session=%s, transcript=%s",
        session_id,
        transcript_path,
    )

    # Process the transcript and create MLflow trace
    trace = process_transcript(transcript_path, session_id)

    if trace is not None:
        return get_hook_response()
    return get_hook_response(
        error=(
            "Failed to process transcript, please check .gemini/mlflow/gemini_tracing.log"
            " for more details"
        ),
    )


def session_end_hook_handler() -> None:
    """CLI hook handler for session end - processes transcript and creates trace.

    This handler is called by Gemini CLI via the SessionEnd hook.
    It reads hook input from stdin, processes the transcript, and creates an MLflow trace.
    """
    if not is_tracing_enabled():
        response = get_hook_response()
        print(json.dumps(response))  # noqa: T201
        return

    try:
        hook_data = read_hook_input()
        session_id = hook_data.get("session_id")
        transcript_path = hook_data.get("transcript_path")

        setup_mlflow()
        response = _process_session_end_hook(session_id, transcript_path)
        print(json.dumps(response))  # noqa: T201

    except Exception as e:
        get_logger().error("Error in SessionEnd hook: %s", e, exc_info=True)
        response = get_hook_response(error=str(e))
        print(json.dumps(response))  # noqa: T201
        sys.exit(1)
