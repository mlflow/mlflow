"""Hook management for Qwen Code integration with MLflow."""

import json
import os
import sys
from pathlib import Path
from typing import Any

from mlflow.qwen_code.config import (
    ENVIRONMENT_FIELD,
    HOOK_FIELD_COMMAND,
    HOOK_FIELD_HOOKS,
    MLFLOW_HOOK_IDENTIFIER,
    MLFLOW_TRACING_ENABLED,
    QWEN_SETTINGS_FILE,
    load_qwen_config,
    save_qwen_config,
)
from mlflow.qwen_code.tracing import (
    QWEN_TRACING_LEVEL,
    get_hook_response,
    get_logger,
    process_transcript,
    read_hook_input,
    setup_mlflow,
)


def upsert_hook(config: dict[str, Any], hook_type: str, subcommand: str) -> None:
    """Insert or update a single MLflow hook in the Qwen settings configuration."""
    if HOOK_FIELD_HOOKS not in config:
        config[HOOK_FIELD_HOOKS] = {}

    if hook_type not in config[HOOK_FIELD_HOOKS]:
        config[HOOK_FIELD_HOOKS][hook_type] = []

    mlflow_cmd = "uv run mlflow" if "UV" in os.environ else "mlflow"
    hook_command = f"{mlflow_cmd} autolog qwen-code {subcommand}"
    mlflow_hook = {"type": "command", HOOK_FIELD_COMMAND: hook_command}

    hook_exists = False
    for hook_group in config[HOOK_FIELD_HOOKS][hook_type]:
        if HOOK_FIELD_HOOKS in hook_group:
            for hook in hook_group[HOOK_FIELD_HOOKS]:
                if MLFLOW_HOOK_IDENTIFIER in hook.get(HOOK_FIELD_COMMAND, ""):
                    hook.update(mlflow_hook)
                    hook_exists = True
                    break

    if not hook_exists:
        config[HOOK_FIELD_HOOKS][hook_type].append({HOOK_FIELD_HOOKS: [mlflow_hook]})


def setup_hooks_config(qwen_dir: Path) -> None:
    """Set up Qwen Code hooks for MLflow tracing in .qwen/settings.json."""
    settings_path = qwen_dir / QWEN_SETTINGS_FILE
    config = load_qwen_config(settings_path)
    upsert_hook(config, "Stop", "stop-hook")
    save_qwen_config(settings_path, config)


def disable_tracing_hooks(qwen_dir: Path) -> bool:
    """Remove MLflow hooks and environment variables from Qwen settings.

    Returns:
        True if hooks/config were removed, False if none found
    """
    settings_path = qwen_dir / QWEN_SETTINGS_FILE
    if not settings_path.exists():
        return False

    config = load_qwen_config(settings_path)
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

    # Remove MLflow environment variables
    if ENVIRONMENT_FIELD in config:
        mlflow_vars = [
            MLFLOW_TRACING_ENABLED,
            "MLFLOW_TRACKING_URI",
            "MLFLOW_EXPERIMENT_ID",
            "MLFLOW_EXPERIMENT_NAME",
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

    if config:
        save_qwen_config(settings_path, config)
    else:
        settings_path.unlink()

    return hooks_removed or env_removed


def _process_stop_hook(session_id: str | None, transcript_path: str | None) -> dict[str, Any]:
    get_logger().log(
        QWEN_TRACING_LEVEL, "Stop hook: session=%s, transcript=%s", session_id, transcript_path
    )

    trace = process_transcript(transcript_path, session_id)

    if trace is not None:
        return get_hook_response()
    return get_hook_response(
        error="Failed to process transcript, please check .qwen/mlflow/qwen_tracing.log"
    )


def stop_hook_handler() -> None:
    """CLI hook handler for conversation end - processes transcript and creates trace."""
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
