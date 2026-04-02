"""Hook management for Codex CLI integration with MLflow."""

import json
import os
import sys
from pathlib import Path
from typing import Any

from mlflow.codex.config import (
    CODEX_HOOKS_FILE,
    HOOK_FIELD_COMMAND,
    HOOK_FIELD_HOOKS,
    MLFLOW_HOOK_IDENTIFIER,
    load_codex_hooks,
    save_codex_hooks,
)
from mlflow.codex.tracing import (
    CODEX_TRACING_LEVEL,
    get_hook_response,
    get_logger,
    process_transcript,
    read_hook_input,
    setup_mlflow,
)


def upsert_hook(config: dict[str, Any], hook_type: str, subcommand: str) -> None:
    """Insert or update a single MLflow hook in the Codex hooks configuration."""
    if hook_type not in config:
        config[hook_type] = []

    mlflow_cmd = "uv run mlflow" if "UV" in os.environ else "mlflow"
    hook_command = f"{mlflow_cmd} autolog codex {subcommand}"
    mlflow_hook = {"type": "command", HOOK_FIELD_COMMAND: hook_command}

    # Check if MLflow hook already exists and update it
    hook_exists = False
    for hook_group in config[hook_type]:
        if HOOK_FIELD_HOOKS in hook_group:
            for hook in hook_group[HOOK_FIELD_HOOKS]:
                if MLFLOW_HOOK_IDENTIFIER in hook.get(HOOK_FIELD_COMMAND, ""):
                    hook.update(mlflow_hook)
                    hook_exists = True
                    break

    if not hook_exists:
        config[hook_type].append({HOOK_FIELD_HOOKS: [mlflow_hook]})


def setup_hooks_config(codex_dir: Path) -> None:
    """Set up Codex hooks for MLflow tracing.

    Creates or updates Stop hook in .codex/hooks.json.
    """
    hooks_path = codex_dir / CODEX_HOOKS_FILE
    config = load_codex_hooks(hooks_path)
    upsert_hook(config, "Stop", "stop-hook")
    save_codex_hooks(hooks_path, config)


def disable_tracing_hooks(codex_dir: Path) -> bool:
    """Remove MLflow hooks from Codex hooks.json.

    Returns:
        True if hooks were removed, False if none found
    """
    hooks_path = codex_dir / CODEX_HOOKS_FILE
    if not hooks_path.exists():
        return False

    config = load_codex_hooks(hooks_path)
    hooks_removed = False

    if "Stop" in config:
        hook_groups = config["Stop"]
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
            config["Stop"] = filtered_groups
        else:
            del config["Stop"]
            hooks_removed = True

    if config:
        save_codex_hooks(hooks_path, config)
    else:
        hooks_path.unlink()

    return hooks_removed


def _process_stop_hook(session_id: str | None, transcript_path: str | None) -> dict[str, Any]:
    get_logger().log(
        CODEX_TRACING_LEVEL, "Stop hook: session=%s, transcript=%s", session_id, transcript_path
    )

    trace = process_transcript(transcript_path, session_id)

    if trace is not None:
        return get_hook_response()
    return get_hook_response(
        error="Failed to process transcript, please check .codex/mlflow/codex_tracing.log"
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
