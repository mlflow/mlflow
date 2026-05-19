"""Plugin bootstrap helpers for Claude Code tracing."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import click

from mlflow.claude_code.config import (
    ENVIRONMENT_FIELD,
    MLFLOW_EXPERIMENT_ID,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACING_ENABLED,
    MLFLOW_TRACKING_URI,
    load_claude_config,
    save_claude_config,
)

CLAUDE_BINARY = "claude"
MARKETPLACE_NAME = "mlflow-plugins"
MARKETPLACE_SOURCE = "mlflow/mlflow"
PLUGIN_ID = f"mlflow-tracing@{MARKETPLACE_NAME}"
MARKETPLACE_SPARSE_PATHS = [".claude-plugin", "libs/typescript/integrations/claude-code"]


def ensure_plugin_installed(target_dir: Path) -> None:
    """Install the MLflow Claude plugin into Claude Code for ``target_dir``."""
    if shutil.which(CLAUDE_BINARY) is None:
        raise click.ClickException(
            "Claude Code CLI (`claude`) is not installed or not on PATH. "
            "Install Claude Code first, then rerun `mlflow autolog claude`."
        )

    _run_claude(
        target_dir,
        "plugin",
        "marketplace",
        "add",
        MARKETPLACE_SOURCE,
        "--scope",
        "local",
        "--sparse",
        *MARKETPLACE_SPARSE_PATHS,
    )
    _run_claude(
        target_dir,
        "plugin",
        "install",
        PLUGIN_ID,
        "--scope",
        "local",
    )


def disable_tracing_plugin(settings_path: Path) -> bool:
    """Remove MLflow Claude config from settings."""
    if not settings_path.exists():
        return False

    config = load_claude_config(settings_path)
    env_removed = _remove_mlflow_env(config)

    if config:
        save_claude_config(settings_path, config)
    else:
        settings_path.unlink()

    return env_removed


def _run_claude(target_dir: Path, *args: str) -> subprocess.CompletedProcess:
    command = [CLAUDE_BINARY, *args]
    try:
        return subprocess.run(
            command,
            cwd=target_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise click.ClickException(f"Failed to run `{' '.join(command)}`:\n{detail}") from exc


def _remove_mlflow_env(config: dict[str, Any]) -> bool:
    env_vars = config.get(ENVIRONMENT_FIELD)
    if not env_vars:
        return False

    removed = False
    for var in (
        MLFLOW_TRACING_ENABLED,
        MLFLOW_TRACKING_URI.name,
        MLFLOW_EXPERIMENT_ID.name,
        MLFLOW_EXPERIMENT_NAME.name,
    ):
        if var in env_vars:
            del env_vars[var]
            removed = True

    if not env_vars:
        del config[ENVIRONMENT_FIELD]

    return removed
