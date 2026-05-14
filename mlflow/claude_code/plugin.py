"""Plugin bootstrap and migration helpers for Claude Code tracing."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import click

from mlflow.claude_code.config import (
    ENVIRONMENT_FIELD,
    HOOK_FIELD_COMMAND,
    HOOK_FIELD_HOOKS,
    MLFLOW_EXPERIMENT_ID,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_HOOK_IDENTIFIER,
    MLFLOW_LEGACY_HOOK_IDENTIFIER,
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
_ALREADY_EXISTS_MARKERS = ("already exists", "already added", "has already been added")
_ALREADY_INSTALLED_MARKERS = (
    "already installed",
    "already at the latest version",
    "is already at the latest version",
)


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
        ok_if=_ALREADY_EXISTS_MARKERS,
    )
    _run_claude(
        target_dir,
        "plugin",
        "install",
        PLUGIN_ID,
        "--scope",
        "local",
        ok_if=_ALREADY_INSTALLED_MARKERS,
    )


def disable_tracing_plugin(settings_path: Path) -> bool:
    """Remove MLflow Claude config from settings and clean up legacy hooks."""
    if not settings_path.exists():
        return False

    config = load_claude_config(settings_path)
    hooks_removed = _remove_legacy_mlflow_hooks(config)
    env_removed = _remove_mlflow_env(config)

    if HOOK_FIELD_HOOKS in config and not config[HOOK_FIELD_HOOKS]:
        del config[HOOK_FIELD_HOOKS]

    if config:
        save_claude_config(settings_path, config)
    else:
        settings_path.unlink()

    return hooks_removed or env_removed


def migrate_legacy_hooks(settings_path: Path) -> bool:
    """Remove old Python MLflow hook entries after plugin-based setup."""
    if not settings_path.exists():
        return False

    config = load_claude_config(settings_path)
    hooks_removed = _remove_legacy_mlflow_hooks(config)
    if not hooks_removed:
        return False

    if HOOK_FIELD_HOOKS in config and not config[HOOK_FIELD_HOOKS]:
        del config[HOOK_FIELD_HOOKS]

    if config:
        save_claude_config(settings_path, config)
    else:
        settings_path.unlink()

    return True


def _run_claude(target_dir: Path, *args: str, ok_if: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    command = [CLAUDE_BINARY, *args]
    try:
        return subprocess.run(  # noqa: S603
            command,
            cwd=target_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        output = "\n".join(part for part in (exc.stdout, exc.stderr) if part).lower()
        if any(marker in output for marker in ok_if):
            return exc
        raise click.ClickException(
            f"Failed to run `{' '.join(command)}`:\n{(exc.stderr or exc.stdout or str(exc)).strip()}"
        ) from exc


def _remove_legacy_mlflow_hooks(config: dict) -> bool:
    hooks_removed = False
    stop_groups = config.get(HOOK_FIELD_HOOKS, {}).get("Stop")
    if not stop_groups:
        return False

    filtered_groups = []
    for group in stop_groups:
        hooks = group.get(HOOK_FIELD_HOOKS)
        if not hooks:
            filtered_groups.append(group)
            continue

        filtered_hooks = [
            hook
            for hook in hooks
            if MLFLOW_HOOK_IDENTIFIER not in hook.get(HOOK_FIELD_COMMAND, "")
            and MLFLOW_LEGACY_HOOK_IDENTIFIER not in hook.get(HOOK_FIELD_COMMAND, "")
        ]
        if filtered_hooks:
            filtered_groups.append({HOOK_FIELD_HOOKS: filtered_hooks})
        else:
            hooks_removed = True

    if filtered_groups:
        config[HOOK_FIELD_HOOKS]["Stop"] = filtered_groups
    else:
        del config[HOOK_FIELD_HOOKS]["Stop"]
        hooks_removed = True

    return hooks_removed


def _remove_mlflow_env(config: dict) -> bool:
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
