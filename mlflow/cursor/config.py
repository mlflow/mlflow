"""Configuration management for Cursor integration with MLflow."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlflow.environment_variables import (
    MLFLOW_EXPERIMENT_ID,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)

# Configuration file constants
HOOKS_FILE = "hooks.json"
CURSOR_DIR = ".cursor"
HOOKS_DIR = "hooks"

# Configuration field constants
HOOK_FIELD_VERSION = "version"
HOOK_FIELD_HOOKS = "hooks"
HOOK_FIELD_COMMAND = "command"

# MLflow environment variable constants
MLFLOW_HOOK_IDENTIFIER = "mlflow.cursor.hooks"
MLFLOW_CURSOR_TRACING_ENABLED = "MLFLOW_CURSOR_TRACING_ENABLED"

# Cursor hook types
CURSOR_HOOK_TYPES = [
    "beforeSubmitPrompt",
    "afterAgentResponse",
    "afterAgentThought",
    "beforeShellExecution",
    "afterShellExecution",
    "beforeMCPExecution",
    "afterMCPExecution",
    "beforeReadFile",
    "afterFileEdit",
    "stop",
    "beforeTabFileRead",
    "afterTabFileEdit",
]


@dataclass
class TracingStatus:
    """Dataclass for tracing status information."""

    enabled: bool
    tracking_uri: str | None = None
    experiment_id: str | None = None
    experiment_name: str | None = None
    reason: str | None = None


def get_cursor_hooks_path(directory: Path) -> Path:
    """Get the path to the Cursor hooks.json file.

    Args:
        directory: Base directory for the Cursor configuration

    Returns:
        Path to the hooks.json file
    """
    return directory / CURSOR_DIR / HOOKS_FILE


def get_cursor_hooks_dir(directory: Path) -> Path:
    """Get the path to the Cursor hooks directory.

    Args:
        directory: Base directory for the Cursor configuration

    Returns:
        Path to the hooks directory
    """
    return directory / CURSOR_DIR / HOOKS_DIR


def load_cursor_config(hooks_path: Path) -> dict[str, Any]:
    """Load existing Cursor hooks configuration from file.

    Args:
        hooks_path: Path to Cursor hooks.json file

    Returns:
        Configuration dictionary, empty dict with version if file doesn't exist
    """
    if hooks_path.exists():
        try:
            with open(hooks_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {HOOK_FIELD_VERSION: 1, HOOK_FIELD_HOOKS: {}}
    return {HOOK_FIELD_VERSION: 1, HOOK_FIELD_HOOKS: {}}


def save_cursor_config(hooks_path: Path, config: dict[str, Any]) -> None:
    """Save Cursor hooks configuration to file.

    Args:
        hooks_path: Path to Cursor hooks.json file
        config: Configuration dictionary to save
    """
    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hooks_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def get_tracing_status(directory: Path) -> TracingStatus:
    """Get current tracing status from Cursor configuration.

    Args:
        directory: Base directory to check for Cursor configuration

    Returns:
        TracingStatus with tracing status information
    """
    hooks_path = get_cursor_hooks_path(directory)
    env_file = directory / CURSOR_DIR / ".env"

    if not hooks_path.exists():
        return TracingStatus(enabled=False, reason="No configuration found")

    config = load_cursor_config(hooks_path)

    # Check if MLflow hooks are configured
    hooks = config.get(HOOK_FIELD_HOOKS, {})
    mlflow_hooks_found = False

    for hook_type in CURSOR_HOOK_TYPES:
        if hook_type in hooks:
            for hook_entry in hooks[hook_type]:
                if isinstance(hook_entry, dict):
                    command = hook_entry.get(HOOK_FIELD_COMMAND, "")
                    if MLFLOW_HOOK_IDENTIFIER in command:
                        mlflow_hooks_found = True
                        break

    if not mlflow_hooks_found:
        return TracingStatus(enabled=False, reason="MLflow hooks not configured")

    # Read environment variables from .env file
    tracking_uri = None
    experiment_id = None
    experiment_name = None

    if env_file.exists():
        try:
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and "=" in line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        if key == MLFLOW_TRACKING_URI.name:
                            tracking_uri = value
                        elif key == MLFLOW_EXPERIMENT_ID.name:
                            experiment_id = value
                        elif key == MLFLOW_EXPERIMENT_NAME.name:
                            experiment_name = value
                        elif key == MLFLOW_CURSOR_TRACING_ENABLED:
                            if value.lower() not in ("true", "1", "yes"):
                                return TracingStatus(
                                    enabled=False, reason="Tracing disabled in .env"
                                )
        except IOError:
            pass

    return TracingStatus(
        enabled=True,
        tracking_uri=tracking_uri,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
    )


def get_env_var(var_name: str, default: str = "") -> str:
    """Get environment variable from Cursor .env file or OS environment as fallback.

    Project-specific configuration in .env takes precedence over
    global OS environment variables.

    Args:
        var_name: Environment variable name
        default: Default value if not found anywhere

    Returns:
        Environment variable value
    """
    # First check Cursor .env file (project-specific configuration takes priority)
    try:
        env_file = Path(os.getcwd()) / CURSOR_DIR / ".env"
        if env_file.exists():
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and "=" in line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        if key.strip() == var_name:
                            return value.strip().strip("'\"")
    except Exception:
        pass

    # Fallback to OS environment
    value = os.getenv(var_name)
    if value is not None:
        return value

    return default


def is_tracing_enabled() -> bool:
    """Check if MLflow Cursor tracing is enabled via environment variable."""
    return get_env_var(MLFLOW_CURSOR_TRACING_ENABLED).lower() in ("true", "1", "yes")


def setup_environment_config(
    directory: Path,
    tracking_uri: str | None = None,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
) -> None:
    """Set up MLflow environment variables in Cursor .env file.

    Args:
        directory: Base directory for the Cursor configuration
        tracking_uri: MLflow tracking URI, defaults to local file storage
        experiment_id: MLflow experiment ID (takes precedence over name)
        experiment_name: MLflow experiment name
    """
    env_file = directory / CURSOR_DIR / ".env"
    env_file.parent.mkdir(parents=True, exist_ok=True)

    # Read existing .env content if it exists
    existing_vars: dict[str, str] = {}
    if env_file.exists():
        try:
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and "=" in line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        existing_vars[key.strip()] = value.strip()
        except IOError:
            pass

    # Update with new values
    existing_vars[MLFLOW_CURSOR_TRACING_ENABLED] = "true"

    if tracking_uri:
        existing_vars[MLFLOW_TRACKING_URI.name] = tracking_uri

    if experiment_id:
        existing_vars[MLFLOW_EXPERIMENT_ID.name] = experiment_id
        existing_vars.pop(MLFLOW_EXPERIMENT_NAME.name, None)
    elif experiment_name:
        existing_vars[MLFLOW_EXPERIMENT_NAME.name] = experiment_name
        existing_vars.pop(MLFLOW_EXPERIMENT_ID.name, None)

    # Write updated .env file
    with open(env_file, "w", encoding="utf-8") as f:
        f.write("# MLflow Cursor Tracing Configuration\n")
        f.write("# This file is auto-generated by `mlflow autolog cursor`\n\n")
        for key, value in sorted(existing_vars.items()):
            f.write(f"{key}={value}\n")
