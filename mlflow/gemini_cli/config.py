"""Configuration management for Gemini CLI integration with MLflow."""

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

# Configuration field constants
HOOK_FIELD_HOOKS = "hooks"
HOOK_FIELD_COMMAND = "command"

# MLflow environment variable constants
MLFLOW_HOOK_IDENTIFIER = "mlflow.gemini_cli.hooks"
MLFLOW_TRACING_ENABLED = "MLFLOW_GEMINI_CLI_TRACING_ENABLED"


@dataclass
class TracingStatus:
    """Dataclass for tracing status information."""

    enabled: bool
    tracking_uri: str | None = None
    experiment_id: str | None = None
    experiment_name: str | None = None
    reason: str | None = None


def load_gemini_config(settings_path: Path) -> dict[str, Any]:
    """Load existing Gemini CLI configuration from settings file.

    Args:
        settings_path: Path to Gemini settings.json file

    Returns:
        Configuration dictionary, empty dict if file doesn't exist or is invalid
    """
    if settings_path.exists():
        try:
            with open(settings_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_gemini_config(settings_path: Path, config: dict[str, Any]) -> None:
    """Save Gemini CLI configuration to settings file.

    Args:
        settings_path: Path to Gemini settings.json file
        config: Configuration dictionary to save
    """
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def get_tracing_status(settings_path: Path) -> TracingStatus:
    """Get current tracing status from Gemini CLI settings.

    Args:
        settings_path: Path to Gemini settings file

    Returns:
        TracingStatus with tracing status information
    """
    if not settings_path.exists():
        return TracingStatus(enabled=False, reason="No configuration found")

    config = load_gemini_config(settings_path)
    hooks = config.get(HOOK_FIELD_HOOKS, {})
    session_end_hooks = hooks.get("SessionEnd", [])

    # Check if any MLflow hook is configured
    enabled = False
    for hook_group in session_end_hooks:
        if HOOK_FIELD_HOOKS in hook_group:
            for hook in hook_group[HOOK_FIELD_HOOKS]:
                if MLFLOW_HOOK_IDENTIFIER in hook.get(HOOK_FIELD_COMMAND, ""):
                    enabled = True
                    break

    # Get environment variables from OS environment (Gemini CLI uses env vars directly)
    tracking_uri = os.environ.get(MLFLOW_TRACKING_URI.name)
    experiment_id = os.environ.get(MLFLOW_EXPERIMENT_ID.name)
    experiment_name = os.environ.get(MLFLOW_EXPERIMENT_NAME.name)

    return TracingStatus(
        enabled=enabled,
        tracking_uri=tracking_uri,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
    )


def get_env_var(var_name: str, default: str = "") -> str:
    """Get environment variable value.

    Args:
        var_name: Environment variable name
        default: Default value if not found

    Returns:
        Environment variable value
    """
    value = os.environ.get(var_name)
    if value is not None:
        return value
    return default


def setup_environment_config(
    tracking_uri: str | None = None,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
) -> dict[str, str]:
    """Build a dictionary of MLflow environment variables for user reference.

    Gemini CLI hooks receive environment variables from the shell, so we
    return the variables the user should set, rather than writing them
    into the settings file.

    Args:
        tracking_uri: MLflow tracking URI
        experiment_id: MLflow experiment ID (takes precedence over name)
        experiment_name: MLflow experiment name

    Returns:
        Dictionary of environment variable name to value
    """
    env_vars: dict[str, str] = {}
    env_vars[MLFLOW_TRACING_ENABLED] = "true"

    if tracking_uri:
        env_vars[MLFLOW_TRACKING_URI.name] = tracking_uri
    if experiment_id:
        env_vars[MLFLOW_EXPERIMENT_ID.name] = experiment_id
    elif experiment_name:
        env_vars[MLFLOW_EXPERIMENT_NAME.name] = experiment_name

    return env_vars
