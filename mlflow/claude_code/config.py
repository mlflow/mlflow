"""Configuration management for Claude Code integration with MLflow."""

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
ENVIRONMENT_FIELD = "environment"

# MLflow environment variable constants
MLFLOW_HOOK_IDENTIFIER = "mlflow.claude_code.hooks"
MLFLOW_TRACING_ENABLED = "MLFLOW_CLAUDE_TRACING_ENABLED"


@dataclass
class TracingStatus:
    """Dataclass for tracing status information."""

    enabled: bool
    tracking_uri: str | None = None
    experiment_id: str | None = None
    experiment_name: str | None = None
    reason: str | None = None


def load_claude_config(settings_path: Path) -> dict[str, Any]:
    """Load existing Claude configuration from settings file.

    Args:
        settings_path: Path to Claude settings.json file

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


def save_claude_config(settings_path: Path, config: dict[str, Any]) -> None:
    """Save Claude configuration to settings file.

    Args:
        settings_path: Path to Claude settings.json file
        config: Configuration dictionary to save
    """
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def get_tracing_status(settings_path: Path) -> TracingStatus:
    """Get current tracing status from Claude settings.

    Checks both settings.json and settings.local.json, with local taking
    precedence for env vars (matching Claude Code's own precedence).

    Args:
        settings_path: Path to Claude settings file (e.g., .claude/settings.json)

    Returns:
        TracingStatus with tracing status information
    """
    local_path = settings_path.parent / "settings.local.json"
    config = load_claude_config(settings_path)
    local_config = load_claude_config(local_path)

    if not config and not local_config:
        return TracingStatus(enabled=False, reason="No configuration found")

    env_vars = config.get(ENVIRONMENT_FIELD, {})
    local_env_vars = local_config.get(ENVIRONMENT_FIELD, {})
    merged_env = {**env_vars, **local_env_vars}

    enabled = merged_env.get(MLFLOW_TRACING_ENABLED) == "true"

    return TracingStatus(
        enabled=enabled,
        tracking_uri=merged_env.get(MLFLOW_TRACKING_URI.name),
        experiment_id=merged_env.get(MLFLOW_EXPERIMENT_ID.name),
        experiment_name=merged_env.get(MLFLOW_EXPERIMENT_NAME.name),
    )


def get_env_var(var_name: str, default: str = "") -> str:
    """Get environment variable from OS or Claude settings as fallback.

    Checks in order (first match wins):
    1. OS environment variables (set by Claude Code from settings at startup)
    2. .claude/settings.local.json env block (user-local, not committed)
    3. .claude/settings.json env block (project-level, may be committed)

    Args:
        var_name: Environment variable name
        default: Default value if not found anywhere

    Returns:
        Environment variable value
    """
    # First check OS environment
    value = os.getenv(var_name)
    if value is not None:
        return value

    # Check settings.local.json (user-local overrides, not committed to git)
    for settings_file in ("settings.local.json", "settings.json"):
        try:
            settings_path = Path(f".claude/{settings_file}")
            if settings_path.exists():
                config = load_claude_config(settings_path)
                env_vars = config.get(ENVIRONMENT_FIELD, {})
                value = env_vars.get(var_name)
                if value is not None:
                    return value
        except Exception:
            pass

    return default


def setup_environment_config(
    settings_path: Path,
    tracking_uri: str | None = None,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
) -> None:
    """Set up MLflow environment variables in Claude settings.

    Args:
        settings_path: Path to Claude settings file
        tracking_uri: MLflow tracking URI, defaults to local file storage
        experiment_id: MLflow experiment ID (takes precedence over name)
        experiment_name: MLflow experiment name
    """
    config = load_claude_config(settings_path)

    if ENVIRONMENT_FIELD not in config:
        config[ENVIRONMENT_FIELD] = {}

    # Always enable tracing
    config[ENVIRONMENT_FIELD][MLFLOW_TRACING_ENABLED] = "true"

    # Set tracking URI
    if tracking_uri:
        config[ENVIRONMENT_FIELD][MLFLOW_TRACKING_URI.name] = tracking_uri

    # Set experiment configuration (ID takes precedence over name)
    if experiment_id:
        config[ENVIRONMENT_FIELD][MLFLOW_EXPERIMENT_ID.name] = experiment_id
        config[ENVIRONMENT_FIELD].pop(MLFLOW_EXPERIMENT_NAME.name, None)
    elif experiment_name:
        config[ENVIRONMENT_FIELD][MLFLOW_EXPERIMENT_NAME.name] = experiment_name
        config[ENVIRONMENT_FIELD].pop(MLFLOW_EXPERIMENT_ID.name, None)

    save_claude_config(settings_path, config)
