"""Configuration management for Claude Code integration with MLflow."""

import json
import os
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


def get_tracing_status(settings_path: Path) -> dict[str, Any]:
    """Get current tracing status from Claude settings.

    Args:
        settings_path: Path to Claude settings file

    Returns:
        Dictionary with tracing status information:
        - enabled: bool indicating if tracing is active
        - tracking_uri: MLflow tracking URI or "Not set"
        - experiment_id: Experiment ID if set
        - experiment_name: Experiment name if set
        - reason: Reason if not enabled
    """
    if not settings_path.exists():
        return {"enabled": False, "reason": "No configuration found"}

    config = load_claude_config(settings_path)
    env_vars = config.get(ENVIRONMENT_FIELD, {})
    enabled = env_vars.get(MLFLOW_TRACING_ENABLED) == "true"

    return {
        "enabled": enabled,
        "tracking_uri": env_vars.get(MLFLOW_TRACKING_URI.name, "Not set"),
        "experiment_id": env_vars.get(MLFLOW_EXPERIMENT_ID.name),
        "experiment_name": env_vars.get(MLFLOW_EXPERIMENT_NAME.name),
    }


def get_env_var(var_name: str, default: str = "") -> str:
    """Get environment variable from OS or Claude settings as fallback.

    Args:
        var_name: Environment variable name
        default: Default value if not found anywhere

    Returns:
        Environment variable value
    """
    # First check OS environment
    value = os.getenv(var_name)
    if value:
        return value

    # Fallback to Claude settings
    try:
        settings_path = Path(".claude/settings.json")
        if settings_path.exists():
            config = load_claude_config(settings_path)
            env_vars = config.get(ENVIRONMENT_FIELD, {})
            return env_vars.get(var_name, default)
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
