"""Configuration management for Qwen Code integration with MLflow."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Qwen Code uses same settings.json structure as Claude Code
HOOK_FIELD_HOOKS = "hooks"
HOOK_FIELD_COMMAND = "command"
ENVIRONMENT_FIELD = "env"

# MLflow hook identifiers
MLFLOW_HOOK_IDENTIFIER = "mlflow autolog qwen-code"
MLFLOW_TRACING_ENABLED = "MLFLOW_QWEN_TRACING_ENABLED"

# Qwen config paths
QWEN_DIR_NAME = ".qwen"
QWEN_SETTINGS_FILE = "settings.json"


@dataclass
class TracingStatus:
    enabled: bool
    tracking_uri: str | None = None
    experiment_id: str | None = None
    experiment_name: str | None = None
    reason: str | None = None


def load_qwen_config(settings_path: Path) -> dict[str, Any]:
    if settings_path.exists():
        try:
            with open(settings_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_qwen_config(settings_path: Path, config: dict[str, Any]) -> None:
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def get_tracing_status(settings_path: Path) -> TracingStatus:
    if not settings_path.exists():
        return TracingStatus(enabled=False, reason="No configuration found")

    config = load_qwen_config(settings_path)
    env_vars = config.get(ENVIRONMENT_FIELD, {})
    enabled = env_vars.get(MLFLOW_TRACING_ENABLED) == "true"

    return TracingStatus(
        enabled=enabled,
        tracking_uri=env_vars.get("MLFLOW_TRACKING_URI"),
        experiment_id=env_vars.get("MLFLOW_EXPERIMENT_ID"),
        experiment_name=env_vars.get("MLFLOW_EXPERIMENT_NAME"),
    )


def setup_environment_config(
    settings_path: Path,
    tracking_uri: str | None = None,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
) -> None:
    config = load_qwen_config(settings_path)

    if ENVIRONMENT_FIELD not in config:
        config[ENVIRONMENT_FIELD] = {}

    config[ENVIRONMENT_FIELD][MLFLOW_TRACING_ENABLED] = "true"

    if tracking_uri:
        config[ENVIRONMENT_FIELD]["MLFLOW_TRACKING_URI"] = tracking_uri

    if experiment_id:
        config[ENVIRONMENT_FIELD]["MLFLOW_EXPERIMENT_ID"] = experiment_id
        config[ENVIRONMENT_FIELD].pop("MLFLOW_EXPERIMENT_NAME", None)
    elif experiment_name:
        config[ENVIRONMENT_FIELD]["MLFLOW_EXPERIMENT_NAME"] = experiment_name
        config[ENVIRONMENT_FIELD].pop("MLFLOW_EXPERIMENT_ID", None)

    save_qwen_config(settings_path, config)
