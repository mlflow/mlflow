"""Configuration management for Opencode integration with MLflow."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# MLflow environment variable constants
MLFLOW_TRACING_ENABLED = "MLFLOW_OPENCODE_TRACING_ENABLED"

# npm package name for the MLflow Opencode plugin
MLFLOW_PLUGIN_NPM_PACKAGE = "mlflow-opencode"

# Opencode config file name
OPENCODE_CONFIG_FILE = "opencode.json"


@dataclass
class TracingStatus:
    enabled: bool
    tracking_uri: str | None = None
    experiment_id: str | None = None
    experiment_name: str | None = None
    reason: str | None = None


def get_opencode_config_path(directory: Path) -> Path:
    return directory / OPENCODE_CONFIG_FILE


def load_json_config(config_path: Path) -> dict[str, Any]:
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_opencode_config(config_path: Path, config: dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _is_mlflow_plugin(plugin_name: str) -> bool:
    return MLFLOW_PLUGIN_NPM_PACKAGE in plugin_name or (
        plugin_name.startswith("file://") and "mlflow" in plugin_name.lower()
    )


def _get_mlflow_config_path(directory: Path) -> Path:
    return directory / ".opencode" / "mlflow.json"


def _load_mlflow_config(directory: Path) -> dict[str, Any]:
    return load_json_config(_get_mlflow_config_path(directory))


def get_env_var(var_name: str, default: str = "") -> str:
    return os.getenv(var_name, default)


def get_tracing_status(config_path: Path) -> TracingStatus:
    if not config_path.exists():
        return TracingStatus(enabled=False, reason="No configuration found")

    config = load_json_config(config_path)

    # Check if MLflow plugin is configured
    plugins = config.get("plugin", [])
    enabled = any(_is_mlflow_plugin(p) for p in plugins)

    # Read MLflow settings from .opencode/mlflow.json (not environment variables)
    directory = config_path.parent
    mlflow_config = _load_mlflow_config(directory)

    tracking_uri = mlflow_config.get("trackingUri")
    experiment_id = mlflow_config.get("experimentId")
    experiment_name = mlflow_config.get("experimentName")

    return TracingStatus(
        enabled=enabled,
        tracking_uri=tracking_uri,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
    )


def setup_hook_config(
    config_path: Path,
    tracking_uri: str | None = None,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
) -> str:
    config = load_json_config(config_path)

    # Ensure plugin array exists
    if "plugin" not in config:
        config["plugin"] = []

    # Remove any existing MLflow plugin entries
    config["plugin"] = [p for p in config["plugin"] if not _is_mlflow_plugin(p)]

    # Add the MLflow plugin npm package
    config["plugin"].append(MLFLOW_PLUGIN_NPM_PACKAGE)

    save_opencode_config(config_path, config)

    # Create MLflow config file that the TypeScript plugin will read
    _create_mlflow_config_file(config_path.parent, tracking_uri, experiment_id, experiment_name)

    return MLFLOW_PLUGIN_NPM_PACKAGE


def _create_mlflow_config_file(
    directory: Path,
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
) -> None:
    opencode_dir = directory / ".opencode"
    opencode_dir.mkdir(parents=True, exist_ok=True)

    config_file = opencode_dir / "mlflow.json"
    config_data: dict[str, Any] = {"enabled": True}

    if tracking_uri:
        config_data["trackingUri"] = tracking_uri
    if experiment_id:
        config_data["experimentId"] = experiment_id
    elif experiment_name:
        config_data["experimentName"] = experiment_name

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)


def disable_tracing(config_path: Path) -> bool:
    if not config_path.exists():
        return False

    config = load_json_config(config_path)
    modified = False

    # Remove MLflow plugin from plugin list
    if "plugin" in config:
        original_len = len(config["plugin"])
        config["plugin"] = [p for p in config["plugin"] if not _is_mlflow_plugin(p)]
        if len(config["plugin"]) != original_len:
            modified = True
        if not config["plugin"]:
            del config["plugin"]

    # Remove MLflow config file if it exists
    mlflow_config = _get_mlflow_config_path(config_path.parent)
    if mlflow_config.exists():
        mlflow_config.unlink()
        modified = True

    if modified:
        if config:
            save_opencode_config(config_path, config)
        else:
            config_path.unlink()

    return modified
