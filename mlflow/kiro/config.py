"""Configuration management for Kiro CLI integration with MLflow."""

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

# ---------------------------------------------------------------------------
# Kiro hook file structure
# ---------------------------------------------------------------------------
# Kiro stores hooks as individual JSON files under .kiro/hooks/<name>.json
# Format:
# {
#   "version": "1.0",
#   "hooks": [
#     {
#       "name": "...",
#       "description": "...",
#       "event": "AgentStop",
#       "actions": [
#         {
#           "type": "command",
#           "command": "mlflow autolog kiro stop-hook"
#         }
#       ]
#     }
#   ]
# }
KIRO_HOOKS_DIR = ".kiro/hooks"
MLFLOW_HOOK_FILE = "mlflow_autolog.json"
MLFLOW_HOOK_NAME = "MLflow Autolog"
KIRO_HOOK_EVENT_AGENT_STOP = "AgentStop"

# Identifier used to recognise an existing MLflow hook command so we can
# update it in-place rather than adding a duplicate.
MLFLOW_HOOK_IDENTIFIER = "mlflow autolog kiro"

# MLflow env-var stored in the Kiro env config file
KIRO_ENV_FILE = ".kiro/mlflow_env.json"
MLFLOW_TRACING_ENABLED = "MLFLOW_KIRO_TRACING_ENABLED"


# ---------------------------------------------------------------------------
# Status dataclass
# ---------------------------------------------------------------------------


@dataclass
class TracingStatus:
    """Dataclass for tracing status information."""

    enabled: bool
    tracking_uri: str | None = None
    experiment_id: str | None = None
    experiment_name: str | None = None
    reason: str | None = None


# ---------------------------------------------------------------------------
# Env config helpers
# ---------------------------------------------------------------------------


def load_kiro_env(env_path: Path) -> dict[str, Any]:
    """Load MLflow environment config stored alongside the Kiro hook file."""
    if env_path.exists():
        try:
            with open(env_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_kiro_env(env_path: Path, env_vars: dict[str, Any]) -> None:
    """Persist MLflow environment config to disk."""
    env_path.parent.mkdir(parents=True, exist_ok=True)
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env_vars, f, indent=2)


def get_env_var(var_name: str, default: str = "") -> str:
    """Get an MLflow environment variable from the Kiro env config or OS env.

    Project-specific configuration in ``.kiro/mlflow_env.json`` takes
    precedence over global OS environment variables.
    """
    try:
        env_path = Path(KIRO_ENV_FILE)
        if env_path.exists():
            env_vars = load_kiro_env(env_path)
            value = env_vars.get(var_name)
            if value is not None:
                return value
    except Exception:
        pass

    value = os.environ.get(var_name)
    if value is not None:
        return value

    return default


# ---------------------------------------------------------------------------
# Hook file helpers
# ---------------------------------------------------------------------------


def _build_hook_payload(command: str) -> dict[str, Any]:
    """Return a fully-formed Kiro hook JSON payload."""
    return {
        "version": "1.0",
        "hooks": [
            {
                "name": MLFLOW_HOOK_NAME,
                "description": (
                    "Automatically logs Kiro agent sessions to MLflow as traces."
                ),
                "event": KIRO_HOOK_EVENT_AGENT_STOP,
                "enabled": True,
                "actions": [
                    {
                        "type": "command",
                        "command": command,
                    }
                ],
            }
        ],
    }


def _mlflow_command() -> str:
    """Return the base mlflow binary, preferring ``uv run mlflow`` when UV is set."""
    if "UV" in os.environ:
        return "uv run mlflow"
    return "mlflow"


def setup_hooks_config(hooks_dir: Path) -> None:
    """Write (or update) the MLflow hook file inside ``.kiro/hooks/``.

    Args:
        hooks_dir: Path to the ``.kiro/hooks`` directory.
    """
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_file = hooks_dir / MLFLOW_HOOK_FILE
    command = f"{_mlflow_command()} autolog kiro stop-hook"
    payload = _build_hook_payload(command)
    with open(hook_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def disable_tracing_hooks(hooks_dir: Path, env_path: Path) -> bool:
    """Remove the MLflow hook file and env config.

    Args:
        hooks_dir: Path to the ``.kiro/hooks`` directory.
        env_path: Path to the ``.kiro/mlflow_env.json`` file.

    Returns:
        True if anything was removed, False if nothing was configured.
    """
    removed = False
    hook_file = hooks_dir / MLFLOW_HOOK_FILE
    if hook_file.exists():
        hook_file.unlink()
        removed = True
    if env_path.exists():
        env_path.unlink()
        removed = True
    return removed


# ---------------------------------------------------------------------------
# Status helper
# ---------------------------------------------------------------------------


def get_tracing_status(hooks_dir: Path, env_path: Path) -> TracingStatus:
    """Return the current Kiro tracing status.

    Args:
        hooks_dir: Path to the ``.kiro/hooks`` directory.
        env_path: Path to the ``.kiro/mlflow_env.json`` file.
    """
    hook_file = hooks_dir / MLFLOW_HOOK_FILE
    if not hook_file.exists():
        return TracingStatus(enabled=False, reason="No MLflow hook file found")

    env_vars = load_kiro_env(env_path)
    enabled = env_vars.get(MLFLOW_TRACING_ENABLED) == "true"

    return TracingStatus(
        enabled=enabled,
        tracking_uri=env_vars.get(MLFLOW_TRACKING_URI.name),
        experiment_id=env_vars.get(MLFLOW_EXPERIMENT_ID.name),
        experiment_name=env_vars.get(MLFLOW_EXPERIMENT_NAME.name),
    )


# ---------------------------------------------------------------------------
# Environment config setup
# ---------------------------------------------------------------------------


def setup_environment_config(
    env_path: Path,
    tracking_uri: str | None = None,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
) -> None:
    """Write MLflow environment variables to ``.kiro/mlflow_env.json``.

    Args:
        env_path: Destination path for the env config.
        tracking_uri: MLflow tracking URI.
        experiment_id: MLflow experiment ID (takes precedence over name).
        experiment_name: MLflow experiment name.
    """
    env_vars = load_kiro_env(env_path)

    # Always enable tracing
    env_vars[MLFLOW_TRACING_ENABLED] = "true"

    if tracking_uri:
        env_vars[MLFLOW_TRACKING_URI.name] = tracking_uri

    if experiment_id:
        env_vars[MLFLOW_EXPERIMENT_ID.name] = experiment_id
        env_vars.pop(MLFLOW_EXPERIMENT_NAME.name, None)
    elif experiment_name:
        env_vars[MLFLOW_EXPERIMENT_NAME.name] = experiment_name
        env_vars.pop(MLFLOW_EXPERIMENT_ID.name, None)

    save_kiro_env(env_path, env_vars)
