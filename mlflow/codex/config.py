"""Configuration management for Codex CLI integration with MLflow."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Configuration field constants (Codex uses same hook schema as Claude Code)
HOOK_FIELD_HOOKS = "hooks"
HOOK_FIELD_COMMAND = "command"

# MLflow hook identifiers
MLFLOW_HOOK_IDENTIFIER = "mlflow autolog codex"

# Codex config paths
CODEX_DIR_NAME = ".codex"
CODEX_HOOKS_FILE = "hooks.json"


@dataclass
class TracingStatus:
    enabled: bool
    tracking_uri: str | None = None
    experiment_id: str | None = None
    experiment_name: str | None = None
    reason: str | None = None


def load_codex_hooks(hooks_path: Path) -> dict[str, Any]:
    if hooks_path.exists():
        try:
            with open(hooks_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_codex_hooks(hooks_path: Path, config: dict[str, Any]) -> None:
    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hooks_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def get_tracing_status(codex_dir: Path) -> TracingStatus:
    hooks_path = codex_dir / CODEX_HOOKS_FILE
    if not hooks_path.exists():
        return TracingStatus(enabled=False, reason="No configuration found")

    config = load_codex_hooks(hooks_path)
    # Check if MLflow Stop hook is registered
    has_hook = False
    for hook_group in config.get("Stop", []):
        for hook in hook_group.get(HOOK_FIELD_HOOKS, []):
            if MLFLOW_HOOK_IDENTIFIER in hook.get(HOOK_FIELD_COMMAND, ""):
                has_hook = True
                break

    if not has_hook:
        return TracingStatus(enabled=False, reason="MLflow hooks not configured")

    return TracingStatus(
        enabled=True,
        tracking_uri=None,
        experiment_id=None,
        experiment_name=None,
    )
