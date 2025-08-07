#!/usr/bin/env python3
"""claude-mlflow: A wrapper for Claude Code that automatically traces interactions with MLflow."""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

HOOK_FIELD_HOOKS = "hooks"
HOOK_FIELD_COMMAND = "command"
MLFLOW_HOOK_IDENTIFIER = "mlflow.claudecode.claude_code_tracing"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def upsert_hook(config: dict[str, Any], hook_type: str, handler_name: str) -> None:
    """Upsert a single MLflow hook into the configuration.

    Args:
        config: The hooks configuration dictionary to modify
        hook_type: The hook type (e.g., 'PostToolUse', 'Stop')
        handler_name: The handler function name (e.g., 'post_tool_use_handler')
    """
    if hook_type not in config[HOOK_FIELD_HOOKS]:
        config[HOOK_FIELD_HOOKS][hook_type] = []

    hook_command = (
        f'python -c "from mlflow.claudecode.claude_code_tracing import {handler_name}; '
        f'{handler_name}()"'
    )

    mlflow_hook = {"type": "command", HOOK_FIELD_COMMAND: hook_command}

    # Check if MLflow hook already exists and update it
    hook_exists = False
    for hook_group in config[HOOK_FIELD_HOOKS][hook_type]:
        if HOOK_FIELD_HOOKS in hook_group:
            for hook in hook_group[HOOK_FIELD_HOOKS]:
                if MLFLOW_HOOK_IDENTIFIER in hook.get(HOOK_FIELD_COMMAND, ""):
                    hook.update(mlflow_hook)
                    hook_exists = True
                    break

    # Add new hook if it doesn't exist
    if not hook_exists:
        config[HOOK_FIELD_HOOKS][hook_type].append({HOOK_FIELD_HOOKS: [mlflow_hook]})


def setup_hooks_config() -> None:
    """Create or update .claude/settings.json with MLflow tracing hooks configuration.

    Creates hooks for PostToolUse and Stop events that call MLflow tracing handlers.
    Updates existing MLflow hooks if found, otherwise adds new ones.
    """
    claude_dir = Path(".claude")
    claude_dir.mkdir(exist_ok=True)
    settings_path = claude_dir / "settings.json"

    existing_config = {}
    if settings_path.exists():
        try:
            with open(settings_path, encoding="utf-8") as f:
                existing_config = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing_config = {}

    if HOOK_FIELD_HOOKS not in existing_config:
        existing_config[HOOK_FIELD_HOOKS] = {}

    upsert_hook(existing_config, "PostToolUse", "post_tool_use_handler")
    upsert_hook(existing_config, "Stop", "stop_hook_handler")

    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(existing_config, f, indent=2)


def setup_claude_hooks() -> None:
    """Configure Claude Code hooks for MLflow tracing."""
    try:
        setup_hooks_config()

        from mlflow.claudecode.claude_code_tracing import setup_mlflow

        setup_mlflow()
    except ImportError as e:
        logger.warning("Could not import MLflow tracing: %s", e)
    except Exception as e:
        logger.warning("Failed to setup tracing hooks: %s", e)


def find_claude_executable() -> str:
    """Find the claude executable in the system PATH."""
    claude_path = subprocess.run(
        ["which", "claude"], capture_output=True, text=True, check=False
    ).stdout.strip()

    if not claude_path:
        logger.error("claude executable not found in PATH")
        logger.error("Please install Claude Code first: https://claude.ai/code")
        sys.exit(1)

    return claude_path


def main(args: Optional[list[str]] = None) -> None:
    """Main entry point for claude-mlflow.

    Args:
        args: Command line arguments to forward to claude, defaults to sys.argv[1:]
    """
    if args is None:
        args = sys.argv[1:]

    os.environ["MLFLOW_CLAUDE_TRACING_ENABLED"] = "true"

    setup_claude_hooks()
    claude_path = find_claude_executable()

    try:
        result = subprocess.run([claude_path] + args, check=False)
        os.environ["MLFLOW_CLAUDE_TRACING_ENABLED"] = "false"
        sys.exit(result.returncode)
    except Exception as e:
        logger.error("Error running claude: %s", e)
        os.environ["MLFLOW_CLAUDE_TRACING_ENABLED"] = "false"
        sys.exit(130 if isinstance(e, KeyboardInterrupt) else 1)


if __name__ == "__main__":
    main()
