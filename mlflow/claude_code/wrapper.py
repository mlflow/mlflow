"""Claude CLI wrapper for automatic MLflow tracing."""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from mlflow.claude_code.config import MLFLOW_TRACING_ENABLED
from mlflow.claude_code.hooks import setup_hooks_config

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_claude_executable() -> str:
    """Find the claude executable in the system PATH.

    Returns:
        Path to claude executable

    Raises:
        SystemExit: If claude executable is not found
    """
    claude_path = subprocess.run(
        ["which", "claude"], capture_output=True, text=True, check=False
    ).stdout.strip()

    if not claude_path:
        logger.error("claude executable not found in PATH")
        logger.error("Please install Claude Code first: https://claude.ai/code")
        sys.exit(1)

    return claude_path


def setup_claude_hooks() -> None:
    """Configure Claude Code hooks for MLflow tracing.

    Sets up hooks in the current directory's .claude/settings.json file.
    Also initializes MLflow tracing if available.
    """
    try:
        settings_path = Path(".claude") / "settings.json"
        setup_hooks_config(settings_path)

        from mlflow.claude_code.tracing import setup_mlflow

        setup_mlflow()
    except ImportError as e:
        logger.warning("Could not import MLflow tracing: %s", e)
    except Exception as e:
        logger.warning("Failed to setup tracing hooks: %s", e)


def run_claude_with_tracing(args: Optional[list[str]] = None) -> None:
    """Run Claude CLI with MLflow tracing enabled.

    Args:
        args: Command line arguments to forward to claude, defaults to sys.argv[1:]
    """
    if args is None:
        args = sys.argv[1:]

    # Enable tracing for this session
    os.environ[MLFLOW_TRACING_ENABLED] = "true"

    try:
        # Set up hooks and find claude
        setup_claude_hooks()
        claude_path = find_claude_executable()

        # Run claude with the provided arguments
        result = subprocess.run([claude_path] + args, check=False)

        # Clean up environment
        os.environ[MLFLOW_TRACING_ENABLED] = "false"
        sys.exit(result.returncode)

    except Exception as e:
        logger.error("Error running claude: %s", e)
        os.environ[MLFLOW_TRACING_ENABLED] = "false"
        sys.exit(130 if isinstance(e, KeyboardInterrupt) else 1)


def main(args: Optional[list[str]] = None) -> None:
    """Main entry point for claude-mlflow wrapper.

    Args:
        args: Command line arguments to forward to claude
    """
    run_claude_with_tracing(args)


if __name__ == "__main__":
    main()
