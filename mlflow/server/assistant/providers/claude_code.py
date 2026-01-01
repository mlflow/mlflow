"""
Claude Code provider for MLflow Assistant.

This module provides the Claude Code CLI integration for the assistant API,
enabling AI-powered trace analysis through the Claude Code CLI.
"""

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any, AsyncGenerator

from mlflow.server.assistant.providers.base import AssistantProvider

_logger = logging.getLogger(__name__)

CLAUDE_CONFIG_FILE = Path.home() / ".mlflow" / "claude-config.json"


class ClaudeCodeProvider(AssistantProvider):
    """Assistant provider using Claude Code CLI."""

    @property
    def name(self) -> str:
        return "claude_code"

    def is_available(self) -> bool:
        """Check if Claude Code CLI is available."""
        return shutil.which("claude") is not None

    def load_config(self) -> dict[str, Any]:
        """Load Claude config from file."""
        if CLAUDE_CONFIG_FILE.exists():
            try:
                with open(CLAUDE_CONFIG_FILE) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    async def run(
        self,
        prompt: str,
        session_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Run Claude Code CLI and stream responses.

        Args:
            prompt: The prompt to send to Claude
            session_id: Claude session ID for resume

        Yields:
            Event dictionaries with type and data
        """
        claude_path = shutil.which("claude")
        if not claude_path:
            yield {
                "type": "error",
                "data": {"error": "Claude CLI not found. Please install Claude Code CLI."},
            }
            return

        config = self.load_config()
        cwd = config.get("projectPath")
        model = config.get("model")

        # NB: --verbose is required when using --output-format=stream-json with -p
        cmd = [claude_path, "-p", prompt, "--output-format", "stream-json", "--verbose"]

        if model and model != "default":
            cmd.extend(["--model", model])

        if session_id:
            cmd.extend(["--resume", session_id])

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            new_session_id = None
            async for line in process.stdout:
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                try:
                    data = json.loads(line_str)
                    msg_type = data.get("type", "")

                    if "session_id" in data:
                        new_session_id = data["session_id"]

                    if msg_type == "assistant":
                        content = data.get("message", {}).get("content", [])
                        text_parts = [
                            block.get("text", "")
                            for block in content
                            if block.get("type") == "text"
                        ]
                        if text_parts:
                            text = " ".join(text_parts)
                            yield {"type": "message", "data": {"text": text}}

                    elif msg_type == "result":
                        yield {
                            "type": "done",
                            "data": {"status": "complete", "session_id": new_session_id},
                        }

                    elif msg_type == "error":
                        error_msg = data.get("error", {}).get("message", "Unknown error")
                        yield {"type": "error", "data": {"error": error_msg}}

                except json.JSONDecodeError:
                    yield {"type": "message", "data": {"text": line_str}}

            await process.wait()

            if process.returncode != 0:
                stderr = await process.stderr.read()
                error_msg = (
                    stderr.decode("utf-8").strip()
                    or f"Process exited with code {process.returncode}"
                )
                yield {"type": "error", "data": {"error": error_msg}}

        except Exception as e:
            _logger.exception("Error running Claude Code CLI")
            yield {"type": "error", "data": {"error": str(e)}}
