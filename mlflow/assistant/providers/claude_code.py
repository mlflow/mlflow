"""
Claude Code provider for MLflow Assistant.

This module provides the Claude Code integration for the assistant API,
enabling AI-powered trace analysis through the Claude Code CLI.
"""

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any, AsyncGenerator

from mlflow.assistant.providers.base import MLFLOW_ASSISTANT_HOME, AssistantProvider, ProviderConfig
from mlflow.assistant.types import (
    ContentBlock,
    Event,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

_logger = logging.getLogger(__name__)


class ClaudeCodeAssistantConfig(ProviderConfig):
    model: str = "default"
    project_path: str | None = None


# TODO: to be updated
CLAUDE_SYSTEM_PROMPT = """You are an MLflow assistant helping users with their MLflow projects."""


class ClaudeCodeProvider(AssistantProvider):
    """Assistant provider using Claude Code CLI."""

    @property
    def name(self) -> str:
        return "claude_code"

    @property
    def config_path(self) -> Path:
        return MLFLOW_ASSISTANT_HOME / "claude-config.json"

    def is_available(self) -> bool:
        return shutil.which("claude") is not None

    def load_config(self) -> ProviderConfig:
        # Use default config if no config file exists
        if not self.config_path.exists():
            return ClaudeCodeAssistantConfig()

        with open(self.config_path) as f:
            return ClaudeCodeAssistantConfig.model_validate_json(f.read())

    async def astream(
        self,
        prompt: str,
        session_id: str | None = None,
        cwd: Path | None = None,
    ) -> AsyncGenerator[Event, None]:
        """
        Stream responses from Claude Code CLI asynchronously.

        Args:
            prompt: The prompt to send to Claude
            session_id: Claude session ID for resume
            cwd: Working directory for the session

        Yields:
            Event objects
        """
        claude_path = shutil.which("claude")
        if not claude_path:
            yield Event.from_error(
                "Claude CLI not found. Please install Claude Code CLI and ensure it's in your PATH."
            )
            return

        # Build command
        # Note: --verbose is required when using --output-format=stream-json with -p
        cmd = [claude_path, "-p", prompt, "--output-format", "stream-json", "--verbose"]

        # Add system prompt
        cmd.extend(["--append-system-prompt", CLAUDE_SYSTEM_PROMPT])

        config = self.load_config()
        if config.model and config.model != "default":
            cmd.extend(["--model", config.model])

        if session_id:
            cmd.extend(["--resume", session_id])

        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            async for line in process.stdout:
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                try:
                    data = json.loads(line_str)
                    if msg := self._parse_message_to_event(data):
                        yield msg

                except json.JSONDecodeError:
                    # Non-JSON output, treat as plain text
                    yield Event.from_message(Message(role="user", content=line_str))

            # Wait for process to complete
            await process.wait()

            if process.returncode != 0:
                stderr = await process.stderr.read()
                error_msg = (
                    stderr.decode("utf-8").strip()
                    or f"Process exited with code {process.returncode}"
                )
                yield Event.from_error(error_msg)

        except Exception as e:
            _logger.exception("Error running Claude Code CLI")
            yield Event.from_error(str(e))
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()

    def _parse_message_to_event(self, data: dict[str, Any]) -> Event | None:
        """
        Parse json message from Claude Code CLI output.

        Reference: https://github.com/anthropics/claude-agent-sdk-python/blob/29c12cd80b256e88f321b2b8f1f5a88445077aa5/src/claude_agent_sdk/_internal/message_parser.py#L24

        Args:
            data: Raw message dictionary from CLI output

        Returns:
            Parsed Event object
        """
        message_type = data.get("type")
        if not message_type:
            return Event.from_error("Message missing 'type' field")

        match message_type:
            case "user":
                try:
                    if isinstance(data["message"]["content"], list):
                        user_content_blocks = []
                        for block in data["message"]["content"]:
                            match block["type"]:
                                case "text":
                                    user_content_blocks.append(TextBlock(text=block["text"]))
                                case "tool_use":
                                    user_content_blocks.append(
                                        ToolUseBlock(
                                            id=block["id"],
                                            name=block["name"],
                                            input=block["input"],
                                        )
                                    )
                                case "tool_result":
                                    user_content_blocks.append(
                                        ToolResultBlock(
                                            tool_use_id=block["tool_use_id"],
                                            content=block.get("content"),
                                            is_error=block.get("is_error"),
                                        )
                                    )
                            msg = Message(role="user", content=user_content_blocks)
                    else:
                        msg = Message(role="user", content=data["message"]["content"])
                    return Event.from_message(msg)
                except KeyError as e:
                    return Event.from_error(f"Failed to parse user message: {e}")

            case "assistant":
                try:
                    if data["message"].get("error"):
                        return Event.from_error(data["message"]["error"])

                    content_blocks: list[ContentBlock] = []
                    for block in data["message"]["content"]:
                        match block["type"]:
                            case "text":
                                content_blocks.append(TextBlock(text=block["text"]))
                            case "thinking":
                                content_blocks.append(
                                    ThinkingBlock(
                                        thinking=block["thinking"],
                                        signature=block["signature"],
                                    )
                                )
                            case "tool_use":
                                content_blocks.append(
                                    ToolUseBlock(
                                        id=block["id"],
                                        name=block["name"],
                                        input=block["input"],
                                    )
                                )
                            case "tool_result":
                                content_blocks.append(
                                    ToolResultBlock(
                                        tool_use_id=block["tool_use_id"],
                                        content=block.get("content"),
                                        is_error=block.get("is_error"),
                                    )
                                )

                    msg = Message(role="assistant", content=content_blocks)
                    return Event.from_message(msg)
                except KeyError as e:
                    return Event.from_error(f"Failed to parse assistant message: {e}")

            case "system":
                # NB: Skip system message. The system message from Claude Code CLI contains
                # the various metadata about runtime, which is not used by the assistant UX.
                return None

            case "error":
                try:
                    error_msg = data.get("error", {}).get("message", str(data.get("error")))
                    return Event.from_error(error_msg)
                except Exception as e:
                    return Event.from_error(f"Failed to parse error message: {e}")

            case "result":
                try:
                    return Event.from_result(
                        result=data.get("result"),
                        session_id=data["session_id"],
                    )
                except KeyError as e:
                    return Event.from_error(f"Failed to parse result message: {e}")

            case "stream_event":
                try:
                    return Event.from_stream_event(event=data["event"])
                except KeyError as e:
                    return Event.from_error(f"Failed to parse stream_event message: {e}")

            case _:
                return Event.from_error(f"Unknown message type: {message_type}")
