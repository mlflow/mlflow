"""
Claude Code provider for MLflow Assistant.

This module provides the Claude Code CLI integration for the assistant API,
enabling AI-powered trace analysis through the Claude Code CLI.
"""

import asyncio
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

from mlflow.assistant.providers.base import AssistantProvider, load_config
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

# Allowed tools for Claude Code CLI
# Restrict to only Bash commands that use MLflow CLI
ALLOWED_TOOLS = [
    "Bash(mlflow:*)",
]

# TODO: to be updated
CLAUDE_SYSTEM_PROMPT = """You are an MLflow assistant helping users with their MLflow projects.

User messages may include a <context> block containing JSON that represents what the user is currently viewing on screen (e.g., traceId, experimentId, selectedTraceIds). Use this context to understand what entities the user is referring to when they ask questions.

## Skills

You have access to MLflow-specific skills that provide guidance on using CLI commands.

## Command Preferences (IMPORTANT)

* When working with MLflow data and operations, ALWAYS use MLflow CLI commands directly.
* Never combine two bash command with `&&` or `||`. That will error out.
* Trust that MLflow CLI commands will work. Do not add error handling or fallbacks to Python.
* When using MLflow CLI, always use `--help` to discover all available
options. Do not skip this step or you will not get the correct command.
"""

class ClaudeCodeProvider(AssistantProvider):
    """Assistant provider using Claude Code CLI."""

    @property
    def name(self) -> str:
        return "claude_code"

    @property
    def display_name(self) -> str:
        return "Claude Code"

    @property
    def description(self) -> str:
        return "AI-powered assistant using Claude Code CLI"

    @property
    def skill_path(self) -> Path:
        return Path.home() / ".claude" / "skills"

    def is_available(self) -> bool:
        return shutil.which("claude") is not None

    def check_connection(self, echo: Callable[[str], None] = print) -> None:
        """
        Check if Claude CLI is installed and authenticated.

        Args:
            echo: Function to print status messages. Defaults to print().

        Raises:
            RuntimeError: If Claude CLI is not installed.
        """
        claude_path = shutil.which("claude")
        if not claude_path:
            raise RuntimeError(
                "Claude CLI not found.\n\n"
                "Please install Claude CLI first:\n"
                "  npm install -g @anthropic-ai/claude-code\n\n"
                "Then authenticate:\n"
                "  claude login"
            )

        echo(f"Claude CLI found: {claude_path}")

        # Check version
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            version = result.stdout.strip() or result.stderr.strip()
            echo(f"Version: {version}")
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            echo("(Could not determine version)")

    def install_skills(self, skills: list[str]) -> list[str]:
        """Install MLflow-specific Claude skills."""
        # Get the skills directory from this package
        skills_source = Path(__file__).parent.parent / "skills"
        skills_dest = Path.home() / ".claude" / "skills"

        if not skills_source.exists():
            raise RuntimeError("Skills directory not found")

        # Create destination directory
        skills_dest.mkdir(parents=True, exist_ok=True)

        # Find all skill directories in the skills directory
        skill_dirs = list(skills_source.glob("*"))
        if not skill_dirs:
            raise RuntimeError("No skills to install")

        installed_skills = []
        for skill_dir in skill_dirs:
            shutil.copytree(skill_dir, skills_dest / skill_dir.name, dirs_exist_ok=True)
            installed_skills.append(skill_dir.name)

        return installed_skills

    async def astream(
        self,
        prompt: str,
        session_id: str | None = None,
        cwd: Path | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[Event, None]:
        """
        Stream responses from Claude Code CLI asynchronously.

        Args:
            prompt: The prompt to send to Claude
            session_id: Claude session ID for resume
            cwd: Working directory for Claude Code CLI
            context: Page context (experimentId, traceId, selectedTraceIds, etc.)

        Yields:
            Event objects
        """
        claude_path = shutil.which("claude")
        if not claude_path:
            yield Event.from_error(
                "Claude CLI not found. Please install Claude Code CLI and ensure it's in your PATH."
            )
            return

        # Build user message with context
        if context:
            user_message = f"<context>\n{json.dumps(context)}\n</context>\n\n{prompt}"
        else:
            user_message = prompt

        # Build command
        # Note: --verbose is required when using --output-format=stream-json with -p
        cmd = [claude_path, "-p", user_message, "--output-format", "stream-json", "--verbose"]

        # Add system prompt
        cmd.extend(["--append-system-prompt", CLAUDE_SYSTEM_PROMPT])

        # Add allowed tools restriction
        for tool in ALLOWED_TOOLS:
            cmd.extend(["--allowed-tools", tool])

        config = load_config(self.name)
        if config.model and config.model != "default":
            cmd.extend(["--model", config.model])

        if session_id:
            cmd.extend(["--resume", session_id])

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
