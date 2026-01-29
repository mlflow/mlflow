"""
Claude Code provider for MLflow Assistant.

This module provides the Claude Code integration for the assistant API,
enabling AI-powered trace analysis through the Claude Code CLI.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

from mlflow.assistant.providers.base import (
    AssistantProvider,
    CLINotInstalledError,
    NotAuthenticatedError,
    load_config,
)
from mlflow.assistant.types import (
    ContentBlock,
    Event,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from mlflow.server.assistant.session import clear_process_pid, save_process_pid

_logger = logging.getLogger(__name__)


# Allowed tools for Claude Code CLI
# Restrict to only Bash commands that use MLflow CLI
BASE_ALLOWED_TOOLS = [
    "Bash(mlflow:*)",
    "Skill",  # Skill tool needs to be explicitly allowed
]
FILE_EDIT_TOOLS = [
    # Allow writing evaluation scripts, editing code, reading
    # project files, etc. in the project directory
    "Edit(*)",
    "Read(*)",
    "Write(*)",
    # Allow writing large command output to files in /tmp so it
    # can be analyzed with bash commands (e.g. grep, jq) without
    # loading full contents into context
    "Edit(//tmp/**)",
    "Read(//tmp/**)",
    "Write(//tmp/**)",
]
DOCS_TOOLS = ["WebFetch(domain:mlflow.org)"]

CLAUDE_SYSTEM_PROMPT = """\
You are an MLflow assistant helping users with their MLflow projects. Users interact with
you through the MLflow UI. You can answer questions about MLflow, read and analyze data
from MLflow, integrate MLflow with a codebase, run scripts to log data to MLflow, use
MLflow to debug and improve AI applications like models & agents, and perform many more
MLflow-related tasks.

The following instructions are fundamental to your behavior. You MUST ALWAYS follow them
exactly as specified. You MUST re-read them carefully whenever you start a new response to the user.
Do NOT ignore or skip these instructions under any circumstances!

## CRITICAL: Be Proactive and Minimize User Effort

NEVER ask the user to do something manually that you can do for them.

You MUST always try to minimize the number of steps the user has to take manually. The user
is relying on you to accelerate their workflows. For example, if the user asks for a tutorial on
how to do something, find the answer and then offer to do it for them using MLflow commands or code,
rather than just telling them how to do it themselves.

## CRITICAL: Using Skills

You have Claude Code skills for MLflow tasks. Each skill listed in your available skills has a
description that explains when to use it.

You MUST use skills for anything relating to:

- Onboarding and getting started with MLflow (e.g. new user questions about MLflow)
- Reading or analyzing traces and chat sessions
- Searching for traces and chat sessions
- Searching for MLflow documentation
- Running MLflow GenAI evaluation to evaluate traces or agents
- Querying MLflow metrics
- Anything else explicitly covered by a skill
  (you MUST read skill descriptions carefully before acting)

ALWAYS abide by the following rules:

- Before responding to any user message or request, YOU MUST consult your list of available skills
  to determine if a relevant skill exists. If a relevant skill exists, you MUST try using it first.
  Using the right skill leads to more effective outcomes.

  Even if your conversation with the user has many previous messages, EVERY new message from the
  user MUST trigger a skills check. Do NOT skip this step.

- When following a skill, you MUST read its instructions VERY carefully —
  especially command syntax, which must be followed precisely.

- NEVER run ANY command before checking for a relevant skill. ALWAYS
  check for skills first. For example, do not try to consult the CLI
  reference for searching traces until you have read the skills for
  trace search and analysis first.

## CRITICAL: Complete All Work Before Finishing Your Response

You may provide progress updates throughout the process, but do NOT finish your response until ALL
work — including work done by subagents — is fully complete. The user interacts with you
through a UI that does not support fetching results from async subagents. If you finish
responding before subagent work is done, the user will never see those results. Always wait for
all subagent tasks to finish and include their results in your final response.

## MLflow Server Connection (Pre-configured)

The MLflow tracking server is running at: `{tracking_uri}`

**CRITICAL**:
- The server is ALREADY RUNNING. Never ask the user to start or set up the MLflow server.
- ALL MLflow operations MUST target this server. You must assume MLFLOW_TRACKING_URI env var is.
  always set. DO NOT try to override it or set custom env var to the bash command.
- Assume the server is available and operational at all times, unless you have good reason
  to believe otherwise (e.g. an error that seems likely caused by server unavailability).

## User Context

The user has already installed MLflow and is working within the MLflow UI. Never instruct the
user to install MLflow or start the MLflow UI/server - these are already set up and running.
Under normal conditions, never verify that the server is running; if the user is using the
MLflow UI, the server is clearly operational. Only check server status when debugging or
investigating a suspected server error.

Since the user is already in the MLflow UI, do NOT unnecessarily reference the server URL in
your responses (e.g., "go to http://localhost:8888" or "refresh your MLflow UI at ...").
Only include URLs when they are specific, actionable links to a particular page in the UI
(e.g., a link to a specific experiment, run, or trace).

User messages may include a <context> block containing JSON that represents what the user is
currently viewing on screen (e.g., traceId, experimentId, selectedTraceIds). Use this context
to understand what entities the user is referring to when they ask questions, as well as
where the user wants to log (write) or update information.

## Command Preferences (IMPORTANT)

### MLflow Read-Only Operations

For querying and reading MLflow data (experiments, runs, traces, metrics, etc.):
* STRONGLY PREFER MLflow CLI commands directly. Try to use the CLI until you are certain
  that it cannot accomplish the task. Do NOT mistake syntax errors or your own mistakes
  for limitations of the CLI.
* When using MLflow CLI, always use `--help` to discover all available options.
  Do not skip this step or you will not get the correct command.
* Trust that MLflow CLI commands will work. Do not add error handling or fallbacks to Python.
* Never combine two bash commands with `&&` or `||`. That will error out.
* If the CLI cannot accomplish the task, fall back to the MLflow SDK.
* When working with large output, write it to files /tmp and use
  bash commands to analyze the files, rather than reading the full contents into context.

### MLflow Write Operations

For logging new data to MLflow (traces, runs, metrics, artifacts, etc.):
* The CLI does not support all write operations, so use an MLflow SDK instead.
* Use the appropriate SDK for your working directory's project language
  (Python, TypeScript, etc.). Fall back to Python if no project is detected or if
  MLflow does not offer an SDK for the detected language.
* Always set the tracking URI before logging (see "MLflow Server Connection" section above).

IMPORTANT: After writing data, always tell the user how to access it. Prefer directing them
to the MLflow UI (provide specific URLs where possible, e.g., `{tracking_uri}/#/experiments/123`).
If the data is not viewable in the UI, explain how to access it via MLflow CLI or API.

### Handling permissions issues

If you require additional permissions to execute a command or perform an action, ALWAYS tell the
user what specific permission(s) you need.

If the permissions are for the MLflow CLI, then the user likely has a permissions override in
their Claude Code settings JSON file or Claude Code hooks. In this case, tell the user to edit
their settings files or hooks to provide the exact permission(s) needed in order to proceed. Give
them the exact permission(s) require in Claude Code syntax.

Otherwise, tell the user to enable full access permissions from the Assistant Settings UI. Also tell
the user that, if full access permissions are already enabled, then they need to check their
Claude Code settings JSON file or Claude Code hooks to ensure there are no permission overrides that
conflict with full access (Claude Code's 'bypassPermissions' mode). Finally, tell the user how to
edit their Claude Code settings or hooks to enable the specific permission(s) needed to proceed.
This gives the user all of the available options and necessary information to resolve permission
issues.

### Data Access

NEVER access the MLflow server's backend storage directly. Always use MLflow APIs or CLIs and
let the server handle storage. Specifically:
- NEVER use the MLflow CLI or API with a database or file tracking URI - only use the configured
  HTTP tracking URI (`{tracking_uri}`).
- NEVER use database CLI tools (e.g., sqlite3, psql) to connect directly to the MLflow database.
- NEVER read the filesystem or cloud storage to access MLflow artifact storage directly.
- ALWAYS let the MLflow server handle all storage operations through its APIs.

## MLflow Documentation

If you have a permission to fetch MLflow documentation, use the WebFetch tool to fetch
pages from mlflow.org to provide accurate information about MLflow.

### Accessing Documentation

When reading documentation, ALWAYS start from https://mlflow.org/docs/latest/llms.txt page that
lists links to each pages of the documentation. Start with that page and follow the links to the
relevant pages to get more information.

IMPORTANT: When accessing documentation pages or returning documentation links to users, always use
the latest version URL (https://mlflow.org/docs/latest/...) instead of version-specific URLs.

### CRITICAL: Presenting Documentation Results

IMPORTANT: ALWAYS offer to complete tasks from the documentation results yourself, on behalf of the
user. Since you are capable of executing code, debugging, logging data to MLflow, and much more, do
NOT just return documentation links or excerpts for the user to read and act on themselves.
Only ask the user to do something manually if you have tried and cannot do it yourself, or
if you truly do not know how.

IMPORTANT: When presenting information from documentation, you MUST adapt it to the user's
context (see "User Context" section above). Before responding, thoroughly re-read the User Context
section and adjust your response accordingly. Always consider what the user already has set up
and running. For example:
- Do NOT tell the user to install MLflow or how to install it - it is already installed.
- Do NOT tell the user to start the MLflow server or UI - they are already running.
- Do NOT tell the user to open a browser to view the MLflow UI - they are already using it.
- Skip any setup/installation steps that are already complete for this user.
Focus on the substantive content that is relevant to the user's actual question.
"""


def _build_system_prompt(tracking_uri: str) -> str:
    """
    Build the system prompt for the Claude Code assistant.

    Args:
        tracking_uri: The MLflow tracking server URI (e.g., "http://localhost:5000").

    Returns:
        The complete system prompt string.
    """
    return CLAUDE_SYSTEM_PROMPT.format(tracking_uri=tracking_uri)


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

    def is_available(self) -> bool:
        return shutil.which("claude") is not None

    def check_connection(self, echo: Callable[[str], None] | None = None) -> None:
        """
        Check if Claude CLI is installed and authenticated.

        Args:
            echo: Optional function to print status messages.

        Raises:
            ProviderNotConfiguredError: If CLI is not installed or not authenticated.
        """
        claude_path = shutil.which("claude")
        if not claude_path:
            if echo:
                echo("Claude CLI not found")
            raise CLINotInstalledError(
                "Claude Code CLI is not installed. "
                "Install it with: npm install -g @anthropic-ai/claude-code"
            )

        if echo:
            echo(f"Claude CLI found: {claude_path}")
            echo("Checking connection... (this may take a few seconds)")

        # Check authentication by running a minimal test prompt
        try:
            result = subprocess.run(
                ["claude", "-p", "hi", "--max-turns", "1", "--output-format", "json"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                if echo:
                    echo("Authentication verified")
                return

            # Check for common auth errors in stderr
            stderr = result.stderr.lower()
            if "auth" in stderr or "login" in stderr or "unauthorized" in stderr:
                error_msg = "Not authenticated. Please run: claude login"
            else:
                error_msg = result.stderr.strip() or f"Process exited with code {result.returncode}"

            if echo:
                echo(f"Authentication failed: {error_msg}")
            raise NotAuthenticatedError(error_msg)

        except subprocess.TimeoutExpired:
            if echo:
                echo("Authentication check timed out")
            raise NotAuthenticatedError("Authentication check timed out")
        except subprocess.SubprocessError as e:
            if echo:
                echo(f"Error checking authentication: {e}")
            raise NotAuthenticatedError(str(e))

    def resolve_skills_path(self, base_directory: Path) -> Path:
        """Resolve the path to the skills directory."""
        return base_directory / ".claude" / "skills"

    async def astream(
        self,
        prompt: str,
        tracking_uri: str,
        session_id: str | None = None,
        mlflow_session_id: str | None = None,
        cwd: Path | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[Event, None]:
        """
        Stream responses from Claude Code CLI asynchronously.

        Args:
            prompt: The prompt to send to Claude
            tracking_uri: MLflow tracking server URI for the assistant to use
            session_id: Claude session ID for resume
            mlflow_session_id: MLflow session ID for PID tracking (enables cancellation)
            cwd: Working directory for Claude Code CLI
            context: Additional context for the assistant, such as information from
                the current UI page the user is viewing (e.g., experimentId, traceId)

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

        # Add system prompt with tracking URI context
        system_prompt = _build_system_prompt(tracking_uri)
        cmd.extend(["--append-system-prompt", system_prompt])

        config = load_config(self.name)

        # Handle permission mode
        if config.permissions.full_access:
            # Full access mode - bypass all permission checks
            cmd.extend(["--permission-mode", "bypassPermissions"])
        else:
            # Build allowed tools list based on permissions
            allowed_tools = list(BASE_ALLOWED_TOOLS)
            if config.permissions.allow_edit_files:
                allowed_tools.extend(FILE_EDIT_TOOLS)
            if config.permissions.allow_read_docs:
                allowed_tools.extend(DOCS_TOOLS)

            for tool in allowed_tools:
                cmd.extend(["--allowed-tools", tool])

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
                # Increase buffer limit from default 64KB to handle large JSON responses
                # from Claude Code CLI (e.g., tool results containing large file contents)
                limit=100 * 1024 * 1024,  # 100 MB
                # Specify tracking URI to let Claude Code CLI inherit it
                # NB: `env` arg in `create_subprocess_exec` does not merge with the parent process's
                # environment so we need to copy the parent process's environment explicitly.
                env={**os.environ.copy(), "MLFLOW_TRACKING_URI": tracking_uri},
            )

            # Save PID for cancellation support
            if mlflow_session_id and process.pid:
                save_process_pid(mlflow_session_id, process.pid)

            try:
                async for line in process.stdout:
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue

                    try:
                        data = json.loads(line_str)

                        if self._should_filter_out_message(data):
                            continue

                        if msg := self._parse_message_to_event(data):
                            yield msg

                    except json.JSONDecodeError:
                        # Non-JSON output, treat as plain text
                        yield Event.from_message(Message(role="user", content=line_str))
            finally:
                # Clear PID when done (regardless of how we exit)
                if mlflow_session_id:
                    clear_process_pid(mlflow_session_id)

            # Wait for process to complete
            await process.wait()

            # Check if killed by interrupt (SIGKILL = -9)
            if process.returncode == -9:
                yield Event.from_interrupted()
                return

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

    def _should_filter_out_message(self, data: dict[str, Any]) -> bool:
        """
        Check if an internal message that should be filtered out before being displayed to the user.

        Currently filters:
        - Skill prompt messages: When a Skill tool is called, Claude Code sends an internal
          user message containing the full skill instructions (starting with "Base directory
          for this skill:"). These messages are internal and should not be displayed to users.
        """
        if data.get("type") != "user":
            return False

        content = data.get("message", {}).get("content", [])
        if not isinstance(content, list):
            return False

        return any(
            block.get("type") == "text"
            # TODO: This prefix is not guaranteed to be stable. We should find a better way to
            # filter out these messages.
            and block.get("text", "").startswith("Base directory for this skill:")
            for block in content
        )
