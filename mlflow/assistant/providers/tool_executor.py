import asyncio
import logging
import os
import shlex
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from mlflow.assistant.config import PermissionsConfig
from mlflow.assistant.custom_view import RENDER_CUSTOM_VIEW_TOOL_NAME
from mlflow.assistant.providers.base import assistant_sandbox_enabled

_logger = logging.getLogger(__name__)


def _uri_without_credentials(name: str, uri: str) -> str | None:
    """Return ``uri`` only when it carries no embedded credentials, else ``None``.

    ``MLFLOW_TRACKING_URI`` / ``MLFLOW_REGISTRY_URI`` can be SQLAlchemy database URIs that embed
    credentials in the netloc (``user:password@host``). Forwarding one with userinfo into the
    sandbox would leak host credentials to sandboxed commands, defeating the isolation the sandbox
    exists for, so a URI with credentials (or one that cannot be parsed to confirm it has none) is
    dropped and logged rather than passed through.
    """
    try:
        parts = urlsplit(uri)
    except ValueError:
        _logger.warning("Not forwarding %s to the sandbox: its URI could not be parsed.", name)
        return None
    if parts.username or parts.password:
        _logger.warning("Not forwarding %s to the sandbox: it contains embedded credentials.", name)
        return None
    return uri


_FILE_TOOLS = {"Read", "Write", "Edit"}
# Restricted mode only permits MLflow CLI and Python; anything else needs Full Access.
_ALLOWED_BASH_COMMANDS = {"mlflow", "python3", "python"}

# Tools executed on the CLIENT (browser), not the server: the assistant loop pauses the turn and
# waits for a client-submitted result instead of routing the call through execute_tool/the static
# permission gate. See openai_compatible.py's tool loop.
CLIENT_TOOLS = {RENDER_CUSTOM_VIEW_TOOL_NAME}


def _is_path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve_file_path(raw_path: str, cwd: Path | None) -> Path:
    p = Path(raw_path).expanduser()
    if not p.is_absolute() and cwd:
        p = cwd / p
    return p.resolve()


def static_permission_error(
    tool_name: str,
    tool_input: dict[str, Any],
    perms: PermissionsConfig,
    cwd: Path | None,
) -> str | None:
    """Return a denial message if the call is NOT permitted under static (non-full-access)
    permissions, or None if it is allowed.

    Shared by ``execute_tool`` (to enforce the policy) and the assistant's per-call permission gate
    (to decide whether an interactive prompt is even needed): a call the static policy already
    allows — e.g. an ``mlflow`` CLI command or an in-workspace file op — runs without prompting,
    just as it did before tool-call permissions existed.
    """
    if perms.full_access:
        return None

    if not isinstance(tool_input, dict):
        # The model's function-call "arguments" string is parsed with json.loads and
        # passed through as-is (mlflow/assistant/providers/openai_compatible.py); any
        # syntactically valid JSON that isn't an object (e.g. "[]", "null", "\"x\"")
        # reaches here unchanged, and every check below calls tool_input.get(...),
        # which would raise AttributeError on a non-dict value rather than returning a
        # denial.
        return "Permission denied: malformed tool input"

    if tool_name == "Bash":
        command = tool_input.get("command", "")
        if not isinstance(command, str):
            # Malformed tool-call JSON (e.g. a model emitting {"command": 123} or
            # {"command": null}) would otherwise raise AttributeError on .strip()
            # below, escaping this function instead of returning a denial.
            return "Permission denied: malformed command"
        command = command.strip()
        try:
            argv = shlex.split(command)
        except ValueError:
            return "Permission denied: malformed command"
        if not argv or argv[0] not in _ALLOWED_BASH_COMMANDS:
            return (
                f"Permission denied: only {', '.join(sorted(_ALLOWED_BASH_COMMANDS))} "
                "commands are allowed"
            )
        # python/python3 can run arbitrary code, including reading any file the
        # process can access, so require the same configured project directory
        # Read/Write/Edit do below. Without this, GHSA-27c7-qx3r-x4f8's impact
        # (arbitrary file read when cwd is None) is reachable via
        # Bash("python3 -c \"print(open(path).read())\"") even though Read
        # itself is denied.
        if argv[0] in {"python", "python3"} and cwd is None:
            return f"Permission denied: {argv[0]} requires a configured project directory"

    if tool_name in _FILE_TOOLS and not perms.allow_edit_files:
        return f"Permission denied: {tool_name} is not allowed"

    if tool_name in {"Write", "Edit"} and not cwd:
        return f"Permission denied: {tool_name} requires a configured project directory"

    if tool_name in _FILE_TOOLS:
        if raw_path := tool_input.get("file_path") or tool_input.get("path", ""):
            if cwd is None:
                return f"Permission denied: {tool_name} requires a configured project directory"
            try:
                target = _resolve_file_path(raw_path, cwd)
            except (ValueError, OSError, TypeError):
                # e.g. an embedded NUL byte (ValueError/OSError), or a non-string
                # file_path such as a list or int from malformed tool-call JSON
                # (TypeError from the Path() constructor itself): Path.resolve()
                # raises rather than returning a path, so this must be caught here
                # rather than left to propagate out of execute_tool, which only wraps
                # the tool dispatch (below) in try/except, not this static check.
                return f"Permission denied: malformed path {raw_path!r}"
            if not _is_path_within(target, cwd):
                return f"Permission denied: path {raw_path} is outside the workspace {cwd}"

    return None


async def execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    cwd: Path | None = None,
    tracking_uri: str | None = None,
    permissions: PermissionsConfig | None = None,
) -> tuple[str, bool]:
    perms = permissions or PermissionsConfig()

    if (denial := static_permission_error(tool_name, tool_input, perms, cwd)) is not None:
        return denial, True

    try:
        match tool_name:
            case "Bash":
                return await _execute_bash(
                    tool_input, cwd=cwd, tracking_uri=tracking_uri, full_access=perms.full_access
                )
            case "Read":
                return await asyncio.to_thread(_execute_read, tool_input, cwd=cwd)
            case "Write":
                return await asyncio.to_thread(_execute_write, tool_input, cwd=cwd)
            case "Edit":
                return await asyncio.to_thread(_execute_edit, tool_input, cwd=cwd)
            case _:
                return f"Unknown tool: {tool_name}", True
    except Exception as e:
        _logger.exception("Tool execution error for %s", tool_name)
        return f"Tool execution failed: {e}", True


async def _execute_bash(
    tool_input: dict[str, Any],
    cwd: Path | None,
    tracking_uri: str | None,
    full_access: bool = False,
) -> tuple[str, bool]:
    # Stripped the same way static_permission_error strips before its own shlex.split, so
    # the two can't tokenize argv[0] differently (e.g. a leading non-ASCII whitespace
    # character that str.strip() removes but shlex's default whitespace set does not).
    command = tool_input.get("command", "")
    if not isinstance(command, str):
        return "No command provided", True
    command = command.strip()
    if not command:
        return "No command provided", True

    if assistant_sandbox_enabled():
        return await _execute_bash_in_sandbox(command, cwd, tracking_uri, full_access)
    return await _execute_bash_on_host(command, cwd, tracking_uri, full_access)


async def _execute_bash_on_host(
    command: str,
    cwd: Path | None,
    tracking_uri: str | None,
    full_access: bool,
) -> tuple[str, bool]:
    env = os.environ.copy()
    if tracking_uri:
        env["MLFLOW_TRACKING_URI"] = tracking_uri

    try:
        if full_access:
            # Shell required: LLM-generated commands may use pipes, redirects, or && chaining.
            # Safe here because full access has no allowlist to bypass in the first place.
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
        else:
            # Restricted mode: static_permission_error only validates argv[0] against the
            # allowlist. Running the raw string through a shell would let shell operators
            # after argv[0] (&&, ;, |, `` `` , $()) smuggle in commands the allowlist never
            # saw, e.g. "mlflow --help && python3 -c '...'" passes the argv[0] check but a
            # shell would still execute the chained python3 call. Executing the
            # already-validated argv directly, with no shell, means anything after argv[0]
            # is passed as a literal argument to the allowlisted program and never
            # interpreted as a separate command.
            try:
                argv = shlex.split(command)
            except ValueError:
                return "Permission denied: malformed command", True
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        output = stdout.decode("utf-8", errors="replace")
        err_output = stderr.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            result = (
                output + err_output if output or err_output else f"Exit code: {proc.returncode}"
            )
            return result.strip(), True

        return (output + err_output).strip() or "(no output)", False
    except asyncio.TimeoutError:
        return "Command timed out after 120 seconds", True


async def _execute_bash_in_sandbox(
    command: str,
    cwd: Path | None,
    tracking_uri: str | None,
    full_access: bool,
) -> tuple[str, bool]:
    """Run the command inside a hardened Docker container instead of on the host.

    The restricted/full-access distinction is preserved: full access runs the command
    through a shell (there is no allowlist to bypass), while restricted mode runs the
    already-validated argv directly with no shell, matching the host path. The container
    itself is the hard boundary; the static permission policy remains defense-in-depth.
    """
    from mlflow.server.sandbox import (
        SandboxUnavailableError,
        run_in_sandbox,
        to_container_host_uri,
    )

    # Start from an empty environment rather than os.environ.copy(): isolating host
    # credentials (e.g. DATABRICKS_TOKEN, cloud keys) from sandboxed commands is a primary
    # reason for the sandbox, so they are intentionally NOT forwarded. Only non-secret MLflow
    # configuration a restricted `mlflow` command needs is passed through, loopback-rewritten
    # so it resolves from inside the container.
    env = {}
    if tracking_uri and (safe := _uri_without_credentials("MLFLOW_TRACKING_URI", tracking_uri)):
        env["MLFLOW_TRACKING_URI"] = to_container_host_uri(safe)
    for var in ("MLFLOW_REGISTRY_URI",):
        if (value := os.environ.get(var)) and (safe := _uri_without_credentials(var, value)):
            env[var] = to_container_host_uri(safe)

    if full_access:
        sandbox_command = [command]
        use_shell = True
    else:
        try:
            argv = shlex.split(command)
        except ValueError:
            return "Permission denied: malformed command", True
        sandbox_command = argv
        use_shell = False

    try:
        # run_in_sandbox uses the blocking docker-py client, so run it off the event loop.
        result = await asyncio.to_thread(
            run_in_sandbox,
            sandbox_command,
            workdir=cwd,
            environment=env,
            timeout=120,
            use_shell=use_shell,
        )
    except SandboxUnavailableError as e:
        # Do not silently fall back to host execution: that would defeat the sandbox the
        # operator explicitly enabled.
        return f"Sandbox is enabled but the command could not be run: {e}", True

    if result.timed_out:
        return "Command timed out after 120 seconds", True
    output = result.output.strip()
    if result.exit_code != 0:
        return output or f"Exit code: {result.exit_code}", True
    return output or "(no output)", False


def _execute_read(tool_input: dict[str, Any], cwd: Path | None = None) -> tuple[str, bool]:
    file_path = tool_input.get("file_path") or tool_input.get("path", "")
    if not file_path:
        return "No file_path provided", True
    try:
        content = _resolve_file_path(file_path, cwd).read_text(encoding="utf-8")
        return content, False
    except Exception as e:
        return str(e), True


def _execute_write(tool_input: dict[str, Any], cwd: Path | None = None) -> tuple[str, bool]:
    file_path = tool_input.get("file_path") or tool_input.get("path", "")
    content = tool_input.get("content", "")
    if not file_path:
        return "No file_path provided", True
    try:
        p = _resolve_file_path(file_path, cwd)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {file_path}", False
    except Exception as e:
        return str(e), True


def _execute_edit(tool_input: dict[str, Any], cwd: Path | None = None) -> tuple[str, bool]:
    file_path = tool_input.get("file_path") or tool_input.get("path", "")
    old_string = tool_input.get("old_string", "")
    new_string = tool_input.get("new_string", "")
    if not file_path:
        return "No file_path provided", True
    try:
        p = _resolve_file_path(file_path, cwd)
        content = p.read_text(encoding="utf-8")
        if old_string not in content:
            return f"old_string not found in {file_path}", True
        new_content = content.replace(old_string, new_string, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"Edited {file_path}", False
    except Exception as e:
        return str(e), True


def build_tools_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": (
                    "Execute a shell command to query or interact with MLflow. "
                    "Use 'mlflow' CLI commands or Python one-liners with the MLflow SDK."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute.",
                        }
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read the contents of a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute or relative path to the file.",
                        }
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Write",
                "description": "Write content to a file (creates or overwrites).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute or relative path to the file.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write.",
                        },
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Edit",
                "description": (
                    "Replace the first occurrence of old_string with new_string in a file."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute or relative path to the file.",
                        },
                        "old_string": {
                            "type": "string",
                            "description": "Exact string to find.",
                        },
                        "new_string": {
                            "type": "string",
                            "description": "String to replace it with.",
                        },
                    },
                    "required": ["file_path", "old_string", "new_string"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": RENDER_CUSTOM_VIEW_TOOL_NAME,
                "description": (
                    "Render a custom trace view in the UI: a reusable, trace-agnostic layout of "
                    "cards, stat tiles, key-value viewers, and assessment boards, built from the "
                    "current trace's data. Call this once you've designed the layout; the client "
                    "renders it and reports back whether it applied successfully."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Short display title for the view.",
                        },
                        "messages": {
                            "type": "array",
                            "description": (
                                "A2UI message list describing the view's component tree."
                            ),
                            "items": {"type": "object"},
                        },
                    },
                    "required": ["title", "messages"],
                },
            },
        },
    ]
