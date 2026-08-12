import asyncio
import logging
import os
import shlex
from pathlib import Path
from typing import Any

from mlflow.assistant.config import PermissionsConfig

_logger = logging.getLogger(__name__)

_FILE_TOOLS = {"Read", "Write", "Edit"}
# Restricted mode only permits MLflow CLI and Python; anything else needs Full Access.
_ALLOWED_BASH_COMMANDS = {"mlflow", "python3", "python"}

RENDER_CUSTOM_VIEW_TOOL_NAME = "render_custom_view"
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


def _extract_option_value(argv: list[str], names: set[str]) -> str | None:
    """Return the value passed to any of the given CLI option names (e.g. {"-u",
    "--artifact-uri"}), handling both "--name value" and "--name=value" forms. None if
    the option isn't present or has no value.
    """
    for i, tok in enumerate(argv):
        if tok in names and i + 1 < len(argv):
            return argv[i + 1]
        for name in names:
            if tok.startswith(name + "="):
                return tok[len(name) + 1 :]
    return None


def _mlflow_subcommand_denial(argv: list[str], cwd: Path | None) -> str | None:
    """Deny mlflow CLI subcommands that can touch the local filesystem or execute code
    independently of any configured project directory, unlike ordinary read-only
    commands (e.g. ``--version``, ``experiments search``), which stay usable without one.
    """
    subcommand = argv[1] if len(argv) > 1 else None

    # "mlflow run <uri>" executes the target project's entry-point ``command:`` via a
    # real shell (see mlflow/projects/backend/local.py), regardless of cwd. There's no
    # argument to validate here: the entry point is arbitrary shell content by design,
    # so this is denied outright rather than gated like the checks below.
    if subcommand == "run":
        return "Permission denied: mlflow run is not allowed outside full access"

    if subcommand == "artifacts":
        artifacts_subcommand = argv[2] if len(argv) > 2 else None
        # "list" only returns metadata for a run's already-tracked artifacts; it can't
        # reach an arbitrary local path, so it needs no extra check.
        if artifacts_subcommand not in {"download", "log-artifact", "log-artifacts"}:
            return None
        # download/log-artifact/log-artifacts read or write local filesystem paths that
        # mlflow does not confine to any workspace on its own, so require the same
        # configured project directory Read/Write/Edit/python do.
        if cwd is None:
            return "Permission denied: mlflow artifacts requires a configured project directory"

        if artifacts_subcommand == "download":
            # A "file://" artifact URI is a raw absolute filesystem path, independent of
            # any run/experiment tracking and of --dst-path; runs:/, models:/, s3://, etc.
            # go through real tracked storage and are unaffected.
            artifact_uri = _extract_option_value(argv, {"-u", "--artifact-uri"})
            if artifact_uri and artifact_uri.lower().startswith("file://"):
                return "Permission denied: mlflow artifacts download does not allow file:// URIs"
            path_value, flag = _extract_option_value(argv, {"-d", "--dst-path"}), "--dst-path"
        else:
            path_value = _extract_option_value(argv, {"-l", "--local-file", "--local-dir"})
            flag = "--local-file/--local-dir"

        if path_value:
            try:
                target = _resolve_file_path(path_value, cwd)
            except (ValueError, OSError):
                return f"Permission denied: malformed path {path_value!r}"
            if not _is_path_within(target, cwd):
                return f"Permission denied: {flag} {path_value} is outside the workspace {cwd}"

    return None


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

    if tool_name == "Bash":
        command = tool_input.get("command", "").strip()
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

        # Most mlflow CLI commands (--version, experiments search, runs list, ...) don't
        # carry an arbitrary file-access or code-execution surface and stay usable
        # without a configured project directory. A few specific subcommands do and are
        # checked individually rather than gated by cwd alone.
        if argv[0] == "mlflow" and (denial := _mlflow_subcommand_denial(argv, cwd)) is not None:
            return denial

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
            except (ValueError, OSError):
                # e.g. an embedded NUL byte: Path.resolve() raises rather than returning a
                # path, so this must be caught here rather than left to propagate out of
                # execute_tool, which only wraps the tool dispatch (below) in try/except,
                # not this static check.
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
    command = tool_input.get("command", "").strip()
    if not command:
        return "No command provided", True

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
