import asyncio
import contextvars
import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from mlflow.assistant.config import PermissionsConfig
from mlflow.assistant.custom_view import RENDER_CUSTOM_VIEW_TOOL_NAME
from mlflow.assistant.providers.base import assistant_sandbox_enabled

_logger = logging.getLogger(__name__)

# Whether the current request comes from a restricted caller: a non-localhost (remote) caller, or,
# on a server with auth and the sandbox on, a caller who is not an admin (see
# ``_is_restricted_caller`` in the Assistant API). Set per request by the Assistant route layer.
# Restricted callers are capped at the restricted permission profile (no full_access): their tool
# calls run in the sandbox, and this cap also stops their stored config or an interactive approval
# from unlocking full_access there. Other callers keep their configured permissions. Defaults to
# False so non-request contexts (e.g. the local CLI) are unrestricted.
_remote_caller: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "mlflow_assistant_remote_caller", default=False
)


def set_remote_caller(remote: bool) -> None:
    """Bind whether the current request is from a restricted caller.

    A restricted caller is a remote (non-localhost) caller, or a local non-admin on a server with
    auth and the sandbox on (see ``_is_restricted_caller`` in the Assistant API). This is not a
    network-locality flag: use ``_is_localhost`` in the API for that.
    """
    _remote_caller.set(remote)


def is_remote_caller() -> bool:
    """Whether the current request is from a restricted caller (see ``set_remote_caller``)."""
    return _remote_caller.get()


def restrict_permissions_for_remote(perms: PermissionsConfig) -> PermissionsConfig:
    """Cap ``perms`` at the restricted profile for a restricted caller (see
    ``set_remote_caller``); a no-op for any other caller.

    ``full_access`` is the arbitrary-code / out-of-workspace escape hatch, so it is the field a
    restricted caller must never obtain; the workspace-confined file and CLI allowances are
    unchanged.
    """
    if not is_remote_caller() or not perms.full_access:
        return perms
    return perms.model_copy(update={"full_access": False})


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
# In the sandbox, restricted commands run through a shell, so they may be combined with pipes,
# ``&&``/``||``/``;`` and redirects. Every command in the chain must then be allowed: the commands
# above, or one of these text tools. None of them can start another program or write a file
# (unlike sed, awk, find, xargs, GNU sort's ``--compress-program`` or uniq's OUTPUT argument, which
# are left out).
_SANDBOX_TEXT_COMMANDS = {"cat", "cut", "echo", "grep", "head", "tail", "tr", "wc"}
_SHELL_COMMAND_SEPARATORS = {"|", "||", "&&", ";"}
# Redirects as /bin/sh (dash) parses them. bash's ``&>`` is not one: dash reads ``cmd &> f next`` as
# ``cmd &`` and then runs ``next`` as a separate command.
_SHELL_REDIRECTS = {"<", ">", ">>", ">|", "<>", ">&", "<&"}
# Redirects that open their target for writing. ``>&``/``<&`` only duplicate a file descriptor.
_SHELL_WRITE_REDIRECTS = {">", ">>", ">|", "<>"}
_FILE_DESCRIPTOR = re.compile(r"\d+")
# Shell syntax that runs a command the checks below would never see: command and process
# substitution, ``${...}`` expansions (which can assign variables such as PATH), and newlines
# (which separate commands like ``;``).
_UNCHECKED_SHELL_SYNTAX = re.compile(r"`|\$\(|\$\{|[<>]\(|\n")

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
        if assistant_sandbox_enabled():
            return _sandbox_shell_permission_error(command, perms, cwd)
        try:
            argv = shlex.split(command)
        except ValueError:
            return "Permission denied: malformed command"
        if not argv or argv[0] not in _ALLOWED_BASH_COMMANDS:
            return (
                f"Permission denied: only {', '.join(sorted(_ALLOWED_BASH_COMMANDS))} "
                "commands are allowed"
            )
        if error := _command_permission_error(argv[0], cwd):
            return error

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


def _command_permission_error(command_name: str, cwd: Path | None) -> str | None:
    # python/python3 can run arbitrary code, including reading any file the process can access, so
    # require the same configured project directory Read/Write/Edit do. Without this,
    # GHSA-27c7-qx3r-x4f8's impact (arbitrary file read when cwd is None) is reachable via
    # Bash("python3 -c \"print(open(path).read())\"") even though Read itself is denied.
    if command_name in {"python", "python3"} and cwd is None:
        return f"Permission denied: {command_name} requires a configured project directory"
    return None


def _sandbox_shell_permission_error(
    command: str, perms: PermissionsConfig, cwd: Path | None
) -> str | None:
    """Check a restricted command that runs through a shell in the sandbox.

    Splits the command into the commands chained by pipes and ``&&``/``||``/``;`` and checks each
    one, so a shell cannot run anything the restricted allowlist would refuse. Rejects shell syntax
    that would run commands these checks never see, and output redirects when file edits are not
    allowed.
    """
    allowed = sorted(_ALLOWED_BASH_COMMANDS | _SANDBOX_TEXT_COMMANDS)
    if _UNCHECKED_SHELL_SYNTAX.search(command):
        return "Permission denied: command substitution and multi-line commands are not allowed"
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    # shlex would treat a '#' inside a word as the start of a comment and drop the rest of the
    # line, while the shell does not, so it would miss any command after it.
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        return "Permission denied: malformed command"

    def is_operator(token: str) -> bool:
        return set(token) <= set("();<>|&")

    # The words of each chained command, without its redirects (operator, target and any file
    # descriptor number before the operator), so a redirect written before the command name, e.g.
    # ``2>/dev/null mlflow ...``, does not hide the name.
    segments: list[list[str]] = [[]]
    tokens_iter = iter(tokens)
    for token in tokens_iter:
        if token in _SHELL_COMMAND_SEPARATORS:
            segments.append([])
        elif is_operator(token):
            if token not in _SHELL_REDIRECTS:
                return (
                    "Permission denied: subshells, background commands and other shell syntax "
                    "are not allowed"
                )
            target = next(tokens_iter, None)
            if target is None or is_operator(target):
                return "Permission denied: malformed command"
            if token in {">&", "<&"} and target != "-" and not _FILE_DESCRIPTOR.fullmatch(target):
                return "Permission denied: malformed command"
            if (
                token in _SHELL_WRITE_REDIRECTS
                and target != "/dev/null"
                and not perms.allow_edit_files
            ):
                return "Permission denied: writing files is not allowed"
            # shlex splits ``2>`` into ``2`` and ``>``; the number is the descriptor, not a word.
            if segments[-1] and _FILE_DESCRIPTOR.fullmatch(segments[-1][-1]):
                segments[-1].pop()
        else:
            segments[-1].append(token)

    for words in segments:
        if not words:
            return "Permission denied: malformed command"
        name = words[0]
        if name not in _ALLOWED_BASH_COMMANDS and name not in _SANDBOX_TEXT_COMMANDS:
            return f"Permission denied: only {', '.join(allowed)} commands are allowed"
        if error := _command_permission_error(name, cwd):
            return error
    return None


async def execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    cwd: Path | None = None,
    tracking_uri: str | None = None,
    permissions: PermissionsConfig | None = None,
) -> tuple[str, bool]:
    # Cap a remote caller at the restricted profile here as the final enforcement point, so no
    # caller-supplied permissions (config-derived or an interactive full-access grant) can hand a
    # remote request full_access, independent of what the provider passed in.
    perms = restrict_permissions_for_remote(permissions or PermissionsConfig())

    if (denial := static_permission_error(tool_name, tool_input, perms, cwd)) is not None:
        return denial, True

    try:
        match tool_name:
            case "Bash":
                return await _execute_bash(
                    tool_input, cwd=cwd, tracking_uri=tracking_uri, full_access=perms.full_access
                )
            case "Read" | "Write" | "Edit":
                # When the sandbox is enabled the file tools must run inside the container like
                # Bash, not on the host, so the container filesystem namespace bounds them.
                if assistant_sandbox_enabled():
                    return await _execute_file_tool_in_sandbox(tool_name, tool_input, cwd)
                return await asyncio.to_thread(_HOST_FILE_TOOLS[tool_name], tool_input, cwd=cwd)
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
        return await _execute_bash_in_sandbox(command, cwd, tracking_uri)
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
            run_args = command
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
                run_args = shlex.split(command)
            except ValueError:
                return "Permission denied: malformed command", True

        proc = await asyncio.to_thread(
            subprocess.run,
            run_args,
            shell=full_access,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            timeout=120,
        )
        output = proc.stdout.decode("utf-8", errors="replace")
        err_output = proc.stderr.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            result = (
                output + err_output if output or err_output else f"Exit code: {proc.returncode}"
            )
            return result.strip(), True

        return (output + err_output).strip() or "(no output)", False
    except subprocess.TimeoutExpired:
        return "Command timed out after 120 seconds", True


async def _execute_bash_in_sandbox(
    command: str,
    cwd: Path | None,
    tracking_uri: str | None,
) -> tuple[str, bool]:
    """Run the command inside a hardened Docker container instead of on the host.

    The command always runs through a shell, so pipes, redirects and ``&&`` work in restricted
    mode too. That is safe because, with the sandbox on, the static permission check
    (``_sandbox_shell_permission_error``) checks every command in the chain, not just the first
    one. On the host, restricted mode still runs the argv with no shell.
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

    try:
        # run_in_sandbox uses the blocking docker-py client, so run it off the event loop.
        result = await asyncio.to_thread(
            run_in_sandbox,
            [command],
            workdir=cwd,
            environment=env,
            timeout=120,
            use_shell=True,
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


_HOST_FILE_TOOLS = {"Read": _execute_read, "Write": _execute_write, "Edit": _execute_edit}

# Each file op runs as a single Python program inside the sandbox container (the image is
# Python-based, like Bash relies on a shell there). Inputs are passed via the environment, never
# the command line, so arbitrary file content is not shell-quoted or exposed in the process args.
# Paths are relative to the workspace mount (cwd), so they resolve to the same files the host tools
# would use. Edit exits 3 when old_string is absent, matching the host tool's error.
_SANDBOX_FILE_TOOL_PROGRAMS = {
    "Read": 'import os,sys\nsys.stdout.write(open(os.environ["MLF_FILE"],encoding="utf-8").read())',
    "Write": (
        "import os\n"
        'p=os.environ["MLF_FILE"]\n'
        "d=os.path.dirname(p)\n"
        "if d:\n"
        "    os.makedirs(d,exist_ok=True)\n"
        'open(p,"w",encoding="utf-8").write(os.environ["MLF_CONTENT"])'
    ),
    "Edit": (
        "import os,sys\n"
        'p=os.environ["MLF_FILE"]\n'
        'old=os.environ["MLF_OLD"]\n'
        'c=open(p,encoding="utf-8").read()\n'
        "if old not in c:\n"
        "    sys.exit(3)\n"
        'open(p,"w",encoding="utf-8").write(c.replace(old,os.environ["MLF_NEW"],1))'
    ),
}


async def _execute_file_tool_in_sandbox(
    tool_name: str, tool_input: dict[str, Any], cwd: Path | None
) -> tuple[str, bool]:
    """Run Read/Write/Edit inside the sandbox container instead of on the host.

    File tools are already path-confined to cwd on the host, but when the operator enables the
    sandbox they must run inside the container like Bash so the container filesystem namespace --
    not only the host-side path check -- bounds them (closing symlink/TOCTOU escapes and matching
    the isolation the operator enabled). cwd is bind-mounted read-write as the container workdir,
    so a path relative to it resolves to the same file.
    """
    from mlflow.server.sandbox import SandboxUnavailableError, run_in_sandbox

    if cwd is None:
        return f"Permission denied: {tool_name} requires a configured project directory", True
    raw_path = tool_input.get("file_path") or tool_input.get("path", "")
    if not raw_path:
        return "No file_path provided", True
    try:
        rel = _resolve_file_path(raw_path, cwd).relative_to(cwd.resolve())
    except (ValueError, OSError, TypeError):
        return f"Permission denied: malformed path {raw_path!r}", True

    env = {"MLF_FILE": str(rel)}
    if tool_name == "Write":
        env["MLF_CONTENT"] = tool_input.get("content", "")
    elif tool_name == "Edit":
        env["MLF_OLD"] = tool_input.get("old_string", "")
        env["MLF_NEW"] = tool_input.get("new_string", "")

    try:
        result = await asyncio.to_thread(
            run_in_sandbox,
            ["python", "-c", _SANDBOX_FILE_TOOL_PROGRAMS[tool_name]],
            workdir=cwd,
            environment=env,
            timeout=120,
        )
    except SandboxUnavailableError as e:
        return f"Sandbox is enabled but {tool_name} could not be run: {e}", True

    if result.timed_out:
        return f"{tool_name} timed out after 120 seconds", True
    if tool_name == "Edit" and result.exit_code == 3:
        return f"old_string not found in {raw_path}", True
    if result.exit_code != 0:
        return result.output.strip() or f"Exit code: {result.exit_code}", True

    if tool_name == "Read":
        return result.output, False
    if tool_name == "Write":
        return f"Wrote {len(env['MLF_CONTENT'])} bytes to {raw_path}", False
    return f"Edited {raw_path}", False


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
