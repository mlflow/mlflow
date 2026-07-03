import asyncio
import logging
import os
import shlex
from pathlib import Path
from typing import Any

from mlflow.assistant.config import PermissionsConfig

_logger = logging.getLogger(__name__)

_FILE_TOOLS = {"Read", "Write", "Edit"}
# Restricted mode only permits the MLflow CLI; anything else needs Full Access.
# python/python3 were removed: allowing them is plain remote code execution as the
# server process.
_ALLOWED_BASH_COMMANDS = {"mlflow"}

# MLflow subcommands the restricted assistant may invoke. This is an allowlist
# (fail-closed): any subcommand not listed here — including future ones — is
# denied. Every entry routes through the tracking API, so Tier-1 token forwarding
# authorizes it as the calling user; most are read/query verbs, and the few that
# mutate (`experiments create/delete/restore/rename/update`, `runs delete/restore`)
# are still authorized as that user. Excluded subcommands such as `run`, `models`,
# `server`, `deployments`, `sagemaker`, `gateway`, `db`, `gc`, and the LLM/agent
# runners either execute arbitrary code, serve/mutate backend state, or permanently
# delete data. `artifacts` is also excluded: `log-artifact --local-file` /
# `download --dst-path` take unconstrained server-local paths, making it an arbitrary
# file read/write primitive that would escape the workspace sandbox the Read/Write
# tools enforce. `doctor` is excluded too: it reads local process state rather than
# the tracking API and prints every MLFLOW_* env var unmasked (including
# MLFLOW_TRACKING_TOKEN/USERNAME/PASSWORD) plus the raw tracking URI, so under remote
# lockdown it is a one-shot server-side secret exfiltration primitive.
_ALLOWED_MLFLOW_SUBCOMMANDS = frozenset({
    "experiments",
    "runs",
    "traces",
    "datasets",
    "scorers",
})

# Sub-subcommands that are denied even though their parent subcommand is allowlisted,
# because they are file-system primitives (not tracking-API queries) that escape the
# workspace sandbox. `experiments csv --filename/-o PATH` writes an arbitrary
# server-local file via pandas.to_csv with no path validation (mlflow/experiments.py),
# and needs no shell to do it. Keyed by parent subcommand.
_DENIED_MLFLOW_SUBSUBCOMMANDS = {
    "experiments": frozenset({"csv"}),
}

# The only value-taking option on the top-level `mlflow` group (mlflow/cli/__init__.py).
# It consumes the following token, which must not be mistaken for the subcommand.
_MLFLOW_GLOBAL_VALUE_FLAGS = frozenset({"--env-file"})


def _mlflow_subcommand(argv: list[str]) -> tuple[str | None, str | None]:
    """Return ``(subcommand, sub_subcommand)`` for a ``mlflow`` invocation, using
    None for either when absent (e.g. ``mlflow --version`` -> (None, None),
    ``mlflow experiments`` -> ("experiments", None)). ``argv[0]`` is assumed to be
    ``mlflow``.

    Skips global flags and the value consumed by ``--env-file`` so the positional
    tokens are identified without a full CLI parser. The sub-subcommand is the second
    positional; per-subcommand option values could in principle precede it, but the
    denied sub-subcommands are all bare verbs, so returning the first positional after
    the subcommand is sufficient (and any mismatch fails closed at the allowlist).
    """
    positionals: list[str] = []
    i = 1
    while i < len(argv) and len(positionals) < 2:
        tok = argv[i]
        if tok in _MLFLOW_GLOBAL_VALUE_FLAGS:
            i += 2  # skip the flag and its value
            continue
        if tok.startswith("-"):
            i += 1  # boolean flag such as --version / --help
            continue
        positionals.append(tok)
        i += 1
    subcommand = positionals[0] if positionals else None
    sub_subcommand = positionals[1] if len(positionals) > 1 else None
    return subcommand, sub_subcommand


def remote_lockdown_active() -> bool:
    """True when the assistant is exposed beyond localhost (MLFLOW_ALLOW_REMOTE_ASSISTANT).

    In this mode the restricted allowlist is absolute: full_access is force-disabled
    and per-call approval cannot override it, since arbitrary shell/code execution on a
    remotely-reachable server is RCE affecting every tenant.
    """
    return bool(os.environ.get("MLFLOW_ALLOW_REMOTE_ASSISTANT"))


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
    # Remote mode must never grant full_access, even if the config requests it:
    # a remotely-reachable assistant with unrestricted shell access is RCE.
    if perms.full_access and not remote_lockdown_active():
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
        if argv[0] == "mlflow":
            subcommand, sub_subcommand = _mlflow_subcommand(argv)
            if subcommand is not None and subcommand not in _ALLOWED_MLFLOW_SUBCOMMANDS:
                return (
                    f"Permission denied: 'mlflow {subcommand}' is not allowed. "
                    f"Allowed subcommands: {', '.join(sorted(_ALLOWED_MLFLOW_SUBCOMMANDS))}"
                )
            denied_actions = _DENIED_MLFLOW_SUBSUBCOMMANDS.get(subcommand, frozenset())
            if sub_subcommand in denied_actions:
                return (
                    f"Permission denied: 'mlflow {subcommand} {sub_subcommand}' is not allowed "
                    "(it can write to an arbitrary server-local path)."
                )

    if tool_name in _FILE_TOOLS and not perms.allow_edit_files:
        return f"Permission denied: {tool_name} is not allowed"

    if tool_name in {"Write", "Edit"} and not cwd:
        return f"Permission denied: {tool_name} requires a configured project directory"

    if tool_name in _FILE_TOOLS and cwd:
        if raw_path := tool_input.get("file_path") or tool_input.get("path", ""):
            target = _resolve_file_path(raw_path, cwd)
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

    # Only genuine (local) full_access gets a real shell. In restricted mode — or
    # any remote deployment — the command runs as a direct argv exec so the
    # subcommand allowlist is the true boundary: shell metacharacters (|, ;, &&,
    # $(), >, backticks, newlines) can't smuggle a second process past the check.
    use_shell = perms.full_access and not remote_lockdown_active()

    try:
        match tool_name:
            case "Bash":
                return await _execute_bash(
                    tool_input, cwd=cwd, tracking_uri=tracking_uri, use_shell=use_shell
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
    use_shell: bool = False,
) -> tuple[str, bool]:
    command = tool_input.get("command", "")
    if not command:
        return "No command provided", True

    env = os.environ.copy()
    if tracking_uri:
        env["MLFLOW_TRACKING_URI"] = tracking_uri

    try:
        if use_shell:
            # full_access only: a real shell so pipes, redirects, and && chaining work.
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
        else:
            # Restricted/remote: exec the parsed argv directly (no shell), so shell
            # metacharacters can't spawn a second process past the allowlist check.
            try:
                argv = shlex.split(command)
            except ValueError:
                return "Permission denied: malformed command", True
            if not argv:
                return "No command provided", True
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
    except FileNotFoundError:
        return f"Command not found: {command}", True


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
                    "Run a command to query or interact with MLflow. In the default "
                    "(restricted) mode only the 'mlflow' CLI is permitted and the command "
                    "is executed directly without a shell, so pipes, redirects, command "
                    "substitution, and chaining (|, >, ;, &&, $(...)) are not interpreted. "
                    "With Full Access enabled, arbitrary shell commands are allowed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "The command to execute. In restricted mode this must be an "
                                "'mlflow' CLI invocation (e.g. 'mlflow experiments search')."
                            ),
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
    ]
