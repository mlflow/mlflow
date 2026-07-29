import asyncio

import pytest

from mlflow.assistant.config import PermissionsConfig
from mlflow.assistant.providers.tool_executor import (
    _execute_bash,
    _mlflow_subcommand,
    execute_tool,
    remote_lockdown_active,
    static_permission_error,
)
from mlflow.environment_variables import MLFLOW_ENABLE_REMOTE_ASSISTANT

# MLFLOW_ENABLE_REMOTE_ASSISTANT is cleared by the autouse fixture in conftest.py.


def test_remote_lockdown_reads_canonical_env_var(monkeypatch):
    # The lockdown must key off the same switch that actually opens the assistant to
    # remote clients (MLFLOW_ENABLE_REMOTE_ASSISTANT, mlflow/server/assistant/api.py),
    # not a differently-named variable that nothing sets — otherwise it is fail-open.
    assert remote_lockdown_active() is False
    monkeypatch.setenv(MLFLOW_ENABLE_REMOTE_ASSISTANT.name, "1")
    assert remote_lockdown_active() is True


@pytest.fixture
def workspace(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("print('hello')")
    (tmp_path / "README.md").write_text("# project")
    return tmp_path


def _run(coro):
    return asyncio.run(coro)


def test_read_resolves_relative_path_against_cwd(workspace):
    result, is_error = _run(execute_tool("Read", {"file_path": "src/main.py"}, cwd=workspace))
    assert not is_error
    assert "print('hello')" in result


def test_read_absolute_path_works_without_cwd(workspace):
    result, is_error = _run(execute_tool("Read", {"file_path": str(workspace / "README.md")}))
    assert not is_error
    assert "# project" in result


def test_write_denied_without_cwd():
    result, is_error = _run(execute_tool("Write", {"file_path": "test.txt", "content": "hi"}))
    assert is_error
    assert "Permission denied" in result


def test_write_resolves_relative_path(workspace):
    result, is_error = _run(
        execute_tool("Write", {"file_path": "output.txt", "content": "data"}, cwd=workspace)
    )
    assert not is_error
    assert (workspace / "output.txt").read_text() == "data"


def test_edit_resolves_relative_path(workspace):
    result, is_error = _run(
        execute_tool(
            "Edit",
            {"file_path": "README.md", "old_string": "# project", "new_string": "# updated"},
            cwd=workspace,
        )
    )
    assert not is_error
    assert (workspace / "README.md").read_text() == "# updated"


def test_path_containment_blocks_escape(workspace):
    result, is_error = _run(
        execute_tool("Read", {"file_path": "../../../etc/passwd"}, cwd=workspace)
    )
    assert is_error
    assert "Permission denied" in result


def test_bash_works_without_cwd():
    # Previously ran python3 (now blocked); the MLflow CLI is the only allowed binary.
    result, is_error = _run(execute_tool("Bash", {"command": "mlflow --version"}))
    assert not is_error
    assert "mlflow" in result.lower()


def test_bash_blocks_non_mlflow_commands():
    result, is_error = _run(execute_tool("Bash", {"command": "echo hello"}))
    assert is_error
    assert "Permission denied" in result


@pytest.mark.parametrize("cmd", ["python3 -c \"print('x')\"", "python foo.py"])
def test_bash_blocks_python(cmd):
    result, is_error = _run(execute_tool("Bash", {"command": cmd}))
    assert is_error
    assert "Permission denied" in result


def test_bash_allows_mlflow_commands():
    result, is_error = _run(execute_tool("Bash", {"command": "mlflow --version"}))
    assert not is_error


@pytest.mark.parametrize(
    "template",
    [
        "mlflow experiments search; touch {sentinel}",
        "mlflow experiments search && touch {sentinel}",
        "mlflow experiments search | touch {sentinel}",
        "mlflow --version || touch {sentinel}",
        "mlflow experiments search $(touch {sentinel})",
        "mlflow experiments search > {sentinel}",
        "mlflow experiments search `touch {sentinel}`",
        "mlflow experiments search\ntouch {sentinel}",
    ],
)
def test_bash_no_shell_injection_via_metacharacters(tmp_path, template):
    # An allowlisted subcommand (passes the static check) followed by a shell
    # metacharacter must NOT spawn a second process. Restricted mode runs argv
    # directly (no /bin/sh -c), so the metacharacters reach mlflow as literal
    # args instead of being interpreted by a shell.
    sentinel = tmp_path / "pwned"
    _run(execute_tool("Bash", {"command": template.format(sentinel=sentinel)}))
    assert not sentinel.exists(), f"shell injection executed: {template!r}"


@pytest.mark.parametrize(
    "sub",
    ["run", "server", "models", "deployments", "sagemaker", "gateway", "db", "gc", "ai", "doctor"],
)
def test_bash_blocks_dangerous_mlflow_subcommands(sub):
    result, is_error = _run(execute_tool("Bash", {"command": f"mlflow {sub} --help"}))
    assert is_error
    assert "Permission denied" in result


@pytest.mark.parametrize(
    "cmd",
    [
        "mlflow experiments search",
        "mlflow experiments get --experiment-id 0",
        "mlflow experiments create --experiment-name x",
        "mlflow runs list --experiment-id 0",
        "mlflow traces get --trace-id t",
        "mlflow --version",
        "mlflow --help",
    ],
)
def test_static_allows_safe_mlflow_commands(cmd):
    assert static_permission_error("Bash", {"command": cmd}, PermissionsConfig(), None) is None


@pytest.mark.parametrize(
    "cmd",
    [
        "mlflow run .",
        "mlflow models serve -m model",
        "mlflow gc",
        "mlflow db upgrade sqlite:///x",
        # `artifacts` is an arbitrary server-local file read/write primitive; denied.
        "mlflow artifacts log-artifact --local-file /etc/passwd --run-id r",
        "mlflow artifacts download --dst-path /tmp/x",
        # `experiments csv --filename/-o PATH` writes an arbitrary server-local file
        # (to_csv, no path validation) inside an otherwise-allowlisted subcommand.
        "mlflow experiments csv --experiment-id 0 --filename /root/.ssh/authorized_keys",
        "mlflow experiments csv --experiment-id 0 -o /etc/cron.d/x",
        "mlflow experiments csv --experiment-id 0",
        # `doctor` prints MLFLOW_* env vars (tokens/passwords) unmasked and the raw
        # tracking URI, reading local process state rather than the tracking API.
        "mlflow doctor",
        "mlflow doctor --mask-envs",
    ],
)
def test_static_denies_dangerous_mlflow_commands(cmd):
    assert static_permission_error("Bash", {"command": cmd}, PermissionsConfig(), None) is not None


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["mlflow"], None),
        (["mlflow", "--version"], None),
        (["mlflow", "experiments"], "experiments"),
        (["mlflow", "experiments", "search"], "experiments"),
        (["mlflow", "experiments", "csv", "--filename", "/tmp/x"], "experiments"),
        # --env-file consumes its value; a value that looks like a subcommand is not one.
        (["mlflow", "--env-file", "run", "experiments", "search"], "experiments"),
        # Glued --env-file=value must skip exactly one slot.
        (["mlflow", "--env-file=x", "run"], "run"),
        # `ui` is an AliasedGroup alias for `server`; not in the allowlist -> caught.
        (["mlflow", "ui"], "ui"),
        (["mlflow", "--env-file", "/tmp/e", "run", "."], "run"),
    ],
)
def test_mlflow_subcommand_parsing(argv, expected):
    assert _mlflow_subcommand(argv) == expected


def test_env_file_flag_value_is_not_mistaken_for_subcommand():
    # `--env-file` consumes the next token; the real subcommand follows and must
    # still be allowed. A value that happens to look like a subcommand must not fool the check.
    err = static_permission_error(
        "Bash", {"command": "mlflow --env-file run experiments search"}, PermissionsConfig(), None
    )
    assert err is None


def test_dangerous_subcommand_after_env_file_flag_is_blocked():
    err = static_permission_error(
        "Bash", {"command": "mlflow --env-file /tmp/e run ."}, PermissionsConfig(), None
    )
    assert err is not None


@pytest.mark.parametrize(
    "cmd",
    [
        # Canonical form (already denied), kept as a regression guard.
        "mlflow experiments csv --experiment-id 0 -o /tmp/x",
        # Bypass: `-x` takes a value, so a positional-index denylist reads the
        # sub-subcommand as "0" and lets `csv` through. A whole-argv verb scan
        # must still deny it regardless of where `csv` appears.
        "mlflow experiments -x 0 csv -o /tmp/x",
        # Glued short option value (`-x0`) is the subtlest shlex form.
        "mlflow experiments -x0 csv -o /tmp/x",
        "mlflow experiments --experiment-id 0 csv --filename /tmp/x",
        "mlflow experiments --experiment-id=0 csv -o /tmp/x",
    ],
)
def test_experiments_csv_denied_regardless_of_arg_order(cmd):
    # `experiments csv` writes an arbitrary server-local file (to_csv, no path
    # validation). The denial must not be evadable by moving a value-taking
    # option in front of the `csv` verb.
    assert static_permission_error("Bash", {"command": cmd}, PermissionsConfig(), None) is not None


def test_remote_env_disables_full_access_bash(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_REMOTE_ASSISTANT.name, "1")
    perms = PermissionsConfig(full_access=True)
    result, is_error = _run(execute_tool("Bash", {"command": "echo hello"}, permissions=perms))
    assert is_error
    assert "Permission denied" in result


def test_remote_env_disables_full_access_file_escape(monkeypatch, workspace):
    monkeypatch.setenv(MLFLOW_ENABLE_REMOTE_ASSISTANT.name, "1")
    perms = PermissionsConfig(full_access=True)
    result, is_error = _run(
        execute_tool("Read", {"file_path": "../../../etc/passwd"}, cwd=workspace, permissions=perms)
    )
    assert is_error
    assert "Permission denied" in result


@pytest.mark.parametrize("tool", ["Read", "Write", "Edit"])
def test_file_tool_without_cwd_denied_under_remote_lockdown(monkeypatch, tool):
    # Under remote lockdown, no workspace root means no containment boundary, so an
    # absolute path like /etc/passwd would otherwise reach the filesystem unchecked.
    # Every file tool must require a configured project directory here, not just
    # Write/Edit. (Local unscoped reads remain allowed — see
    # test_read_absolute_path_works_without_cwd.)
    monkeypatch.setenv(MLFLOW_ENABLE_REMOTE_ASSISTANT.name, "1")
    err = static_permission_error(tool, {"file_path": "/etc/passwd"}, PermissionsConfig(), None)
    assert err is not None


def test_remote_read_without_cwd_cannot_escape(monkeypatch):
    # Regression: remote lockdown + full_access config, no cwd. Read must be denied
    # rather than falling through to read an arbitrary server-local file.
    monkeypatch.setenv(MLFLOW_ENABLE_REMOTE_ASSISTANT.name, "1")
    perms = PermissionsConfig(full_access=True)
    result, is_error = _run(execute_tool("Read", {"file_path": "/etc/passwd"}, permissions=perms))
    assert is_error
    assert "Permission denied" in result


@pytest.mark.parametrize("tool", ["Read", "Write", "Edit"])
def test_nul_byte_path_denied_not_crashed(workspace, tool):
    # A NUL byte in file_path makes Path.resolve() raise ValueError. That call lives
    # in static_permission_error (outside execute_tool's try/except), so an unguarded
    # resolve would propagate instead of denying. It must fail closed with a denial.
    err = static_permission_error(
        tool, {"file_path": "a\x00b"}, PermissionsConfig(allow_edit_files=True), workspace
    )
    assert err is not None
    assert "Permission denied" in err


def test_bash_full_access_allows_any_command():
    perms = PermissionsConfig(full_access=True)
    result, is_error = _run(execute_tool("Bash", {"command": "echo hello"}, permissions=perms))
    assert not is_error
    assert "hello" in result


def test_command_not_found_echoes_only_executable(tmp_path):
    # The FileNotFoundError message must not leak user-controlled arguments or
    # server-local paths — only the executable name.
    secret_arg = str(tmp_path / "secret-internal-path")
    result, is_error = _run(
        _execute_bash(
            {"command": f"definitely-not-a-real-binary --token {secret_arg}"},
            cwd=None,
            tracking_uri=None,
        )
    )
    assert is_error
    assert result == "Command not found: definitely-not-a-real-binary"
    assert secret_arg not in result


def test_bash_full_access_still_uses_shell(tmp_path):
    # Local full_access is the explicit "give me a real shell" opt-in: pipes,
    # redirects, and chaining must still work there.
    sentinel = tmp_path / "ok"
    perms = PermissionsConfig(full_access=True)
    result, is_error = _run(
        execute_tool("Bash", {"command": f"echo hi && touch {sentinel}"}, permissions=perms)
    )
    assert not is_error
    assert sentinel.exists()


def test_full_access_bypasses_permission_checks(workspace):
    perms = PermissionsConfig(full_access=True)
    result, is_error = _run(
        execute_tool(
            "Read",
            {"file_path": "../../../etc/hosts"},
            cwd=workspace,
            permissions=perms,
        )
    )
    # Should not get "Permission denied" (may get file not found depending on OS)
    assert "Permission denied" not in result
