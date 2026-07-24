import asyncio

import pytest

from mlflow.assistant.config import PermissionsConfig
from mlflow.assistant.providers.tool_executor import (
    _mlflow_subcommand,
    execute_tool,
    static_permission_error,
)

# MLFLOW_ALLOW_REMOTE_ASSISTANT is cleared by the autouse fixture in conftest.py.


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
        (["mlflow"], (None, None)),
        (["mlflow", "--version"], (None, None)),
        (["mlflow", "experiments"], ("experiments", None)),
        (["mlflow", "experiments", "search"], ("experiments", "search")),
        (["mlflow", "experiments", "csv", "--filename", "/tmp/x"], ("experiments", "csv")),
        # --env-file consumes its value; a value that looks like a subcommand is not one.
        (["mlflow", "--env-file", "run", "experiments", "search"], ("experiments", "search")),
        # Glued --env-file=value must skip exactly one slot.
        (["mlflow", "--env-file=x", "run"], ("run", None)),
        # `ui` is an AliasedGroup alias for `server`; not in the allowlist -> caught.
        (["mlflow", "ui"], ("ui", None)),
        (["mlflow", "--env-file", "/tmp/e", "run", "."], ("run", ".")),
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


def test_remote_env_disables_full_access_bash(monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_REMOTE_ASSISTANT", "1")
    perms = PermissionsConfig(full_access=True)
    result, is_error = _run(execute_tool("Bash", {"command": "echo hello"}, permissions=perms))
    assert is_error
    assert "Permission denied" in result


def test_remote_env_disables_full_access_file_escape(monkeypatch, workspace):
    monkeypatch.setenv("MLFLOW_ALLOW_REMOTE_ASSISTANT", "1")
    perms = PermissionsConfig(full_access=True)
    result, is_error = _run(
        execute_tool("Read", {"file_path": "../../../etc/passwd"}, cwd=workspace, permissions=perms)
    )
    assert is_error
    assert "Permission denied" in result


def test_bash_full_access_allows_any_command():
    perms = PermissionsConfig(full_access=True)
    result, is_error = _run(execute_tool("Bash", {"command": "echo hello"}, permissions=perms))
    assert not is_error
    assert "hello" in result


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
