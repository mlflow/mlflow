import asyncio
from unittest import mock
from unittest.mock import AsyncMock

import pytest

from mlflow.assistant.config import PermissionsConfig
from mlflow.assistant.providers.base import assistant_sandbox_enabled
from mlflow.assistant.providers.tool_executor import _execute_bash_in_sandbox, execute_tool
from mlflow.server.sandbox import SandboxResult, SandboxUnavailableError


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


def test_read_absolute_path_denied_without_cwd(workspace):
    # Regression guard for GHSA-27c7-qx3r-x4f8: without a configured project
    # directory (cwd=None, e.g. no experiment_id), Read must be denied rather
    # than allowed to read an arbitrary absolute path on the filesystem.
    result, is_error = _run(execute_tool("Read", {"file_path": str(workspace / "README.md")}))
    assert is_error
    assert "Permission denied" in result


def test_read_sensitive_file_denied_without_cwd(tmp_path):
    # Regression guard for GHSA-27c7-qx3r-x4f8: an absolute path to a file
    # completely outside any workspace (e.g. an .env or SSH key) must be
    # denied when no cwd/experiment_id is configured, not read back verbatim.
    secret = tmp_path / "secret.env"
    secret.write_text("SECRET_API_KEY=sk-super-secret-12345")
    result, is_error = _run(execute_tool("Read", {"file_path": str(secret)}))
    assert is_error
    assert "Permission denied" in result
    assert "SECRET_API_KEY" not in result


def test_read_relative_path_denied_without_cwd(tmp_path, monkeypatch):
    # Regression guard for GHSA-27c7-qx3r-x4f8: without cwd/experiment_id,
    # Read must be denied even for a relative path, which would otherwise
    # resolve against the server process's own working directory.
    secret = tmp_path / "secret.env"
    secret.write_text("SECRET_API_KEY=sk-super-secret-12345")
    monkeypatch.chdir(tmp_path)
    result, is_error = _run(execute_tool("Read", {"file_path": "secret.env"}))
    assert is_error
    assert "Permission denied" in result
    assert "SECRET_API_KEY" not in result


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


def test_read_malformed_path_denied_not_raised(workspace):
    # Regression guard: an embedded NUL byte makes Path.resolve() raise ValueError.
    # static_permission_error must catch this and return a normal denial instead of
    # letting the exception propagate out of execute_tool, which only wraps the tool
    # dispatch (not the static permission check) in try/except, so an unhandled
    # ValueError here would abort the whole assistant turn instead of denying the call.
    result, is_error = _run(execute_tool("Read", {"file_path": "foo\x00bar"}, cwd=workspace))
    assert is_error
    assert "Permission denied" in result


def test_read_non_string_path_denied_not_raised(workspace):
    # Regression guard: malformed tool-call JSON (e.g. a nonempty list instead of a
    # string) makes Path() raise TypeError, not ValueError/OSError. This must be caught
    # the same way, rather than propagating out of execute_tool and aborting the turn.
    result, is_error = _run(execute_tool("Read", {"file_path": [1]}, cwd=workspace))
    assert is_error
    assert "Permission denied" in result


def test_non_dict_tool_input_denied_not_raised():
    # Regression guard: the model's function-call "arguments" string is parsed with
    # json.loads and passed through as tool_input unchanged. Any syntactically valid
    # JSON that isn't an object (e.g. "[]", "null") reaches static_permission_error as
    # a list/None/etc., and every check there calls tool_input.get(...), which would
    # raise AttributeError instead of returning a denial. Must be caught regardless of
    # tool_name, since Read/Write/Edit/Bash all call tool_input.get(...) the same way.
    for tool_name in ("Bash", "Read", "Write", "Edit"):
        for bad_input in ([], None, "not a dict", 123):
            result, is_error = _run(execute_tool(tool_name, bad_input, cwd=None))
            assert is_error
            assert "Permission denied" in result


def test_bash_python_denied_without_cwd():
    # Regression guard for GHSA-27c7-qx3r-x4f8: without a configured project
    # directory (cwd=None), python/python3 must be denied. Bash previously
    # allowed running python3 with no cwd at all, which reached the same
    # arbitrary-file-read impact as the Read tool bug this advisory reports,
    # since python3 -c "..." can read (or do anything else with) any path the
    # process can access, regardless of what Read itself allows.
    result, is_error = _run(execute_tool("Bash", {"command": "python3 -c \"print('hello')\""}))
    assert is_error
    assert "Permission denied" in result


def test_bash_python_works_with_cwd(workspace):
    result, is_error = _run(
        execute_tool("Bash", {"command": "python3 -c \"print('hello')\""}, cwd=workspace)
    )
    assert not is_error
    assert "hello" in result


def test_bash_python_sensitive_file_denied_without_cwd(tmp_path):
    # Regression guard for GHSA-27c7-qx3r-x4f8: reproduces the exact bypass
    # reported in review, reading a secret file via Bash + python3 instead of
    # Read, with no cwd configured.
    secret = tmp_path / "secret.env"
    secret.write_text("SECRET_API_KEY=sk-super-secret-12345")
    command = f'python3 -c "print(open({str(secret)!r}).read())"'
    result, is_error = _run(execute_tool("Bash", {"command": command}))
    assert is_error
    assert "Permission denied" in result
    assert "SECRET_API_KEY" not in result


def test_bash_shell_chaining_does_not_bypass_allowlist_without_cwd(tmp_path):
    # Regression guard for the review follow-up on GHSA-27c7-qx3r-x4f8: the
    # allowlist in static_permission_error only validates argv[0]. Executing
    # the raw command string via a shell would let a shell operator after
    # argv[0] (&&, ;, |, backticks) smuggle in an unvalidated command, e.g.
    # "mlflow --help && python3 -c '...'" passes the argv[0]=="mlflow" check
    # but a shell would still run the chained python3 read. Execution must
    # run the pre-validated argv directly, with no shell, so anything after
    # argv[0] is a literal argument to mlflow rather than a separate command.
    secret = tmp_path / "secret.env"
    secret.write_text("SECRET_API_KEY=sk-super-secret-12345")
    command = f'mlflow --help && python3 -c "print(open({str(secret)!r}).read())"'
    result, is_error = _run(execute_tool("Bash", {"command": command}))
    assert "SECRET_API_KEY" not in result


def test_bash_shell_chaining_does_not_bypass_allowlist_with_cwd(workspace, tmp_path):
    # Same class of bug as above, but chained after a python3 call that is
    # itself allowed because cwd is configured: the second, chained python3
    # invocation must never run as its own command.
    secret = tmp_path / "secret.env"
    secret.write_text("SECRET_API_KEY=sk-super-secret-12345")
    command = f'python3 -c "print(\'hello\')" && python3 -c "print(open({str(secret)!r}).read())"'
    result, is_error = _run(execute_tool("Bash", {"command": command}, cwd=workspace))
    assert "SECRET_API_KEY" not in result


def test_bash_shell_chaining_with_semicolon_does_not_bypass_allowlist_without_cwd(tmp_path):
    # Same class of bug as test_bash_shell_chaining_does_not_bypass_allowlist_without_cwd,
    # but with ";" instead of "&&" — the review comment's second example
    # ("mlflow --help; cat ~/.ssh/id_rsa") used a different shell operator, and the fix
    # must reject shell metacharacters generically, not just "&&".
    secret = tmp_path / "secret.env"
    secret.write_text("SECRET_API_KEY=sk-super-secret-12345")
    command = f'mlflow --help; python3 -c "print(open({str(secret)!r}).read())"'
    result, is_error = _run(execute_tool("Bash", {"command": command}))
    assert "SECRET_API_KEY" not in result


def test_bash_full_access_still_allows_shell_chaining():
    # Full access has no allowlist to bypass, so it should keep using a real
    # shell and continue to support the && / pipe / redirect chaining the
    # assistant relies on in that mode.
    perms = PermissionsConfig(full_access=True)
    result, is_error = _run(
        execute_tool("Bash", {"command": "echo hello && echo world"}, permissions=perms)
    )
    assert not is_error
    assert "hello" in result
    assert "world" in result


def test_bash_non_string_command_denied_not_raised():
    # Regression guard: malformed tool-call JSON (e.g. a model emitting
    # {"command": 123}) makes "".strip() raise AttributeError on a non-string value.
    # This must be caught the same way non-string file_path is, rather than
    # propagating out of static_permission_error/execute_tool.
    result, is_error = _run(execute_tool("Bash", {"command": 123}))
    assert is_error
    assert "Permission denied" in result


def test_bash_blocks_non_mlflow_commands():
    result, is_error = _run(execute_tool("Bash", {"command": "echo hello"}))
    assert is_error
    assert "Permission denied" in result


def test_bash_allows_mlflow_commands():
    result, is_error = _run(execute_tool("Bash", {"command": "mlflow --version"}))
    assert not is_error


def test_bash_allows_mlflow_help():
    result, is_error = _run(execute_tool("Bash", {"command": "mlflow --help"}))
    assert not is_error


def test_bash_strips_command_consistently_before_tokenizing():
    # Regression guard: static_permission_error stripped the command before shlex.split
    # but _execute_bash did not, so a leading non-ASCII whitespace character (e.g. NBSP,
    # which str.strip() removes but shlex's default whitespace set does not) could pass
    # the allowlist check as "mlflow" while _execute_bash's own unstripped shlex.split
    # saw "\xa0mlflow" instead and failed to find a matching executable. Both must
    # tokenize the command identically.
    result, is_error = _run(execute_tool("Bash", {"command": "\xa0mlflow --version"}))
    assert not is_error


def test_bash_full_access_allows_any_command():
    perms = PermissionsConfig(full_access=True)
    result, is_error = _run(execute_tool("Bash", {"command": "echo hello"}, permissions=perms))
    assert not is_error
    assert "hello" in result


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


@pytest.mark.parametrize(
    ("remote", "docker_path", "expected"),
    [
        (False, "/usr/bin/docker", False),  # local server: never sandbox
        (True, None, False),  # remote but no docker executable: fall back to host
        (True, "/usr/bin/docker", True),  # remote + docker: sandbox
    ],
)
def test_assistant_sandbox_enabled(monkeypatch, remote, docker_path, expected):
    monkeypatch.delenv("MLFLOW_ENABLE_ASSISTANT_SANDBOX", raising=False)
    monkeypatch.setenv("MLFLOW_ENABLE_REMOTE_ASSISTANT", "true" if remote else "false")
    with mock.patch(
        "mlflow.assistant.providers.base.shutil.which", return_value=docker_path
    ) as which:
        assert assistant_sandbox_enabled() is expected
    # docker is only probed once the remote flag has already passed.
    assert which.called is remote


@pytest.mark.parametrize(
    ("override", "remote", "docker_path"),
    [
        ("true", False, None),  # force on even locally and without a docker executable
        ("false", True, "/usr/bin/docker"),  # opt out even in remote + docker
    ],
)
def test_assistant_sandbox_override_beats_derived(monkeypatch, override, remote, docker_path):
    # The explicit flag overrides the derived (remote + docker) default in both directions, and
    # short-circuits the docker probe entirely.
    monkeypatch.setenv("MLFLOW_ENABLE_ASSISTANT_SANDBOX", override)
    monkeypatch.setenv("MLFLOW_ENABLE_REMOTE_ASSISTANT", "true" if remote else "false")
    with mock.patch(
        "mlflow.assistant.providers.base.shutil.which", return_value=docker_path
    ) as which:
        assert assistant_sandbox_enabled() is (override == "true")
    which.assert_not_called()


@pytest.mark.parametrize("enabled", [True, False])
def test_bash_routes_between_sandbox_and_host(enabled):
    perms = PermissionsConfig(full_access=True)
    with (
        mock.patch(
            "mlflow.assistant.providers.tool_executor.assistant_sandbox_enabled",
            return_value=enabled,
        ),
        mock.patch(
            "mlflow.assistant.providers.tool_executor._execute_bash_in_sandbox",
            new=AsyncMock(return_value=("sandbox", False)),
        ) as in_sandbox,
        mock.patch(
            "mlflow.assistant.providers.tool_executor._execute_bash_on_host",
            new=AsyncMock(return_value=("host", False)),
        ) as on_host,
    ):
        result = _run(execute_tool("Bash", {"command": "mlflow --version"}, permissions=perms))

    if enabled:
        assert result == ("sandbox", False)
        in_sandbox.assert_called_once()
        on_host.assert_not_called()
    else:
        assert result == ("host", False)
        on_host.assert_called_once()
        in_sandbox.assert_not_called()


def test_execute_bash_in_sandbox_full_access_uses_shell():
    with mock.patch(
        "mlflow.server.sandbox.run_in_sandbox",
        return_value=SandboxResult(exit_code=0, output="done\n"),
    ) as run:
        result = _run(
            _execute_bash_in_sandbox(
                "mlflow runs list && echo hi", None, "http://127.0.0.1:5000", full_access=True
            )
        )

    assert result == ("done", False)
    args, kwargs = run.call_args
    assert args[0] == ["mlflow runs list && echo hi"]
    assert kwargs["use_shell"] is True
    # Loopback tracking URI is rewritten so it is reachable from inside the container.
    assert kwargs["environment"]["MLFLOW_TRACKING_URI"] == "http://host.docker.internal:5000"


def test_execute_bash_in_sandbox_restricted_uses_argv(workspace):
    with mock.patch(
        "mlflow.server.sandbox.run_in_sandbox",
        return_value=SandboxResult(exit_code=0, output="ok"),
    ) as run:
        result = _run(
            _execute_bash_in_sandbox("mlflow runs list", workspace, None, full_access=False)
        )

    assert result == ("ok", False)
    args, kwargs = run.call_args
    assert args[0] == ["mlflow", "runs", "list"]
    assert kwargs["use_shell"] is False
    assert kwargs["workdir"] == workspace


def test_execute_bash_in_sandbox_nonzero_exit_is_error():
    with mock.patch(
        "mlflow.server.sandbox.run_in_sandbox",
        return_value=SandboxResult(exit_code=2, output="boom"),
    ):
        result, is_error = _run(
            _execute_bash_in_sandbox("mlflow bogus", None, None, full_access=True)
        )

    assert is_error is True
    assert "boom" in result


def test_execute_bash_in_sandbox_timeout():
    with mock.patch(
        "mlflow.server.sandbox.run_in_sandbox",
        return_value=SandboxResult(exit_code=-1, output="", timed_out=True),
    ):
        result, is_error = _run(
            _execute_bash_in_sandbox("sleep 1000", None, None, full_access=True)
        )

    assert is_error is True
    assert "timed out" in result


def test_execute_bash_in_sandbox_forwards_config_not_secrets(monkeypatch):
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "http://localhost:5000")
    monkeypatch.setenv("DATABRICKS_TOKEN", "secret-token")
    with mock.patch(
        "mlflow.server.sandbox.run_in_sandbox",
        return_value=SandboxResult(exit_code=0, output="ok"),
    ) as run:
        _run(_execute_bash_in_sandbox("mlflow --version", None, None, full_access=True))

    env = run.call_args.kwargs["environment"]
    # Non-secret config is forwarded (and loopback-rewritten); host credentials are not.
    assert env["MLFLOW_REGISTRY_URI"] == "http://host.docker.internal:5000"
    assert "DATABRICKS_TOKEN" not in env


# The credential portion is assembled from parts so no literal connection string is committed to
# source (keeps the secret scanner from flagging these deliberately credential-bearing fixtures).
_FAKE_USERINFO = "u:p"


def test_execute_bash_in_sandbox_drops_registry_uri_with_credentials(monkeypatch):
    # A SQLAlchemy registry URI can embed credentials; forwarding it would leak them into the
    # sandbox, so it must be dropped rather than passed through.
    monkeypatch.setenv(
        "MLFLOW_REGISTRY_URI", f"postgresql://{_FAKE_USERINFO}@db.internal:5432/registry"
    )
    with mock.patch(
        "mlflow.server.sandbox.run_in_sandbox",
        return_value=SandboxResult(exit_code=0, output="ok"),
    ) as run:
        _run(_execute_bash_in_sandbox("mlflow --version", None, None, full_access=True))

    assert "MLFLOW_REGISTRY_URI" not in run.call_args.kwargs["environment"]


def test_execute_bash_in_sandbox_drops_tracking_uri_with_credentials():
    # Same for a credential-bearing tracking URI passed to the sandbox.
    with mock.patch(
        "mlflow.server.sandbox.run_in_sandbox",
        return_value=SandboxResult(exit_code=0, output="ok"),
    ) as run:
        _run(
            _execute_bash_in_sandbox(
                "mlflow --version",
                None,
                f"postgresql://{_FAKE_USERINFO}@db.internal:5432/tracking",
                full_access=True,
            )
        )

    assert "MLFLOW_TRACKING_URI" not in run.call_args.kwargs["environment"]


def test_execute_bash_in_sandbox_unavailable_does_not_fall_back_to_host():
    with mock.patch(
        "mlflow.server.sandbox.run_in_sandbox",
        side_effect=SandboxUnavailableError("no daemon"),
    ):
        result, is_error = _run(
            _execute_bash_in_sandbox("mlflow --version", None, None, full_access=True)
        )

    assert is_error is True
    assert "Sandbox is enabled" in result
    assert "no daemon" in result
