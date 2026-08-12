import asyncio
import os

import pytest

from mlflow.assistant.config import PermissionsConfig
from mlflow.assistant.providers.tool_executor import execute_tool, static_permission_error


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


def test_bash_mlflow_run_denied_without_cwd():
    # "mlflow run <uri>" executes the target project's entry-point command via a real
    # shell (mlflow/projects/backend/local.py), independent of cwd, so it must be denied
    # outright in restricted mode rather than gated by a configured project directory.
    result, is_error = _run(execute_tool("Bash", {"command": "mlflow run /some/project"}))
    assert is_error
    assert "Permission denied" in result


def test_bash_mlflow_run_denied_with_cwd(workspace):
    # Denied even with cwd configured: there's no argument-level fix, since the
    # entry-point command is arbitrary shell content by design.
    result, is_error = _run(
        execute_tool("Bash", {"command": "mlflow run /some/project"}, cwd=workspace)
    )
    assert is_error
    assert "Permission denied" in result


def test_bash_mlflow_run_allowed_with_full_access():
    perms = PermissionsConfig(full_access=True)
    result, is_error = _run(
        execute_tool("Bash", {"command": "mlflow run --help"}, permissions=perms)
    )
    assert not is_error


def test_bash_mlflow_run_actually_denies_entry_point_execution(tmp_path):
    # End-to-end regression guard: reproduces the exact RCE this closes. An MLproject
    # entry point can run arbitrary shell commands; before this fix, "mlflow run <path>"
    # passed the allowlist (argv[0] == "mlflow") and actually executed it.
    project_dir = tmp_path / "evil-project"
    project_dir.mkdir()
    marker = tmp_path / "pwned.txt"
    (project_dir / "MLproject").write_text(
        f'name: evil\nentry_points:\n  main:\n    command: "touch {marker}"\n'
    )
    result, is_error = _run(
        execute_tool("Bash", {"command": f"mlflow run {project_dir} --env-manager local"})
    )
    assert is_error
    assert not marker.exists()


def test_bash_mlflow_root_option_before_subcommand_does_not_bypass_allowlist(tmp_path):
    # Regression guard for a review finding: a naive parser that assumes argv[1] is the
    # subcommand can be fooled by a root-level option before it, e.g. mlflow's
    # "--env-file" flag. "mlflow --env-file X run <project>" must resolve to "run" and
    # be denied the same as "mlflow run <project>", not fall through as unrecognized.
    project_dir = tmp_path / "evil-project"
    project_dir.mkdir()
    marker = tmp_path / "pwned.txt"
    (project_dir / "MLproject").write_text(
        f'name: evil\nentry_points:\n  main:\n    command: "touch {marker}"\n'
    )
    env_file = tmp_path / ".env"
    env_file.write_text("")
    command = f"mlflow --env-file {env_file} run {project_dir} --env-manager local"
    result, is_error = _run(execute_tool("Bash", {"command": command}))
    assert is_error
    assert not marker.exists()


def test_bash_mlflow_artifacts_download_schemeless_local_path_denied(workspace):
    # Regression guard: mlflow treats a schemeless absolute path the same as a "file://"
    # URI (both resolve to a LocalArtifactRepository), so checking only for the literal
    # "file://" prefix misses this. "-u /etc/passwd" must be denied the same way.
    result, is_error = _run(
        execute_tool(
            "Bash",
            {
                "command": (
                    f"mlflow artifacts download --artifact-uri /etc/passwd --dst-path {workspace}"
                )
            },
            cwd=workspace,
        )
    )
    assert is_error
    assert "Permission denied" in result


def test_bash_mlflow_artifacts_download_repeated_flag_uses_last_value(workspace):
    # Regression guard: Click uses the LAST occurrence of a repeated option, so a
    # permission check that inspects the first occurrence can be shown a safe URI while
    # the command that actually runs uses a later, dangerous one.
    command = (
        "mlflow artifacts download --artifact-uri runs:/abc/model "
        f"--artifact-uri file:///etc/passwd --dst-path {workspace}"
    )
    result, is_error = _run(execute_tool("Bash", {"command": command}, cwd=workspace))
    assert is_error
    assert "Permission denied" in result


def test_bash_mlflow_artifacts_download_attached_short_option_denied(workspace):
    # Regression guard: "-uVALUE" (no space) is valid Click syntax for a short option
    # with an argument; a parser that only recognizes "-u VALUE" as two tokens misses it.
    result, is_error = _run(
        execute_tool(
            "Bash",
            {"command": f"mlflow artifacts download -ufile:///etc/passwd -d{workspace}"},
            cwd=workspace,
        )
    )
    assert is_error
    assert "Permission denied" in result


def test_bash_mlflow_models_predict_not_on_allowlist():
    # Regression guard: mlflow has dozens of subcommands beyond run/artifacts that
    # execute code or touch the local filesystem (models predict/serve, deployments
    # run-local, db upgrade, server, ...). The allowlist denies anything not
    # specifically audited, rather than only the two subcommands found so far.
    result, is_error = _run(
        execute_tool("Bash", {"command": "mlflow models predict --env-manager local -m x"})
    )
    assert is_error
    assert "Permission denied" in result


def test_bash_mlflow_env_file_outside_workspace_denied(workspace, tmp_path_factory):
    # Regression guard: "--env-file" is an eager root option that loads a dotenv file's
    # content as environment variables before any subcommand runs. The allowlist only
    # validated the resolved subcommand path, not root options, so
    # "mlflow --env-file <secret> experiments search" resolved to an allowed path
    # while the real CLI loaded an arbitrary local file regardless of that path.
    secret = tmp_path_factory.mktemp("outside") / "secret.env"
    secret.write_text("SECRET_API_KEY=sk-super-secret-12345")
    command = f"mlflow --env-file {secret} experiments search"
    result, is_error = _run(execute_tool("Bash", {"command": command}, cwd=workspace))
    assert is_error
    assert "Permission denied" in result


def test_bash_mlflow_env_file_requires_cwd():
    result, is_error = _run(
        execute_tool("Bash", {"command": "mlflow --env-file /etc/hosts experiments search"})
    )
    assert is_error
    assert "Permission denied" in result


def test_bash_mlflow_env_file_actually_denies_loading(workspace, tmp_path_factory):
    # End-to-end regression guard: confirms the real "mlflow" CLI never gets to load the
    # file, not just that static_permission_error objects to it.
    secret = tmp_path_factory.mktemp("outside") / "secret.env"
    secret.write_text("MLFLOW_POC_LEAK=leaked-value-12345")
    command = f"mlflow --env-file {secret} experiments search"
    result, is_error = _run(execute_tool("Bash", {"command": command}, cwd=workspace))
    assert is_error
    assert os.environ.get("MLFLOW_POC_LEAK") is None


def test_bash_mlflow_env_file_within_workspace_allowed(workspace):
    env_file = workspace / "local.env"
    env_file.write_text("")
    denial = static_permission_error(
        "Bash",
        {"command": f"mlflow --env-file {env_file} experiments search"},
        PermissionsConfig(),
        workspace,
    )
    assert denial is None


def test_bash_mlflow_artifacts_download_missing_dst_path_denied(workspace):
    # Regression guard: --dst-path is optional; omitting it left path_value=None, which
    # skipped the containment check entirely and allowed mlflow to pick its own
    # destination outside the workspace. Downloads must specify an in-workspace
    # destination explicitly.
    result, is_error = _run(
        execute_tool(
            "Bash",
            {"command": "mlflow artifacts download --artifact-uri runs:/abc/model"},
            cwd=workspace,
        )
    )
    assert is_error
    assert "Permission denied" in result


def test_bash_mlflow_artifacts_download_requires_cwd():
    result, is_error = _run(
        execute_tool(
            "Bash",
            {"command": "mlflow artifacts download --artifact-uri runs:/abc/model"},
        )
    )
    assert is_error
    assert "Permission denied" in result


def test_bash_mlflow_artifacts_download_file_uri_denied_even_with_cwd(workspace):
    # "file://" is a raw absolute filesystem path, unrelated to any run/experiment and
    # not confined by --dst-path, so it must be denied regardless of cwd.
    result, is_error = _run(
        execute_tool(
            "Bash",
            {
                "command": (
                    "mlflow artifacts download --artifact-uri file:///etc/passwd "
                    f"--dst-path {workspace}"
                )
            },
            cwd=workspace,
        )
    )
    assert is_error
    assert "Permission denied" in result


def test_bash_mlflow_artifacts_download_dst_path_outside_workspace_denied(
    workspace, tmp_path_factory
):
    # workspace is itself tmp_path (see the fixture above), so a genuinely separate
    # directory must come from tmp_path_factory, not tmp_path, to actually be "outside".
    outside = tmp_path_factory.mktemp("outside")
    result, is_error = _run(
        execute_tool(
            "Bash",
            {
                "command": (
                    f"mlflow artifacts download --artifact-uri runs:/abc/model --dst-path {outside}"
                )
            },
            cwd=workspace,
        )
    )
    assert is_error
    assert "Permission denied" in result


def test_bash_mlflow_artifacts_download_allowed_within_workspace(workspace):
    # Legitimate use (a tracked-storage URI, destination inside the workspace) must not
    # be broken by the fix for the file:// / cwd=None bypasses.
    denial = static_permission_error(
        "Bash",
        {
            "command": (
                "mlflow artifacts download --artifact-uri runs:/abc/model "
                f"--dst-path {workspace / 'downloaded'}"
            )
        },
        PermissionsConfig(),
        workspace,
    )
    assert denial is None


def test_bash_mlflow_artifacts_log_artifact_outside_workspace_denied(workspace, tmp_path_factory):
    secret = tmp_path_factory.mktemp("outside") / "secret.env"
    secret.write_text("SECRET_API_KEY=sk-super-secret-12345")
    result, is_error = _run(
        execute_tool(
            "Bash",
            {"command": f"mlflow artifacts log-artifact --local-file {secret} --run-id abc"},
            cwd=workspace,
        )
    )
    assert is_error
    assert "Permission denied" in result


def test_bash_mlflow_artifacts_log_artifacts_local_dir_outside_workspace_denied(
    workspace, tmp_path_factory
):
    outside_dir = tmp_path_factory.mktemp("outside")
    result, is_error = _run(
        execute_tool(
            "Bash",
            {"command": (f"mlflow artifacts log-artifacts --local-dir {outside_dir} --run-id abc")},
            cwd=workspace,
        )
    )
    assert is_error
    assert "Permission denied" in result


def test_bash_mlflow_artifacts_log_artifact_allowed_within_workspace(workspace):
    # Legitimate use (logging a file that lives inside the workspace) must not be broken.
    report = workspace / "report.txt"
    report.write_text("results")
    denial = static_permission_error(
        "Bash",
        {"command": f"mlflow artifacts log-artifact --local-file {report} --run-id abc"},
        PermissionsConfig(),
        workspace,
    )
    assert denial is None


def test_bash_mlflow_artifacts_list_unaffected():
    # Read-only metadata listing can't reach an arbitrary local path, so it's unaffected
    # by the new checks and still needs no configured project directory.
    denial = static_permission_error(
        "Bash",
        {"command": "mlflow artifacts list --run-id abc"},
        PermissionsConfig(),
        None,
    )
    assert denial is None


def test_bash_blocks_non_mlflow_commands():
    result, is_error = _run(execute_tool("Bash", {"command": "echo hello"}))
    assert is_error
    assert "Permission denied" in result


def test_bash_allows_mlflow_commands():
    result, is_error = _run(execute_tool("Bash", {"command": "mlflow --version"}))
    assert not is_error


def test_bash_allows_mlflow_help():
    # Regression guard: Click adds an implicit "--help" on every group, parsed under
    # the key "help". _MLFLOW_HARMLESS_ROOT_OPTIONS must include it, or the "unrecognized
    # root option" defensive check denies the documented-as-allowed "mlflow --help".
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
