import asyncio

import pytest

from mlflow.assistant.config import PermissionsConfig
from mlflow.assistant.providers.tool_executor import execute_tool


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
    result, is_error = _run(execute_tool("Bash", {"command": "python3 -c \"print('hello')\""}))
    assert not is_error
    assert "hello" in result


def test_bash_blocks_non_mlflow_commands():
    result, is_error = _run(execute_tool("Bash", {"command": "echo hello"}))
    assert is_error
    assert "Permission denied" in result


def test_bash_allows_mlflow_commands():
    result, is_error = _run(execute_tool("Bash", {"command": "mlflow --version"}))
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
