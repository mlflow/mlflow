import asyncio
from unittest import mock

import pytest

from mlflow.assistant.config import PermissionsConfig
from mlflow.assistant.providers.tool_executor import build_tools_schema, execute_tool


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


@pytest.mark.parametrize("cmd", ["wc -c {f}", "ls {d}", "stat {f}", "du {f}"])
def test_bash_allows_file_size_commands(workspace, cmd):
    command = cmd.format(f=workspace / "README.md", d=workspace)
    result, is_error = _run(execute_tool("Bash", {"command": command}))
    assert "Permission denied" not in result


def test_trace_analyse_returns_filtered_result():
    with mock.patch(
        "mlflow.assistant.providers.tool_executor.apply_jq_to_trace",
        return_value='["chat","llm"]\n',
    ) as mock_jq:
        result, is_error = _run(
            execute_tool(
                "trace_analyse",
                {"trace_id": "tr-1", "jq_filter": "[.data.spans[].name]"},
                tracking_uri="http://localhost:5000",
            )
        )
    assert not is_error
    assert result == '["chat","llm"]\n'
    mock_jq.assert_called_once_with(
        "tr-1", "[.data.spans[].name]", tracking_uri="http://localhost:5000"
    )


def test_trace_analyse_reports_error():
    with mock.patch(
        "mlflow.assistant.providers.tool_executor.apply_jq_to_trace",
        side_effect=ValueError("jq error: boom"),
    ) as mock_jq:
        result, is_error = _run(
            execute_tool("trace_analyse", {"trace_id": "tr-1", "jq_filter": "."})
        )
    assert is_error
    assert "jq error: boom" in result
    mock_jq.assert_called_once()


def test_build_tools_schema_includes_trace_analyse():
    tools = {t["function"]["name"]: t["function"] for t in build_tools_schema()}
    assert "trace_analyse" in tools
    params = tools["trace_analyse"]["parameters"]
    assert params["required"] == ["trace_id"]
    assert set(params["properties"]) == {"trace_id", "jq_filter"}


def test_trace_analyse_requires_trace_id():
    result, is_error = _run(execute_tool("trace_analyse", {"jq_filter": "."}))
    assert is_error
    assert "No trace_id provided" in result


def test_trace_analyse_allowed_without_full_access():
    with mock.patch(
        "mlflow.assistant.providers.tool_executor.apply_jq_to_trace",
        return_value="{}",
    ) as mock_jq:
        result, is_error = _run(
            execute_tool("trace_analyse", {"trace_id": "tr-1"}, permissions=PermissionsConfig())
        )
    assert not is_error
    mock_jq.assert_called_once_with("tr-1", None, tracking_uri=None)
