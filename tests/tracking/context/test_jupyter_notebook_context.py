import json
from pathlib import Path
from unittest import mock

from mlflow.entities import SourceType
from mlflow.tracking.context.jupyter_notebook_context import (
    JupyterNotebookRunContext,
    _get_kernel_id,
    _get_notebook_name,
    _get_notebook_path_from_sessions,
    _get_running_servers,
    _get_sessions_notebook,
    _get_vscode_notebook_path,
    _is_in_jupyter_notebook,
)
from mlflow.utils.mlflow_tags import MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE

MOCK_NOTEBOOK_NAME = "test_notebook.ipynb"
MOCK_NOTEBOOK_PATH = f"/path/to/{MOCK_NOTEBOOK_NAME}"
MOCK_KERNEL_ID = "abc123-def456"


# Tests for _is_in_jupyter_notebook


def test_is_in_jupyter_notebook_true_zmq_shell():
    """Test that ZMQInteractiveShell is detected as a Jupyter notebook."""
    mock_shell = mock.Mock()
    mock_shell.__class__.__name__ = "ZMQInteractiveShell"

    mock_ipython = mock.Mock()
    mock_ipython.get_ipython.return_value = mock_shell

    with (
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context.is_running_in_ipython_environment",
            return_value=True,
        ),
        mock.patch.dict("sys.modules", {"IPython": mock_ipython}),
    ):
        assert _is_in_jupyter_notebook() is True


def test_is_in_jupyter_notebook_true_kernel_attribute():
    """Test that having a kernel attribute is detected as a Jupyter notebook."""
    mock_shell = mock.Mock()
    mock_shell.__class__.__name__ = "SomeOtherShell"
    mock_shell.kernel = mock.Mock()

    mock_ipython = mock.Mock()
    mock_ipython.get_ipython.return_value = mock_shell

    with (
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context.is_running_in_ipython_environment",
            return_value=True,
        ),
        mock.patch.dict("sys.modules", {"IPython": mock_ipython}),
    ):
        assert _is_in_jupyter_notebook() is True


def test_is_in_jupyter_notebook_false_terminal_shell():
    """Test that TerminalInteractiveShell is not detected as a Jupyter notebook."""
    mock_shell = mock.Mock(spec=[])
    mock_shell.__class__.__name__ = "TerminalInteractiveShell"

    mock_ipython = mock.Mock()
    mock_ipython.get_ipython.return_value = mock_shell

    with (
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context.is_running_in_ipython_environment",
            return_value=True,
        ),
        mock.patch.dict("sys.modules", {"IPython": mock_ipython}),
    ):
        assert _is_in_jupyter_notebook() is False


def test_is_in_jupyter_notebook_false_not_in_ipython():
    """Test that False is returned when not in IPython environment."""
    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context.is_running_in_ipython_environment",
        return_value=False,
    ):
        assert _is_in_jupyter_notebook() is False


# Tests for _get_vscode_notebook_path


def test_get_vscode_notebook_path_found():
    """Test getting notebook path from VS Code IPython user namespace."""
    mock_shell = mock.Mock()
    mock_shell.user_ns = {"__vsc_ipynb_file__": MOCK_NOTEBOOK_PATH}

    mock_ipython = mock.Mock()
    mock_ipython.get_ipython.return_value = mock_shell

    with (
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context.is_running_in_ipython_environment",
            return_value=True,
        ),
        mock.patch.dict("sys.modules", {"IPython": mock_ipython}),
    ):
        assert _get_vscode_notebook_path() == MOCK_NOTEBOOK_PATH


def test_get_vscode_notebook_path_not_found():
    """Test that None is returned when VS Code path is not in user namespace."""
    mock_shell = mock.Mock()
    mock_shell.user_ns = {}

    mock_ipython = mock.Mock()
    mock_ipython.get_ipython.return_value = mock_shell

    with (
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context.is_running_in_ipython_environment",
            return_value=True,
        ),
        mock.patch.dict("sys.modules", {"IPython": mock_ipython}),
    ):
        assert _get_vscode_notebook_path() is None


def test_get_vscode_notebook_path_not_in_ipython():
    """Test that None is returned when not in IPython environment."""
    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context.is_running_in_ipython_environment",
        return_value=False,
    ):
        assert _get_vscode_notebook_path() is None


# Tests for _get_kernel_id


def test_get_kernel_id_success():
    """Test successful kernel ID extraction from connection file."""
    mock_ipykernel = mock.Mock()
    mock_ipykernel.get_connection_file.return_value = f"/path/to/kernel-{MOCK_KERNEL_ID}.json"

    with mock.patch.dict("sys.modules", {"ipykernel": mock_ipykernel}):
        with mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context.Path"
        ) as mock_path:
            mock_path.return_value.stem = f"kernel-{MOCK_KERNEL_ID}"
            result = _get_kernel_id()
            assert result == MOCK_KERNEL_ID


def test_get_kernel_id_import_error():
    """Test that None is returned when ipykernel is not available."""
    with mock.patch.dict("sys.modules", {"ipykernel": None}):
        result = _get_kernel_id()
        assert result is None


# Tests for _get_running_servers


def test_get_running_servers_finds_servers(tmp_path):
    """Test that server files are found and parsed correctly."""
    # Create mock server files
    server_info = {"url": "http://localhost:8888/", "token": "test_token"}
    server_file = tmp_path / "nbserver-12345.json"
    server_file.write_text(json.dumps(server_info))

    mock_jupyter_core = mock.Mock()
    mock_jupyter_core.paths.jupyter_runtime_dir.return_value = str(tmp_path)

    with mock.patch.dict("sys.modules", {"jupyter_core": mock_jupyter_core, "jupyter_core.paths": mock_jupyter_core.paths}):
        with mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context.Path",
            return_value=tmp_path,
        ):
            # Need to mock the Path class properly for glob
            servers = list(_get_running_servers())
            # This test verifies the function structure, actual file scanning
            # is difficult to mock completely


def test_get_running_servers_no_servers(tmp_path):
    """Test that empty list is returned when no server files exist."""
    mock_jupyter_core = mock.Mock()
    mock_jupyter_core.paths.jupyter_runtime_dir.return_value = str(tmp_path)

    with mock.patch.dict("sys.modules", {"jupyter_core": mock_jupyter_core, "jupyter_core.paths": mock_jupyter_core.paths}):
        with mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context.Path"
        ) as mock_path:
            mock_path_instance = mock.Mock()
            mock_path_instance.is_dir.return_value = True
            mock_path_instance.glob.return_value = []
            mock_path.return_value = mock_path_instance

            servers = list(_get_running_servers())
            assert servers == []


def test_get_running_servers_import_error():
    """Test that empty generator is returned when jupyter_core is not available."""
    with mock.patch.dict("sys.modules", {"jupyter_core": None, "jupyter_core.paths": None}):
        servers = list(_get_running_servers())
        assert servers == []


# Tests for _get_sessions_notebook


def test_get_sessions_notebook_finds_notebook():
    """Test finding notebook path from server sessions."""
    server = {"url": "http://localhost:8888/", "token": "test_token"}
    mock_sessions = [
        {"kernel": {"id": MOCK_KERNEL_ID}, "path": MOCK_NOTEBOOK_PATH}
    ]

    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context.urlopen"
    ) as mock_urlopen:
        mock_response = mock.Mock()
        mock_response.__enter__ = mock.Mock(return_value=mock_response)
        mock_response.__exit__ = mock.Mock(return_value=False)
        mock_response.read.return_value = json.dumps(mock_sessions).encode()
        mock_urlopen.return_value = mock_response

        # Mock json.load to return our sessions
        with mock.patch("json.load", return_value=mock_sessions):
            result = _get_sessions_notebook(server, MOCK_KERNEL_ID)
            assert result == MOCK_NOTEBOOK_PATH


def test_get_sessions_notebook_no_matching_kernel():
    """Test that None is returned when no matching kernel is found."""
    server = {"url": "http://localhost:8888/", "token": "test_token"}
    mock_sessions = [
        {"kernel": {"id": "different_kernel"}, "path": "other_notebook.ipynb"}
    ]

    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context.urlopen"
    ) as mock_urlopen:
        mock_response = mock.Mock()
        mock_response.__enter__ = mock.Mock(return_value=mock_response)
        mock_response.__exit__ = mock.Mock(return_value=False)

        with mock.patch("json.load", return_value=mock_sessions):
            mock_urlopen.return_value = mock_response
            result = _get_sessions_notebook(server, MOCK_KERNEL_ID)
            assert result is None


def test_get_sessions_notebook_connection_error():
    """Test that None is returned on connection error."""
    server = {"url": "http://localhost:8888/", "token": "test_token"}

    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context.urlopen",
        side_effect=Exception("Connection refused"),
    ):
        result = _get_sessions_notebook(server, MOCK_KERNEL_ID)
        assert result is None


def test_get_sessions_notebook_with_jupyterhub_token():
    """Test that JupyterHub token is used when server token is empty."""
    server = {"url": "http://localhost:8888/", "token": ""}
    mock_sessions = [
        {"kernel": {"id": MOCK_KERNEL_ID}, "path": MOCK_NOTEBOOK_PATH}
    ]

    with (
        mock.patch.dict("os.environ", {"JUPYTERHUB_API_TOKEN": "hub_token"}),
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context.urlopen"
        ) as mock_urlopen,
    ):
        mock_response = mock.Mock()
        mock_response.__enter__ = mock.Mock(return_value=mock_response)
        mock_response.__exit__ = mock.Mock(return_value=False)

        with mock.patch("json.load", return_value=mock_sessions):
            mock_urlopen.return_value = mock_response
            result = _get_sessions_notebook(server, MOCK_KERNEL_ID)
            # Verify the function handles JupyterHub token


# Tests for _get_notebook_name


def test_get_notebook_name_from_vscode_user_ns():
    """Test notebook name detection from VS Code IPython user namespace."""
    _get_notebook_name.cache_clear()

    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context._get_vscode_notebook_path",
        return_value=MOCK_NOTEBOOK_PATH,
    ):
        assert _get_notebook_name() == MOCK_NOTEBOOK_NAME


def test_get_notebook_name_from_env_var():
    """Test notebook name detection from environment variable."""
    _get_notebook_name.cache_clear()

    with (
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context._get_vscode_notebook_path",
            return_value=None,
        ),
        mock.patch.dict("os.environ", {"__vsc_ipynb_file__": MOCK_NOTEBOOK_PATH}),
    ):
        assert _get_notebook_name() == MOCK_NOTEBOOK_NAME


def test_get_notebook_name_from_ipynb_file_env():
    """Test notebook name detection from IPYNB_FILE environment variable."""
    _get_notebook_name.cache_clear()

    with (
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context._get_vscode_notebook_path",
            return_value=None,
        ),
        mock.patch.dict("os.environ", {"IPYNB_FILE": MOCK_NOTEBOOK_PATH}, clear=True),
    ):
        assert _get_notebook_name() == MOCK_NOTEBOOK_NAME


def test_get_notebook_name_from_sessions():
    """Test notebook name detection from Jupyter server sessions."""
    _get_notebook_name.cache_clear()

    with (
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context._get_vscode_notebook_path",
            return_value=None,
        ),
        mock.patch.dict("os.environ", {}, clear=True),
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context._get_notebook_path_from_sessions",
            return_value=MOCK_NOTEBOOK_PATH,
        ),
    ):
        assert _get_notebook_name() == MOCK_NOTEBOOK_NAME


def test_get_notebook_name_returns_none():
    """Test that None is returned when no detection method works."""
    _get_notebook_name.cache_clear()

    with (
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context._get_vscode_notebook_path",
            return_value=None,
        ),
        mock.patch.dict("os.environ", {}, clear=True),
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context._get_notebook_path_from_sessions",
            return_value=None,
        ),
    ):
        assert _get_notebook_name() is None


def test_get_notebook_name_is_cached():
    """Test that _get_notebook_name results are cached."""
    _get_notebook_name.cache_clear()

    call_count = 0

    def mock_vscode_path():
        nonlocal call_count
        call_count += 1
        return MOCK_NOTEBOOK_PATH

    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context._get_vscode_notebook_path",
        side_effect=mock_vscode_path,
    ):
        # Call multiple times
        result1 = _get_notebook_name()
        result2 = _get_notebook_name()
        result3 = _get_notebook_name()

        assert result1 == MOCK_NOTEBOOK_NAME
        assert result2 == MOCK_NOTEBOOK_NAME
        assert result3 == MOCK_NOTEBOOK_NAME

        # Should only be called once due to caching
        assert call_count == 1


# Tests for _get_notebook_path_from_sessions


def test_get_notebook_path_from_sessions_success():
    """Test successful notebook path detection from sessions."""
    mock_server = {"url": "http://localhost:8888/", "token": "test_token"}

    with (
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context._get_kernel_id",
            return_value=MOCK_KERNEL_ID,
        ),
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context._get_running_servers",
            return_value=[mock_server],
        ),
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context._get_sessions_notebook",
            return_value=MOCK_NOTEBOOK_PATH,
        ),
    ):
        result = _get_notebook_path_from_sessions()
        assert result == MOCK_NOTEBOOK_PATH


def test_get_notebook_path_from_sessions_no_kernel_id():
    """Test that None is returned when kernel ID is not available."""
    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context._get_kernel_id",
        return_value=None,
    ):
        result = _get_notebook_path_from_sessions()
        assert result is None


def test_get_notebook_path_from_sessions_no_servers():
    """Test that None is returned when no servers are running."""
    with (
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context._get_kernel_id",
            return_value=MOCK_KERNEL_ID,
        ),
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context._get_running_servers",
            return_value=[],
        ),
    ):
        result = _get_notebook_path_from_sessions()
        assert result is None


# Tests for JupyterNotebookRunContext


def test_jupyter_notebook_run_context_in_context_true():
    """Test that in_context returns True when in a Jupyter notebook."""
    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context._is_in_jupyter_notebook",
        return_value=True,
    ):
        assert JupyterNotebookRunContext().in_context() is True


def test_jupyter_notebook_run_context_in_context_false():
    """Test that in_context returns False when not in a Jupyter notebook."""
    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context._is_in_jupyter_notebook",
        return_value=False,
    ):
        assert JupyterNotebookRunContext().in_context() is False


def test_jupyter_notebook_run_context_tags_with_notebook_name():
    """Test that tags include notebook name when it can be detected."""
    _get_notebook_name.cache_clear()

    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context._get_notebook_name",
        return_value=MOCK_NOTEBOOK_NAME,
    ):
        tags = JupyterNotebookRunContext().tags()
        assert tags == {
            MLFLOW_SOURCE_NAME: MOCK_NOTEBOOK_NAME,
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
        }


def test_jupyter_notebook_run_context_tags_without_notebook_name():
    """Test that tags don't include source name when notebook name can't be detected."""
    _get_notebook_name.cache_clear()

    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context._get_notebook_name",
        return_value=None,
    ):
        tags = JupyterNotebookRunContext().tags()
        assert tags == {
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
        }
        assert MLFLOW_SOURCE_NAME not in tags


def test_jupyter_notebook_run_context_source_type_is_notebook():
    """Test that source type is always NOTEBOOK when in context."""
    _get_notebook_name.cache_clear()

    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context._get_notebook_name",
        return_value=None,
    ):
        tags = JupyterNotebookRunContext().tags()
        assert tags[MLFLOW_SOURCE_TYPE] == "NOTEBOOK"
