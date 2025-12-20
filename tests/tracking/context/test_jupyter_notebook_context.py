import json
from unittest import mock

import pytest

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


@pytest.mark.parametrize(
    ("shell_name", "has_kernel", "is_in_ipython", "expected"),
    [
        ("ZMQInteractiveShell", False, True, True),
        ("SomeOtherShell", True, True, True),
        ("TerminalInteractiveShell", False, True, False),
        ("AnyShell", False, False, False),
    ],
)
def test_is_in_jupyter_notebook(shell_name, has_kernel, is_in_ipython, expected):
    mock_shell = mock.Mock(spec=["kernel"] if has_kernel else [])
    mock_shell.__class__.__name__ = shell_name
    if has_kernel:
        mock_shell.kernel = mock.Mock()

    mock_ipython = mock.Mock()
    mock_ipython.get_ipython.return_value = mock_shell

    with (
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context.is_running_in_ipython_environment",
            return_value=is_in_ipython,
        ),
        mock.patch.dict("sys.modules", {"IPython": mock_ipython}),
    ):
        assert _is_in_jupyter_notebook() is expected


@pytest.mark.parametrize(
    ("user_ns", "is_in_ipython", "expected"),
    [
        ({"__vsc_ipynb_file__": MOCK_NOTEBOOK_PATH}, True, MOCK_NOTEBOOK_PATH),
        ({}, True, None),
        ({"__vsc_ipynb_file__": MOCK_NOTEBOOK_PATH}, False, None),
    ],
)
def test_get_vscode_notebook_path(user_ns, is_in_ipython, expected):
    mock_shell = mock.Mock()
    mock_shell.user_ns = user_ns

    mock_ipython = mock.Mock()
    mock_ipython.get_ipython.return_value = mock_shell

    with (
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context.is_running_in_ipython_environment",
            return_value=is_in_ipython,
        ),
        mock.patch.dict("sys.modules", {"IPython": mock_ipython}),
    ):
        assert _get_vscode_notebook_path() == expected


def test_get_kernel_id_success():
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
    with mock.patch.dict("sys.modules", {"ipykernel": None}):
        result = _get_kernel_id()
        assert result is None


def test_get_running_servers_finds_servers(tmp_path):
    server_info = {"url": "http://localhost:8888/", "token": "test_token"}
    server_file = tmp_path / "nbserver-12345.json"
    server_file.write_text(json.dumps(server_info))

    mock_jupyter_core = mock.Mock()
    mock_jupyter_core.paths.jupyter_runtime_dir.return_value = str(tmp_path)

    with mock.patch.dict(
        "sys.modules",
        {"jupyter_core": mock_jupyter_core, "jupyter_core.paths": mock_jupyter_core.paths},
    ):
        with mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context.Path",
            return_value=tmp_path,
        ):
            servers = list(_get_running_servers())


def test_get_running_servers_no_servers(tmp_path):
    mock_jupyter_core = mock.Mock()
    mock_jupyter_core.paths.jupyter_runtime_dir.return_value = str(tmp_path)

    with mock.patch.dict(
        "sys.modules",
        {"jupyter_core": mock_jupyter_core, "jupyter_core.paths": mock_jupyter_core.paths},
    ):
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
    with mock.patch.dict("sys.modules", {"jupyter_core": None, "jupyter_core.paths": None}):
        servers = list(_get_running_servers())
        assert servers == []


def test_get_sessions_notebook_finds_notebook():
    server = {"url": "http://localhost:8888/", "token": "test_token"}
    mock_sessions = [{"kernel": {"id": MOCK_KERNEL_ID}, "path": MOCK_NOTEBOOK_PATH}]

    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context.urlopen"
    ) as mock_urlopen:
        mock_response = mock.Mock()
        mock_response.__enter__ = mock.Mock(return_value=mock_response)
        mock_response.__exit__ = mock.Mock(return_value=False)
        mock_response.read.return_value = json.dumps(mock_sessions).encode()
        mock_urlopen.return_value = mock_response

        with mock.patch("json.load", return_value=mock_sessions):
            result = _get_sessions_notebook(server, MOCK_KERNEL_ID)
            assert result == MOCK_NOTEBOOK_PATH


def test_get_sessions_notebook_no_matching_kernel():
    server = {"url": "http://localhost:8888/", "token": "test_token"}
    mock_sessions = [{"kernel": {"id": "different_kernel"}, "path": "other_notebook.ipynb"}]

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
    server = {"url": "http://localhost:8888/", "token": "test_token"}

    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context.urlopen",
        side_effect=Exception("Connection refused"),
    ):
        result = _get_sessions_notebook(server, MOCK_KERNEL_ID)
        assert result is None


def test_get_sessions_notebook_with_jupyterhub_token():
    server = {"url": "http://localhost:8888/", "token": ""}
    mock_sessions = [{"kernel": {"id": MOCK_KERNEL_ID}, "path": MOCK_NOTEBOOK_PATH}]

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
            call_args = mock_urlopen.call_args
            assert "hub_token" in call_args[0][0]


@pytest.mark.parametrize(
    ("vscode_path", "env_vars", "sessions_path", "expected"),
    [
        (MOCK_NOTEBOOK_PATH, {}, None, MOCK_NOTEBOOK_NAME),
        (None, {"__vsc_ipynb_file__": MOCK_NOTEBOOK_PATH}, None, MOCK_NOTEBOOK_NAME),
        (None, {"IPYNB_FILE": MOCK_NOTEBOOK_PATH}, None, MOCK_NOTEBOOK_NAME),
        (None, {}, MOCK_NOTEBOOK_PATH, MOCK_NOTEBOOK_NAME),
        (None, {}, None, None),
    ],
)
def test_get_notebook_name(vscode_path, env_vars, sessions_path, expected):
    _get_notebook_name.cache_clear()

    with (
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context._get_vscode_notebook_path",
            return_value=vscode_path,
        ),
        mock.patch.dict("os.environ", env_vars, clear=True),
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context._get_notebook_path_from_sessions",
            return_value=sessions_path,
        ),
    ):
        assert _get_notebook_name() == expected


def test_get_notebook_name_is_cached():
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
        result1 = _get_notebook_name()
        result2 = _get_notebook_name()
        result3 = _get_notebook_name()

        assert result1 == MOCK_NOTEBOOK_NAME
        assert result2 == MOCK_NOTEBOOK_NAME
        assert result3 == MOCK_NOTEBOOK_NAME
        assert call_count == 1


def test_get_notebook_path_from_sessions_success():
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
    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context._get_kernel_id",
        return_value=None,
    ):
        result = _get_notebook_path_from_sessions()
        assert result is None


def test_get_notebook_path_from_sessions_no_servers():
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


@pytest.mark.parametrize(
    ("is_in_jupyter", "expected"),
    [
        (True, True),
        (False, False),
    ],
)
def test_jupyter_notebook_run_context_in_context(is_in_jupyter, expected):
    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context._is_in_jupyter_notebook",
        return_value=is_in_jupyter,
    ):
        assert JupyterNotebookRunContext().in_context() is expected


@pytest.mark.parametrize(
    ("notebook_name", "expected_tags"),
    [
        (
            MOCK_NOTEBOOK_NAME,
            {
                MLFLOW_SOURCE_NAME: MOCK_NOTEBOOK_NAME,
                MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
            },
        ),
        (
            None,
            {
                MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
            },
        ),
    ],
)
def test_jupyter_notebook_run_context_tags(notebook_name, expected_tags):
    _get_notebook_name.cache_clear()

    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context._get_notebook_name",
        return_value=notebook_name,
    ):
        tags = JupyterNotebookRunContext().tags()
        assert tags == expected_tags
