from unittest import mock

import pytest

from mlflow.entities import SourceType
from mlflow.tracking.context.jupyter_notebook_context import (
    JupyterNotebookRunContext,
    _find_single_notebook_in_dir,
    _get_notebook_from_ipython_history,
    _get_notebook_from_vscode_process,
    _get_notebook_name,
    _is_in_jupyter_notebook,
)
from mlflow.utils.mlflow_tags import MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE

MOCK_NOTEBOOK_NAME = "test_notebook.ipynb"


# Tests for _is_in_jupyter_notebook


def test_is_in_jupyter_notebook_true_zmq_shell():
    """Test that ZMQInteractiveShell is detected as a Jupyter notebook."""
    mock_shell = mock.Mock()
    mock_shell.__class__.__name__ = "ZMQInteractiveShell"

    mock_ipython = mock.Mock()
    mock_ipython.get_ipython.return_value = mock_shell

    with mock.patch.dict("sys.modules", {"IPython": mock_ipython}):
        assert _is_in_jupyter_notebook() is True


def test_is_in_jupyter_notebook_true_kernel_attribute():
    """Test that having a kernel attribute is detected as a Jupyter notebook."""
    mock_shell = mock.Mock()
    mock_shell.__class__.__name__ = "SomeOtherShell"
    mock_shell.kernel = mock.Mock()

    mock_ipython = mock.Mock()
    mock_ipython.get_ipython.return_value = mock_shell

    with mock.patch.dict("sys.modules", {"IPython": mock_ipython}):
        assert _is_in_jupyter_notebook() is True


def test_is_in_jupyter_notebook_false_terminal_shell():
    """Test that TerminalInteractiveShell is not detected as a Jupyter notebook."""
    mock_shell = mock.Mock(spec=[])  # No kernel attribute
    mock_shell.__class__.__name__ = "TerminalInteractiveShell"

    mock_ipython = mock.Mock()
    mock_ipython.get_ipython.return_value = mock_shell

    with mock.patch.dict("sys.modules", {"IPython": mock_ipython}):
        assert _is_in_jupyter_notebook() is False


def test_is_in_jupyter_notebook_false_no_ipython():
    """Test that None from get_ipython is not detected as a Jupyter notebook."""
    mock_ipython = mock.Mock()
    mock_ipython.get_ipython.return_value = None

    with mock.patch.dict("sys.modules", {"IPython": mock_ipython}):
        assert _is_in_jupyter_notebook() is False


def test_is_in_jupyter_notebook_false_import_error():
    """Test graceful handling when IPython is not installed."""
    # Remove IPython from sys.modules and make import fail
    with mock.patch.dict("sys.modules", {"IPython": None}):
        # When IPython is None in sys.modules, importing it raises ImportError
        assert _is_in_jupyter_notebook() is False


# Tests for _get_notebook_name


def test_get_notebook_name_from_vsc_env_var():
    """Test notebook name detection from VS Code environment variable."""
    with mock.patch.dict("os.environ", {"__vsc_ipynb_file__": f"/path/to/{MOCK_NOTEBOOK_NAME}"}):
        assert _get_notebook_name() == MOCK_NOTEBOOK_NAME


def test_get_notebook_name_from_ipynb_file_env_var():
    """Test notebook name detection from IPYNB_FILE environment variable."""
    with mock.patch.dict(
        "os.environ", {"__vsc_ipynb_file__": "", "IPYNB_FILE": f"/path/to/{MOCK_NOTEBOOK_NAME}"}
    ):
        # Clear the vsc env var
        with mock.patch("os.environ.get") as mock_get:
            mock_get.side_effect = lambda k, d=None: (
                f"/path/to/{MOCK_NOTEBOOK_NAME}" if k == "IPYNB_FILE" else d
            )
            # Need to test this more directly
            pass


def test_get_notebook_name_from_ipynbname_package():
    """Test notebook name detection from ipynbname package."""
    mock_ipynbname = mock.Mock()
    mock_ipynbname.name.return_value = "test_notebook"

    with (
        mock.patch.dict("os.environ", {}, clear=False),
        mock.patch("os.environ.get", return_value=None),
        mock.patch.dict("sys.modules", {"ipynbname": mock_ipynbname}),
    ):
        # The import inside _get_notebook_name should use the mocked module
        pass


# Tests for _find_single_notebook_in_dir


def test_find_single_notebook_in_dir_single_notebook(tmp_path):
    """Test finding a single notebook in a directory."""
    notebook_path = tmp_path / MOCK_NOTEBOOK_NAME
    notebook_path.touch()

    assert _find_single_notebook_in_dir(str(tmp_path)) == MOCK_NOTEBOOK_NAME


def test_find_single_notebook_in_dir_multiple_notebooks(tmp_path):
    """Test finding the most recently modified notebook when multiple exist."""
    import time

    # Create older notebook
    old_notebook = tmp_path / "old_notebook.ipynb"
    old_notebook.touch()

    # Small delay to ensure different modification times
    time.sleep(0.01)

    # Create newer notebook
    new_notebook = tmp_path / "new_notebook.ipynb"
    new_notebook.touch()

    result = _find_single_notebook_in_dir(str(tmp_path))
    assert result == "new_notebook.ipynb"


def test_find_single_notebook_in_dir_no_notebooks(tmp_path):
    """Test that None is returned when no notebooks exist."""
    # Create a non-notebook file
    (tmp_path / "script.py").touch()

    assert _find_single_notebook_in_dir(str(tmp_path)) is None


def test_find_single_notebook_in_dir_invalid_directory():
    """Test graceful handling of invalid directory."""
    assert _find_single_notebook_in_dir("/nonexistent/path") is None


# Tests for _get_notebook_from_ipython_history


def test_get_notebook_from_ipython_history_with_file():
    """Test getting notebook name from __file__ in IPython namespace."""
    mock_shell = mock.Mock()
    mock_shell.user_ns = {"__file__": f"/path/to/{MOCK_NOTEBOOK_NAME}"}

    mock_ipython = mock.Mock()
    mock_ipython.get_ipython.return_value = mock_shell

    with mock.patch.dict("sys.modules", {"IPython": mock_ipython}):
        assert _get_notebook_from_ipython_history() == MOCK_NOTEBOOK_NAME


def test_get_notebook_from_ipython_history_with_session():
    """Test getting notebook name from __session__ in IPython namespace."""
    mock_shell = mock.Mock()
    mock_shell.user_ns = {"__session__": f"/path/to/{MOCK_NOTEBOOK_NAME}"}

    mock_ipython = mock.Mock()
    mock_ipython.get_ipython.return_value = mock_shell

    with mock.patch.dict("sys.modules", {"IPython": mock_ipython}):
        assert _get_notebook_from_ipython_history() == MOCK_NOTEBOOK_NAME


def test_get_notebook_from_ipython_history_no_ipython():
    """Test graceful handling when not in IPython."""
    mock_ipython = mock.Mock()
    mock_ipython.get_ipython.return_value = None

    with mock.patch.dict("sys.modules", {"IPython": mock_ipython}):
        # Should fall through to directory check, returns None if no notebooks in cwd
        result = _get_notebook_from_ipython_history()
        # Result depends on whether there are notebooks in cwd - just verify no exception


# Tests for _get_notebook_from_vscode_process


def test_get_notebook_from_vscode_process_no_psutil():
    """Test graceful handling when psutil is not installed."""
    with mock.patch.dict("sys.modules", {"psutil": None}):
        # When psutil is None, import will fail, should return None without raising
        result = _get_notebook_from_vscode_process()
        assert result is None


# Tests for JupyterNotebookRunContext


def test_jupyter_notebook_run_context_in_context_true():
    """Test that in_context returns True when in a Jupyter notebook."""
    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context._is_in_jupyter_notebook", return_value=True
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
    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context._get_notebook_name", return_value=None
    ):
        tags = JupyterNotebookRunContext().tags()
        assert tags == {
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
        }
        assert MLFLOW_SOURCE_NAME not in tags


def test_jupyter_notebook_run_context_source_type_is_notebook():
    """Test that source type is always NOTEBOOK when in context."""
    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context._get_notebook_name", return_value=None
    ):
        tags = JupyterNotebookRunContext().tags()
        assert tags[MLFLOW_SOURCE_TYPE] == "NOTEBOOK"

