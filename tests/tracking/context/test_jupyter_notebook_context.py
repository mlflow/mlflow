"""
Tests for Jupyter Notebook run context provider.
"""

import sys
from unittest import mock

from mlflow.entities import SourceType
from mlflow.tracking.context.jupyter_notebook_context import (
    JupyterNotebookRunContext,
    _get_jupyter_notebook_path,
    _is_in_jupyter_notebook,
)
from mlflow.utils.mlflow_tags import MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE


def test_is_in_jupyter_notebook_without_ipython():
    """Test detection when IPython is not available."""
    with mock.patch.dict("sys.modules", {"IPython": None}):
        assert not _is_in_jupyter_notebook()


def test_is_in_jupyter_notebook_not_in_notebook():
    """Test detection when IPython is available but not in a notebook."""
    with mock.patch("IPython.get_ipython", return_value=None):
        assert not _is_in_jupyter_notebook()


def test_is_in_jupyter_notebook_in_notebook():
    """Test detection when running in a Jupyter notebook."""
    mock_ipython = mock.MagicMock()
    mock_ipython.kernel = mock.MagicMock()  # notebooks have a kernel attribute

    with mock.patch("IPython.get_ipython", return_value=mock_ipython):
        assert _is_in_jupyter_notebook()


def test_is_in_jupyter_notebook_in_ipython_terminal():
    """Test detection when in IPython terminal (not a notebook)."""
    mock_ipython = mock.MagicMock()
    del mock_ipython.kernel  # terminal IPython doesn't have kernel attribute

    with mock.patch("IPython.get_ipython", return_value=mock_ipython):
        assert not _is_in_jupyter_notebook()


def test_get_jupyter_notebook_path_not_in_jupyter():
    """Test getting notebook path when not in Jupyter."""
    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context._is_in_jupyter_notebook",
        return_value=False,
    ):
        assert _get_jupyter_notebook_path() is None


def test_get_jupyter_notebook_path_vscode_jupyter():
    """Test getting notebook path in VS Code Jupyter extension."""
    test_notebook_path = "/path/to/notebook.ipynb"

    mock_ipython = mock.MagicMock()
    mock_ipython.kernel = mock.MagicMock()
    mock_ipython.config = {}

    mock_main = mock.MagicMock()
    mock_main.__vsc_ipynb_file__ = test_notebook_path

    old_main = sys.modules.get("__main__")
    sys.modules["__main__"] = mock_main

    try:
        with (
            mock.patch("IPython.get_ipython", return_value=mock_ipython),
            mock.patch("os.path.isfile", return_value=True),
            mock.patch("os.path.abspath", side_effect=lambda x: x),
        ):
            result = _get_jupyter_notebook_path()
            assert result == test_notebook_path
    finally:
        if old_main:
            sys.modules["__main__"] = old_main


def test_get_jupyter_notebook_path_single_notebook_in_directory(tmp_path):
    """Test fallback method when there's exactly one notebook in the current directory."""
    # Create a single notebook file
    notebook_file = tmp_path / "test_notebook.ipynb"
    notebook_file.write_text("{}")

    mock_ipython = mock.MagicMock()
    mock_ipython.kernel = mock.MagicMock()
    mock_ipython.config = {}

    with (
        mock.patch("IPython.get_ipython", return_value=mock_ipython),
        mock.patch("pathlib.Path.cwd", return_value=tmp_path),
        mock.patch("os.path.isfile", return_value=False),  # VS Code method fails
    ):
        result = _get_jupyter_notebook_path()
        assert result == str(notebook_file.absolute())


def test_get_jupyter_notebook_path_multiple_notebooks_in_directory(tmp_path):
    """Test fallback method when there are multiple notebooks - should not guess."""
    # Create multiple notebook files
    (tmp_path / "notebook1.ipynb").write_text("{}")
    (tmp_path / "notebook2.ipynb").write_text("{}")

    mock_ipython = mock.MagicMock()
    mock_ipython.kernel = mock.MagicMock()
    mock_ipython.config = {}

    with (
        mock.patch("IPython.get_ipython", return_value=mock_ipython),
        mock.patch("pathlib.Path.cwd", return_value=tmp_path),
        mock.patch("os.path.isfile", return_value=False),  # VS Code method fails
    ):
        result = _get_jupyter_notebook_path()
        # Should return None when we can't determine which notebook
        assert result is None


def test_get_jupyter_notebook_path_ignores_checkpoint_files(tmp_path):
    # Create a notebook and a checkpoint file
    notebook_file = tmp_path / "test_notebook.ipynb"
    notebook_file.write_text("{}")

    checkpoint_dir = tmp_path / ".ipynb_checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "test_notebook-checkpoint.ipynb").write_text("{}")

    mock_ipython = mock.MagicMock()
    mock_ipython.kernel = mock.MagicMock()
    mock_ipython.config = {}

    with (
        mock.patch("IPython.get_ipython", return_value=mock_ipython),
        mock.patch("pathlib.Path.cwd", return_value=tmp_path),
        mock.patch("os.path.isfile", return_value=False),  # VS Code method fails
    ):
        result = _get_jupyter_notebook_path()
        # Should find the non-checkpoint notebook
        assert result == str(notebook_file.absolute())


def test_jupyter_notebook_run_context_in_context():
    """Test JupyterNotebookRunContext.in_context() method."""
    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context._is_in_jupyter_notebook"
    ) as mock_is_in_jupyter:
        mock_is_in_jupyter.return_value = True
        assert JupyterNotebookRunContext().in_context() is True

        mock_is_in_jupyter.return_value = False
        assert JupyterNotebookRunContext().in_context() is False


def test_jupyter_notebook_run_context_tags_with_notebook_path():
    """Test tags generation when notebook path is available."""
    test_notebook_path = "/path/to/notebook.ipynb"

    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context._get_jupyter_notebook_path",
        return_value=test_notebook_path,
    ):
        tags = JupyterNotebookRunContext().tags()
        assert tags == {
            MLFLOW_SOURCE_NAME: test_notebook_path,
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
        }


def test_jupyter_notebook_run_context_tags_without_notebook_path():
    """Test tags generation when notebook path cannot be determined."""
    with mock.patch(
        "mlflow.tracking.context.jupyter_notebook_context._get_jupyter_notebook_path",
        return_value=None,
    ):
        tags = JupyterNotebookRunContext().tags()
        # Should return empty dict to allow other providers to set the source
        assert tags == {}


def test_get_jupyter_notebook_path_with_jpy_session_name(tmp_path, monkeypatch):
    """Test getting notebook path from JPY_SESSION_NAME environment variable."""
    notebook_name = "my_notebook.ipynb"
    notebook_file = tmp_path / notebook_name
    notebook_file.write_text("{}")

    mock_ipython = mock.MagicMock()
    mock_ipython.kernel = mock.MagicMock()
    mock_ipython.config = {}

    monkeypatch.setenv("JPY_SESSION_NAME", notebook_name)

    with (
        mock.patch("IPython.get_ipython", return_value=mock_ipython),
        mock.patch("os.getcwd", return_value=str(tmp_path)),
        mock.patch("os.path.isfile", side_effect=lambda p: p == str(notebook_file)),
        mock.patch("os.path.abspath", side_effect=lambda x: x),
    ):
        result = _get_jupyter_notebook_path()
        assert result == str(notebook_file)
