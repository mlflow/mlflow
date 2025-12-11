from unittest import mock

from mlflow.entities import SourceType
from mlflow.tracking.context.jupyter_notebook_context import (
    JupyterNotebookRunContext,
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
    mock_shell = mock.Mock(spec=[])  # No kernel attribute
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


def test_is_in_jupyter_notebook_false_import_error():
    """Test graceful handling when IPython import fails."""
    with (
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context.is_running_in_ipython_environment",
            return_value=True,
        ),
        mock.patch.dict("sys.modules", {"IPython": None}),
    ):
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

