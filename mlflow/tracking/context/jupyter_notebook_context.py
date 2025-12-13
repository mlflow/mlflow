import os

from mlflow.entities import SourceType
from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils.databricks_utils import is_running_in_ipython_environment
from mlflow.utils.mlflow_tags import MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE


def _get_notebook_name():
    """
    Attempt to get the current Jupyter notebook name using multiple methods.

    Returns:
        The notebook filename if found, None otherwise.
    """
    # Method 1: Check for VS Code notebook environment variable
    # VS Code sets this when running a notebook
    vsc_notebook = os.environ.get("__vsc_ipynb_file__")
    if vsc_notebook:
        return os.path.basename(vsc_notebook)

    # Method 2: Check for Jupyter's IPYNB_FILE environment variable
    ipynb_file = os.environ.get("IPYNB_FILE")
    if ipynb_file:
        return os.path.basename(ipynb_file)

    # Method 3: Try to get notebook name from ipynbname package (if available)
    # This package works reliably in many Jupyter environments
    try:
        import ipynbname

        return str(ipynbname.name()) + ".ipynb"
    except Exception:
        pass

    return None


def _is_in_jupyter_notebook():
    """
    Check if we're running inside a Jupyter notebook (not just IPython).

    Returns:
        True if we're in a Jupyter notebook environment, False otherwise.
    """
    if not is_running_in_ipython_environment():
        return False

    try:
        from IPython import get_ipython

        ip = get_ipython()

        # Check the class name to identify Jupyter vs plain IPython
        # Jupyter notebooks use ZMQInteractiveShell
        shell_class = ip.__class__.__name__
        if shell_class == "ZMQInteractiveShell":
            return True

        # Additional check: see if we have kernel attribute (indicates notebook kernel)
        if hasattr(ip, "kernel"):
            return True

        return False
    except (ImportError, ModuleNotFoundError):
        return False


class JupyterNotebookRunContext(RunContextProvider):
    """
    Context provider for local Jupyter notebooks.

    This provider sets the source name to the notebook filename and source type
    to NOTEBOOK when running inside a Jupyter notebook environment.
    """

    def in_context(self):
        return _is_in_jupyter_notebook()

    def tags(self):
        notebook_name = _get_notebook_name()
        tags = {
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
        }
        if notebook_name:
            tags[MLFLOW_SOURCE_NAME] = notebook_name
        return tags

