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

    # Method 4: Try to get notebook info from IPython kernel
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return None

        # Check if we're in a ZMQ-based kernel (Jupyter)
        kernel = getattr(ip, "kernel", None)
        if kernel is None:
            return None

        # Try to get connection file and parse session info
        connection_file = getattr(kernel, "connection_file", None)
        if connection_file:
            # Connection file is like kernel-<id>.json
            # We can use this to find the notebook via Jupyter server API
            notebook_name = _get_notebook_from_jupyter_server(connection_file)
            if notebook_name:
                return notebook_name
    except Exception:
        pass

    # Method 5: Check IPython namespace for clues
    notebook_name = _get_notebook_from_ipython_history()
    if notebook_name:
        return notebook_name

    return None


def _get_notebook_from_ipython_history():
    """
    Try to infer notebook name from IPython's internal state.

    Returns:
        The notebook filename if found, None otherwise.
    """
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return None

        # Check if there's a __file__ variable set (some environments set this)
        user_ns = ip.user_ns
        if "__file__" in user_ns:
            f = user_ns["__file__"]
            if f and f.endswith(".ipynb"):
                return os.path.basename(f)

        # Check __session__ if available
        if "__session__" in user_ns:
            session = user_ns["__session__"]
            if session and session.endswith(".ipynb"):
                return os.path.basename(session)

    except Exception:
        pass

    return None


def _get_notebook_from_jupyter_server(connection_file):
    """
    Try to get notebook name by querying the Jupyter server API.

    Args:
        connection_file: Path to the kernel connection file.

    Returns:
        The notebook filename if found, None otherwise.
    """
    try:
        import json
        import urllib.request

        # Extract kernel ID from connection file name
        # Connection file looks like: /path/to/kernel-<uuid>.json
        kernel_id = os.path.basename(connection_file).replace("kernel-", "").replace(".json", "")

        # Try to find Jupyter server URL and token from environment or runtime dir
        servers = _list_running_jupyter_servers()
        for server in servers:
            try:
                url = f"{server['url']}api/sessions"
                token = server.get("token", "")
                if token:
                    url += f"?token={token}"

                req = urllib.request.Request(url)
                req.add_header("Authorization", f"token {token}")
                with urllib.request.urlopen(req, timeout=1) as response:
                    sessions = json.loads(response.read().decode())
                    for session in sessions:
                        if session.get("kernel", {}).get("id") == kernel_id:
                            notebook_path = session.get("notebook", {}).get(
                                "path"
                            ) or session.get("path")
                            if notebook_path:
                                return os.path.basename(notebook_path)
            except Exception:
                continue
    except Exception:
        pass

    return None


def _list_running_jupyter_servers():
    """List running Jupyter servers."""
    try:
        from notebook.notebookapp import list_running_servers

        return list(list_running_servers())
    except ImportError:
        pass

    try:
        from jupyter_server.serverapp import list_running_servers

        return list(list_running_servers())
    except ImportError:
        pass

    return []


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

