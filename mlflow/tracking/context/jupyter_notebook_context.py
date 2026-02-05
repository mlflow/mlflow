import json
import os
from collections.abc import Generator
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from mlflow.entities import SourceType
from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils.databricks_utils import is_running_in_ipython_environment
from mlflow.utils.mlflow_tags import MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE


@lru_cache(maxsize=1)
def _get_notebook_name() -> str | None:
    """
    Attempt to get the current Jupyter notebook name using multiple methods.

    Returns:
        The notebook filename if found, None otherwise.
    """
    # Method 1: Check VS Code notebook path in IPython user namespace
    # VS Code's Jupyter extension stores the notebook path in ip.user_ns
    if notebook_path := _get_vscode_notebook_path():
        return os.path.basename(notebook_path)

    # Method 2: Check environment variables
    if vsc_notebook := os.environ.get("__vsc_ipynb_file__"):
        return os.path.basename(vsc_notebook)

    if ipynb_file := os.environ.get("IPYNB_FILE"):
        return os.path.basename(ipynb_file)

    # Method 3: Query Jupyter server sessions (for JupyterLab/classic Jupyter)
    if notebook_path := _get_notebook_path_from_sessions():
        return os.path.basename(notebook_path)

    return None


def _get_vscode_notebook_path() -> str | None:
    """
    Get notebook path from VS Code's IPython user namespace.

    VS Code's Jupyter extension stores the notebook file path in the
    IPython user namespace under '__vsc_ipynb_file__'.

    Returns:
        The notebook path if found, None otherwise.
    """
    if not is_running_in_ipython_environment():
        return None

    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip and "__vsc_ipynb_file__" in ip.user_ns:
            return ip.user_ns["__vsc_ipynb_file__"]
    except Exception:
        pass

    return None


def _get_notebook_path_from_sessions() -> str | None:
    """
    Get notebook path by querying Jupyter server sessions.

    This queries running Jupyter servers to find the notebook
    associated with the current kernel.

    Returns:
        The notebook path if found, None otherwise.
    """
    try:
        kernel_id = _get_kernel_id()
        if not kernel_id:
            return None

        for server in _get_running_servers():
            try:
                if notebook_path := _get_sessions_notebook(server, kernel_id):
                    return notebook_path
            except Exception:
                continue
    except Exception:
        pass

    return None


def _get_kernel_id() -> str | None:
    """
    Get the current kernel ID from the connection file.

    Returns:
        The kernel ID string, or None if not found.
    """
    try:
        import ipykernel

        connection_file = Path(ipykernel.get_connection_file()).stem
        # Connection file is like: kernel-<uuid>
        return connection_file.split("-", 1)[1]
    except Exception:
        pass

    return None


def _get_running_servers() -> Generator[dict[str, Any], None, None]:
    """
    Get list of running Jupyter servers by scanning the runtime directory.

    Yields:
        Server info dictionaries with 'url' and 'token' keys.
    """
    try:
        from jupyter_core.paths import jupyter_runtime_dir

        runtime_dir = Path(jupyter_runtime_dir())
        if not runtime_dir.is_dir():
            return

        # Get server files, sorted by modification time (most recent first)
        server_files = sorted(
            list(runtime_dir.glob("nbserver-*.json"))  # jupyter notebook (or lab 2)
            + list(runtime_dir.glob("jpserver-*.json")),  # jupyterlab 3
            key=os.path.getmtime,
            reverse=True,
        )

        for server_file in server_files:
            try:
                with open(server_file) as f:
                    yield json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
    except ImportError:
        pass


def _get_sessions_notebook(server: dict[str, Any], kernel_id: str) -> str | None:
    """
    Query a server's sessions API to find the notebook for a kernel.

    Args:
        server: Server info dict with 'url' and optionally 'token'.
        kernel_id: The kernel ID to search for.

    Returns:
        The notebook path if found, None otherwise.
    """
    url = server.get("url", "").rstrip("/")
    token = server.get("token") or os.getenv("JUPYTERHUB_API_TOKEN", "")

    sessions_url = f"{url}/api/sessions"
    if token:
        sessions_url += f"?token={token}"

    try:
        with urlopen(sessions_url, timeout=0.5) as response:
            sessions = json.load(response)

        for session in sessions:
            if session.get("kernel", {}).get("id") == kernel_id:
                return session.get("path")
    except Exception:
        pass

    return None


def _is_in_jupyter_notebook() -> bool:
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

        # Jupyter notebooks use ZMQInteractiveShell
        shell_class = ip.__class__.__name__
        if shell_class == "ZMQInteractiveShell":
            return True

        # Also check for kernel attribute
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
