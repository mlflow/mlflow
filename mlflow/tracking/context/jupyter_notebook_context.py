"""
Context provider for Jupyter Notebook environments.

This module provides functionality to track git metadata when running MLflow
experiments from Jupyter Notebooks (.ipynb files).
"""

import logging
import os
import re
from pathlib import Path
from urllib.parse import urljoin

from mlflow.entities import SourceType
from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils.mlflow_tags import MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE

_logger = logging.getLogger(__name__)


def _is_in_jupyter_notebook():
    """
    Check if the code is running in a Jupyter notebook environment.

    Returns:
        bool: True if running in Jupyter notebook, False otherwise.
    """
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            return False

        # Check if we're in a Jupyter/IPython kernel
        # The kernel name contains 'kernel' for Jupyter notebooks
        return hasattr(ipython, "kernel")
    except (ImportError, AttributeError):
        return False


def _get_jupyter_notebook_path():
    """
    Attempt to get the path to the currently running Jupyter notebook.

    Returns:
        str | None: The absolute path to the notebook file, or None if it cannot be determined.
    """
    if not _is_in_jupyter_notebook():
        return None

    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            return None

        # Method 0: Check for VS Code Jupyter __vsc_ipynb_file__ FIRST
        try:
            import __main__

            if hasattr(__main__, "__vsc_ipynb_file__"):
                notebook_path = __main__.__vsc_ipynb_file__
                if os.path.isfile(notebook_path):
                    return os.path.abspath(notebook_path)
        except Exception:
            pass

        # Method 1: Try to get from IPython's connection file
        # This works when running from a Jupyter notebook server
        try:
            import requests
            from notebook import notebookapp

            # Get the kernel ID from the connection file
            connection_file = ipython.config.get("IPKernelApp", {}).get("connection_file", "")
            if not connection_file:
                # No connection file available, skip this method
                raise ValueError("No connection file")

            kernel_id = re.search(r"kernel-(.*).json", connection_file)
            if not kernel_id:
                return None

            kernel_id = kernel_id.group(1)

            # Query the Jupyter notebook server for the notebook path
            for server in notebookapp.list_running_servers():
                try:
                    # Try to get the list of sessions
                    url = urljoin(server["url"], "api/sessions")
                    response = requests.get(url, params={"token": server.get("token", "")})
                    if response.status_code == 200:
                        sessions = response.json()
                        for session in sessions:
                            if session["kernel"]["id"] == kernel_id:
                                notebook_path = session["notebook"]["path"]
                                # Combine with server root directory
                                notebook_dir = server.get("notebook_dir", "")
                                full_path = os.path.join(notebook_dir, notebook_path)
                                if os.path.isfile(full_path):
                                    return os.path.abspath(full_path)
                except Exception as e:
                    msg = f"Failed to get notebook path from server {server['url']}: {e}"
                    _logger.debug(msg)
                    continue
        except Exception as e:
            _logger.debug(f"Failed to determine Jupyter notebook path: {e}")

        # Method 2: Try JupyterLab method
        try:
            import ipykernel
            from notebook.notebookapp import list_running_servers

            # Get kernel ID
            connection_file = ipykernel.get_connection_file()
            match = re.search(r"kernel-([\w\-]+)\.json", connection_file)
            kernel_id = match.group(1)

            # Query all running servers
            for server in list_running_servers():
                url = urljoin(server["url"], "api/sessions")
                response = requests.get(url, params={"token": server.get("token", "")})
                for session in response.json():
                    if session["kernel"]["id"] == kernel_id:
                        relative_path = session["notebook"]["path"]
                        return os.path.join(server["notebook_dir"], relative_path)
        except Exception as e:
            _logger.debug(f"JupyterLab method failed: {e}")

        # Method 3: Check common environment variables
        try:
            # Some Jupyter environments set this
            if "JPY_SESSION_NAME" in os.environ:
                session_name = os.environ["JPY_SESSION_NAME"]
                # Try to find the file in the current directory
                if session_name.endswith(".ipynb"):
                    if os.path.isfile(session_name):
                        return os.path.abspath(session_name)
                    # Try in current working directory
                    cwd_path = os.path.join(os.getcwd(), session_name)
                    if os.path.isfile(cwd_path):
                        return os.path.abspath(cwd_path)
        except Exception as e:
            _logger.debug(f"Environment variable method failed: {e}")

        # Method 4: Last resort - search for .ipynb files in current directory
        # This is less reliable but better than returning ipykernel_launcher.py
        try:
            cwd = Path.cwd()
            ipynb_files = list(cwd.glob("*.ipynb"))
            # Filter out checkpoint files
            ipynb_files = [f for f in ipynb_files if ".ipynb_checkpoints" not in str(f)]

            if len(ipynb_files) == 1:
                # If there's exactly one notebook in the current directory, use it
                return str(ipynb_files[0].absolute())
        except Exception as e:
            _logger.debug(f"Directory search method failed: {e}")

    except Exception as e:
        _logger.debug(f"Overall notebook path detection failed: {e}")

    return None


class JupyterNotebookRunContext(RunContextProvider):
    """
    Run context provider for Jupyter Notebook environments.

    This provider detects when MLflow is running in a Jupyter notebook
    and attempts to determine the actual notebook file path for git tracking.
    """

    def in_context(self):
        """Check if we're in a Jupyter notebook environment."""
        return _is_in_jupyter_notebook()

    def tags(self):
        """
        Generate tags for Jupyter notebook context.

        Returns:
            dict: Tags including the notebook source name and type.
        """
        notebook_path = _get_jupyter_notebook_path()

        if notebook_path is not None:
            return {
                MLFLOW_SOURCE_NAME: notebook_path,
                MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
            }

        # If we can't determine the notebook path, return empty dict
        # to allow other context providers to set the source
        return {}
