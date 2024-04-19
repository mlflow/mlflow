import logging
import os
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Union

from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.tracking import DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking._tracking_service.registry import TrackingStoreRegistry
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.uri import _DATABRICKS_UNITY_CATALOG_SCHEME

_logger = logging.getLogger(__name__)
_tracking_uri = None


def is_tracking_uri_set():
    """Returns True if the tracking URI has been set, False otherwise."""
    if _tracking_uri or MLFLOW_TRACKING_URI.get():
        return True
    return False


def set_tracking_uri(uri: Union[str, Path]) -> None:
    """
    Set the tracking server URI. This does not affect the
    currently active run (if one exists), but takes effect for successive runs.

    Args:
        uri:

            - An empty string, or a local file path, prefixed with ``file:/``. Data is stored
              locally at the provided file (or ``./mlruns`` if empty).
            - An HTTP URI like ``https://my-tracking-server:5000``.
            - A Databricks workspace, provided as the string "databricks" or, to use a Databricks
              CLI `profile <https://github.com/databricks/databricks-cli#installation>`_,
              "databricks://<profileName>".
            - A :py:class:`pathlib.Path` instance

    .. code-block:: python
        :test:
        :caption: Example

        import mlflow

        mlflow.set_tracking_uri("file:///tmp/my_tracking")
        tracking_uri = mlflow.get_tracking_uri()
        print(f"Current tracking uri: {tracking_uri}")

    .. code-block:: text
        :caption: Output

        Current tracking uri: file:///tmp/my_tracking
    """
    if isinstance(uri, Path):
        # On Windows with Python3.8 (https://bugs.python.org/issue38671)
        # .resolve() doesn't return the absolute path if the directory doesn't exist
        # so we're calling .absolute() first to get the absolute path on Windows,
        # then .resolve() to clean the path
        uri = uri.absolute().resolve().as_uri()
    global _tracking_uri
    _tracking_uri = uri


@contextmanager
def _use_tracking_uri(uri: str) -> None:
    """Temporarily use the specified tracking URI.

    Args:
        uri: The tracking URI to use.

    """
    global _tracking_uri
    old_tracking_uri = _tracking_uri
    try:
        _tracking_uri = uri
        yield
    finally:
        _tracking_uri = old_tracking_uri


def _resolve_tracking_uri(tracking_uri=None):
    return tracking_uri or get_tracking_uri()


def get_tracking_uri() -> str:
    """Get the current tracking URI. This may not correspond to the tracking URI of
    the currently active run, since the tracking URI can be updated via ``set_tracking_uri``.

    Returns:
        The tracking URI.

    .. code-block:: python

        import mlflow

        # Get the current tracking uri
        tracking_uri = mlflow.get_tracking_uri()
        print(f"Current tracking uri: {tracking_uri}")

    .. code-block:: text

        Current tracking uri: file:///.../mlruns
    """
    global _tracking_uri
    if _tracking_uri is not None:
        return _tracking_uri
    elif uri := MLFLOW_TRACKING_URI.get():
        return uri
    else:
        return path_to_local_file_uri(os.path.abspath(DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH))


def _get_file_store(store_uri, **_):
    return FileStore(store_uri, store_uri)


def _get_sqlalchemy_store(store_uri, artifact_uri):
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

    if artifact_uri is None:
        artifact_uri = DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
    return SqlAlchemyStore(store_uri, artifact_uri)


def _get_rest_store(store_uri, **_):
    return RestStore(partial(get_default_host_creds, store_uri))


def _get_databricks_rest_store(store_uri, **_):
    return RestStore(partial(get_databricks_host_creds, store_uri))


def _get_databricks_uc_rest_store(store_uri, **_):
    from mlflow.exceptions import MlflowException
    from mlflow.version import VERSION

    global _tracking_store_registry
    supported_schemes = [
        scheme
        for scheme in _tracking_store_registry._registry
        if scheme != _DATABRICKS_UNITY_CATALOG_SCHEME
    ]
    raise MlflowException(
        f"Detected Unity Catalog tracking URI '{store_uri}'. "
        "Setting the tracking URI to a Unity Catalog backend is not supported in the current "
        f"version of the MLflow client ({VERSION}). "
        "Please specify a different tracking URI via mlflow.set_tracking_uri, with "
        "one of the supported schemes: "
        f"{supported_schemes}. If you're trying to access models in the Unity "
        "Catalog, please upgrade to the latest version of the MLflow Python "
        "client, then specify a Unity Catalog model registry URI via "
        f"mlflow.set_registry_uri('{_DATABRICKS_UNITY_CATALOG_SCHEME}') or "
        f"mlflow.set_registry_uri('{_DATABRICKS_UNITY_CATALOG_SCHEME}://profile_name'), where "
        "'profile_name' is the name of the Databricks CLI profile to use for "
        "authentication. Be sure to leave the tracking URI configured to use "
        "one of the supported schemes listed above."
    )


_tracking_store_registry = TrackingStoreRegistry()


def _register_tracking_stores():
    global _tracking_store_registry
    _tracking_store_registry.register("", _get_file_store)
    _tracking_store_registry.register("file", _get_file_store)
    _tracking_store_registry.register("databricks", _get_databricks_rest_store)
    _tracking_store_registry.register(
        _DATABRICKS_UNITY_CATALOG_SCHEME, _get_databricks_uc_rest_store
    )

    for scheme in ["http", "https"]:
        _tracking_store_registry.register(scheme, _get_rest_store)

    for scheme in DATABASE_ENGINES:
        _tracking_store_registry.register(scheme, _get_sqlalchemy_store)

    _tracking_store_registry.register_entrypoints()


def _register(scheme, builder):
    _tracking_store_registry.register(scheme, builder)


_register_tracking_stores()


def _get_store(store_uri=None, artifact_uri=None):
    return _tracking_store_registry.get_store(store_uri, artifact_uri)


_artifact_repos_cache = OrderedDict()


def _get_artifact_repo(run_id):
    return _artifact_repos_cache.get(run_id)


# TODO(sueann): move to a projects utils module
def _get_git_url_if_present(uri):
    """Return the path git_uri#sub_directory if the URI passed is a local path that's part of
    a Git repo, or returns the original URI otherwise.

    Args:
        uri: The expanded uri.

    Returns:
        The git_uri#sub_directory if the uri is part of a Git repo, otherwise return the original
        uri.
    """
    if "#" in uri:
        # Already a URI in git repo format
        return uri
    try:
        from git import GitCommandNotFound, InvalidGitRepositoryError, NoSuchPathError, Repo
    except ImportError as e:
        _logger.warning(
            "Failed to import Git (the git executable is probably not on your PATH),"
            " so Git SHA is not available. Error: %s",
            e,
        )
        return uri
    try:
        # Check whether this is part of a git repo
        repo = Repo(uri, search_parent_directories=True)

        # Repo url
        repo_url = f"file://{repo.working_tree_dir}"

        # Sub directory
        rlpath = uri.replace(repo.working_tree_dir, "")
        if rlpath == "":
            git_path = repo_url
        elif rlpath[0] == "/":
            git_path = repo_url + "#" + rlpath[1:]
        else:
            git_path = repo_url + "#" + rlpath
        return git_path
    except (InvalidGitRepositoryError, GitCommandNotFound, ValueError, NoSuchPathError):
        return uri
