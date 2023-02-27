import os
from functools import partial
import logging
from pathlib import Path
from typing import Union
from contextlib import contextmanager

from mlflow.environment_variables import MLFLOW_TRACKING_AWS_SIGV4
from mlflow.store.tracking import DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking._tracking_service.registry import TrackingStoreRegistry
from mlflow.utils import env, rest_utils
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.uri import _DATABRICKS_UNITY_CATALOG_SCHEME

_TRACKING_URI_ENV_VAR = "MLFLOW_TRACKING_URI"

# Extra environment variables which take precedence for setting the basic/bearer
# auth on http requests.
_TRACKING_USERNAME_ENV_VAR = "MLFLOW_TRACKING_USERNAME"
_TRACKING_PASSWORD_ENV_VAR = "MLFLOW_TRACKING_PASSWORD"
_TRACKING_TOKEN_ENV_VAR = "MLFLOW_TRACKING_TOKEN"

# sets verify param of 'requests.request' function
# see https://requests.readthedocs.io/en/master/api/
_TRACKING_INSECURE_TLS_ENV_VAR = "MLFLOW_TRACKING_INSECURE_TLS"
_TRACKING_SERVER_CERT_PATH_ENV_VAR = "MLFLOW_TRACKING_SERVER_CERT_PATH"

# sets cert param of 'requests.request' function
# see https://requests.readthedocs.io/en/master/api/
_TRACKING_CLIENT_CERT_PATH_ENV_VAR = "MLFLOW_TRACKING_CLIENT_CERT_PATH"

_logger = logging.getLogger(__name__)
_tracking_uri = None


def is_tracking_uri_set():
    """Returns True if the tracking URI has been set, False otherwise."""
    if _tracking_uri or env.get_env(_TRACKING_URI_ENV_VAR):
        return True
    return False


def set_tracking_uri(uri: Union[str, Path]) -> None:
    """
    Set the tracking server URI. This does not affect the
    currently active run (if one exists), but takes effect for successive runs.

    :param uri:

                - An empty string, or a local file path, prefixed with ``file:/``. Data is stored
                  locally at the provided file (or ``./mlruns`` if empty).
                - An HTTP URI like ``https://my-tracking-server:5000``.
                - A Databricks workspace, provided as the string "databricks" or, to use a
                  Databricks CLI
                  `profile <https://github.com/databricks/databricks-cli#installation>`_,
                  "databricks://<profileName>".
                - A :py:class:`pathlib.Path` instance

    .. test-code-block:: python
        :caption: Example

        import mlflow

        mlflow.set_tracking_uri("file:///tmp/my_tracking")
        tracking_uri = mlflow.get_tracking_uri()
        print("Current tracking uri: {}".format(tracking_uri))

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
def _use_tracking_uri(uri: str, local_store_root_path: str = None) -> None:
    """
    Similar to `mlflow.tracking.set_tracking_uri` function but return a context manager.
    :param uri: tracking URI to use.
    :param local_store_root_path: the local store root path for the tracking URI.
    """
    global _tracking_uri
    cwd = os.getcwd()
    old_tracking_uri = _tracking_uri
    try:
        if local_store_root_path is not None:
            os.chdir(local_store_root_path)
        _tracking_uri = uri
        yield
    finally:
        _tracking_uri = old_tracking_uri
        os.chdir(cwd)


def _resolve_tracking_uri(tracking_uri=None):
    return tracking_uri or get_tracking_uri()


def get_tracking_uri() -> str:
    """
    Get the current tracking URI. This may not correspond to the tracking URI of
    the currently active run, since the tracking URI can be updated via ``set_tracking_uri``.

    :return: The tracking URI.

    .. code-block:: python
        :caption: Example

        import mlflow

        # Get the current tracking uri
        tracking_uri = mlflow.get_tracking_uri()
        print("Current tracking uri: {}".format(tracking_uri))

    .. code-block:: text
        :caption: Output

        Current tracking uri: file:///.../mlruns
    """
    global _tracking_uri
    if _tracking_uri is not None:
        return _tracking_uri
    elif env.get_env(_TRACKING_URI_ENV_VAR) is not None:
        return env.get_env(_TRACKING_URI_ENV_VAR)
    else:
        return path_to_local_file_uri(os.path.abspath(DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH))


def _get_file_store(store_uri, **_):
    return FileStore(store_uri, store_uri)


def _get_sqlalchemy_store(store_uri, artifact_uri):
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

    if artifact_uri is None:
        artifact_uri = DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
    return SqlAlchemyStore(store_uri, artifact_uri)


def _get_default_host_creds(store_uri):
    return rest_utils.MlflowHostCreds(
        host=store_uri,
        username=os.environ.get(_TRACKING_USERNAME_ENV_VAR),
        password=os.environ.get(_TRACKING_PASSWORD_ENV_VAR),
        token=os.environ.get(_TRACKING_TOKEN_ENV_VAR),
        aws_sigv4=MLFLOW_TRACKING_AWS_SIGV4.get(),
        ignore_tls_verification=os.environ.get(_TRACKING_INSECURE_TLS_ENV_VAR) == "true",
        client_cert_path=os.environ.get(_TRACKING_CLIENT_CERT_PATH_ENV_VAR),
        server_cert_path=os.environ.get(_TRACKING_SERVER_CERT_PATH_ENV_VAR),
    )


def _get_rest_store(store_uri, **_):
    return RestStore(partial(_get_default_host_creds, store_uri))


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
_tracking_store_registry.register("", _get_file_store)
_tracking_store_registry.register("file", _get_file_store)
_tracking_store_registry.register("databricks", _get_databricks_rest_store)
_tracking_store_registry.register(_DATABRICKS_UNITY_CATALOG_SCHEME, _get_databricks_uc_rest_store)

for scheme in ["http", "https"]:
    _tracking_store_registry.register(scheme, _get_rest_store)

for scheme in DATABASE_ENGINES:
    _tracking_store_registry.register(scheme, _get_sqlalchemy_store)

_tracking_store_registry.register_entrypoints()


def _get_store(store_uri=None, artifact_uri=None):
    return _tracking_store_registry.get_store(store_uri, artifact_uri)


# TODO(sueann): move to a projects utils module
def _get_git_url_if_present(uri):
    """
    Return the path git_uri#sub_directory if the URI passed is a local path that's part of
    a Git repo, or returns the original URI otherwise.
    :param uri: The expanded uri
    :return: The git_uri#sub_directory if the uri is part of a Git repo,
             otherwise return the original uri
    """
    if "#" in uri:
        # Already a URI in git repo format
        return uri
    try:
        from git import Repo, InvalidGitRepositoryError, GitCommandNotFound, NoSuchPathError
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
        repo_url = "file://%s" % repo.working_tree_dir

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
