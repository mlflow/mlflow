import os
import sys
from functools import partial

from mlflow.store.tracking import DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.rest_store import RestStore, DatabricksRestStore
from mlflow.tracking._tracking_service.registry import TrackingStoreRegistry
from mlflow.utils import env, rest_utils
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.databricks_utils import get_databricks_host_creds

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

_tracking_uri = None


def is_tracking_uri_set():
    """Returns True if the tracking URI has been set, False otherwise."""
    if _tracking_uri or env.get_env(_TRACKING_URI_ENV_VAR):
        return True
    return False


def set_tracking_uri(uri: str) -> None:
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

    .. code-block:: python
        :caption: Example

        import mlflow

        mlflow.set_tracking_uri("file:///tmp/my_tracking")
        tracking_uri = mlflow.get_tracking_uri()
        print("Current tracking uri: {}".format(tracking_uri))

    .. code-block:: text
        :caption: Output

        Current tracking uri: file:///tmp/my_tracking
    """
    global _tracking_uri
    _tracking_uri = uri


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
        ignore_tls_verification=os.environ.get(_TRACKING_INSECURE_TLS_ENV_VAR) == "true",
        client_cert_path=os.environ.get(_TRACKING_CLIENT_CERT_PATH_ENV_VAR),
        server_cert_path=os.environ.get(_TRACKING_SERVER_CERT_PATH_ENV_VAR),
    )


def _get_rest_store(store_uri, **_):
    return RestStore(partial(_get_default_host_creds, store_uri))


def _get_databricks_rest_store(store_uri, **_):
    return DatabricksRestStore(partial(get_databricks_host_creds, store_uri))


_tracking_store_registry = TrackingStoreRegistry()
_tracking_store_registry.register("", _get_file_store)
_tracking_store_registry.register("file", _get_file_store)
_tracking_store_registry.register("databricks", _get_databricks_rest_store)

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
        print(
            "Notice: failed to import Git (the git executable is probably not on your PATH),"
            " so Git SHA is not available. Error: %s" % e,
            file=sys.stderr,
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
