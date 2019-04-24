from __future__ import print_function

import os
import sys
import warnings

import entrypoints
from six.moves import urllib

from mlflow.exceptions import MlflowException
from mlflow.store import DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.store.dbmodels.db_types import DATABASE_ENGINES
from mlflow.store.file_store import FileStore
from mlflow.store.rest_store import RestStore
from mlflow.utils import env, rest_utils
from mlflow.utils.databricks_utils import get_databricks_host_creds


_TRACKING_URI_ENV_VAR = "MLFLOW_TRACKING_URI"
_LOCAL_FS_URI_PREFIX = "file:///"
_REMOTE_URI_PREFIX = "http://"

# Extra environment variables which take precedence for setting the basic/bearer
# auth on http requests.
_TRACKING_USERNAME_ENV_VAR = "MLFLOW_TRACKING_USERNAME"
_TRACKING_PASSWORD_ENV_VAR = "MLFLOW_TRACKING_PASSWORD"
_TRACKING_TOKEN_ENV_VAR = "MLFLOW_TRACKING_TOKEN"
_TRACKING_INSECURE_TLS_ENV_VAR = "MLFLOW_TRACKING_INSECURE_TLS"

_tracking_uri = None


def is_tracking_uri_set():
    """Returns True if the tracking URI has been set, False otherwise."""
    if _tracking_uri or env.get_env(_TRACKING_URI_ENV_VAR):
        return True
    return False


def set_tracking_uri(uri):
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
    """
    global _tracking_uri
    _tracking_uri = uri


def get_tracking_uri():
    """
    Get the current tracking URI. This may not correspond to the tracking URI of
    the currently active run, since the tracking URI can be updated via ``set_tracking_uri``.

    :return: The tracking URI.
    """
    global _tracking_uri
    if _tracking_uri is not None:
        return _tracking_uri
    elif env.get_env(_TRACKING_URI_ENV_VAR) is not None:
        return env.get_env(_TRACKING_URI_ENV_VAR)
    else:
        return os.path.abspath(DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH)


def _is_local_uri(uri):
    scheme = urllib.parse.urlparse(uri).scheme
    return uri != 'databricks' and (scheme == '' or scheme == 'file')


def _is_http_uri(uri):
    scheme = urllib.parse.urlparse(uri).scheme
    return scheme == 'http' or scheme == 'https'


def _is_databricks_uri(uri):
    """Databricks URIs look like 'databricks' (default profile) or 'databricks://profile'"""
    scheme = urllib.parse.urlparse(uri).scheme
    return scheme == 'databricks' or uri == 'databricks'


def _get_file_store(store_uri, **_):
    path = urllib.parse.urlparse(store_uri).path if store_uri else None
    return FileStore(path, path)


def _is_database_uri(uri):
    if urllib.parse.urlparse(uri).scheme not in DATABASE_ENGINES:
        return False
    return True


def _get_sqlalchemy_store(store_uri, artifact_uri):
    from mlflow.store.sqlalchemy_store import SqlAlchemyStore
    if artifact_uri is None:
        artifact_uri = DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
    return SqlAlchemyStore(store_uri, artifact_uri)


def _get_rest_store(store_uri, **_):
    def get_default_host_creds():
        return rest_utils.MlflowHostCreds(
            host=store_uri,
            username=os.environ.get(_TRACKING_USERNAME_ENV_VAR),
            password=os.environ.get(_TRACKING_PASSWORD_ENV_VAR),
            token=os.environ.get(_TRACKING_TOKEN_ENV_VAR),
            ignore_tls_verification=os.environ.get(_TRACKING_INSECURE_TLS_ENV_VAR) == 'true',
        )
    return RestStore(get_default_host_creds)


def get_db_profile_from_uri(uri):
    """
    Get the Databricks profile specified by the tracking URI (if any), otherwise
    returns None.
    """
    parsed_uri = urllib.parse.urlparse(uri)
    if parsed_uri.scheme == "databricks":
        return parsed_uri.netloc
    return None


def _get_databricks_rest_store(store_uri, **_):
    profile = get_db_profile_from_uri(store_uri)
    return RestStore(lambda: get_databricks_host_creds(profile))


class TrackingStoreRegistry:
    """Scheme-based registry for tracking store implementations

    This class allows the registration of a function or class to provide an
    implementation for a given scheme of `store_uri` through the `register`
    methods. Implementations declared though the entrypoints
    `mlflow.tracking_store` group can be automatically registered through the
    `register_entrypoints` method.

    When instantiating a store through the `get_store` method, the scheme of
    the store URI provided (or inferred from environment) will be used to
    select which implementation to instantiate, which will be called with same
    arguments passed to the `get_store` method.
    """

    def __init__(self):
        self._registry = {}

    def register(self, scheme, store_builder):
        self._registry[scheme] = store_builder

    def register_entrypoints(self):
        """Register tracking stores provided by other packages"""
        for entrypoint in entrypoints.get_group_all("mlflow.tracking_store"):
            try:
                self.register(entrypoint.name, entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    'Failure attempting to register tracking store for scheme "{}": {}'.format(
                        entrypoint.name, str(exc)
                    ),
                    stacklevel=2
                )

    def get_store(self, store_uri=None, artifact_uri=None):
        """Get a store from the registry based on the scheme of store_uri

        :param store_uri: The store URI. If None, it will be inferred from the environment. This URI
                          is used to select which tracking store implementation to instantiate and
                          is passed to the constructor of the implementation.
        :param artifact_uri: Artifact repository URI. Passed through to the tracking store
                             implementation.

        :return: An instance of `mlflow.store.AbstractStore` that fulfills the store URI
                 requirements.
        """
        store_uri = store_uri if store_uri is not None else get_tracking_uri()

        if store_uri == 'databricks':
            # Add colon so databricks is parsed as scheme
            store_uri += ':'

        scheme = urllib.parse.urlparse(store_uri).scheme
        try:
            store_builder = self._registry[scheme]
        except KeyError:
            raise MlflowException(
                "Could not find a registered tracking store for: {}. "
                "Currently registered schemes are: {}".format(
                    store_uri, list(self._registry.keys())
                )
            )
        return store_builder(store_uri=store_uri, artifact_uri=artifact_uri)


_tracking_store_registry = TrackingStoreRegistry()
_tracking_store_registry.register('', _get_file_store)
_tracking_store_registry.register('file', _get_file_store)
_tracking_store_registry.register('databricks', _get_databricks_rest_store)

for scheme in ['http', 'https']:
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
    if '#' in uri:
        # Already a URI in git repo format
        return uri
    try:
        from git import Repo, InvalidGitRepositoryError, GitCommandNotFound, NoSuchPathError
    except ImportError as e:
        print("Notice: failed to import Git (the git executable is probably not on your PATH),"
              " so Git SHA is not available. Error: %s" % e, file=sys.stderr)
        return uri
    try:
        # Check whether this is part of a git repo
        repo = Repo(uri, search_parent_directories=True)

        # Repo url
        repo_url = "file://%s" % repo.working_tree_dir

        # Sub directory
        rlpath = uri.replace(repo.working_tree_dir, '')
        if (rlpath == ''):
            git_path = repo_url
        elif (rlpath[0] == '/'):
            git_path = repo_url + '#' + rlpath[1:]
        else:
            git_path = repo_url + '#' + rlpath
        return git_path
    except (InvalidGitRepositoryError, GitCommandNotFound, ValueError, NoSuchPathError):
        return uri
