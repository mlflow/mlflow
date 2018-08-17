from __future__ import print_function

import os
import sys

from six.moves import urllib

from mlflow.store.file_store import FileStore
from mlflow.store.rest_store import RestStore, DatabricksStore
from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.utils import env, rest_utils


_TRACKING_URI_ENV_VAR = "MLFLOW_TRACKING_URI"
_LOCAL_FS_URI_PREFIX = "file:///"
_REMOTE_URI_PREFIX = "http://"

_tracking_uri = None


def set_tracking_uri(uri):
    """
    Set the tracking server URI to the passed-in value. This does not affect the
    currently active run (if one exists), but takes effect for any successive runs.

    The provided URI can be one of three types:

    - An empty string, or a local file path, prefixed with ``file:/``. Data is stored
      locally at the provided file (or ``./mlruns`` if empty).
    - An HTTP URI like ``https://my-tracking-server:5000``.
    - A Databricks workspace, provided as just the string 'databricks' or, to use a specific
      Databricks profile (per the Databricks CLI), 'databricks://profileName'.
    """
    global _tracking_uri
    _tracking_uri = uri


def get_tracking_uri():
    """
    Return the current tracking URI. This may not correspond to the tracking URI of
    the currently active run, since the tracking URI can be updated via ``set_tracking_uri``.

    :return: the tracking URI.
    """
    global _tracking_uri
    if _tracking_uri is not None:
        return _tracking_uri
    elif env.get_env(_TRACKING_URI_ENV_VAR) is not None:
        return env.get_env(_TRACKING_URI_ENV_VAR)
    else:
        return os.path.abspath("./mlruns")


def _get_store(store_uri=None):
    store_uri = store_uri if store_uri else get_tracking_uri()
    # Default: if URI hasn't been set, return a FileStore
    if store_uri is None:
        return FileStore()
    # Pattern-match on the URI
    if _is_databricks_uri(store_uri):
        return _get_databricks_rest_store(store_uri)
    if _is_local_uri(store_uri):
        return _get_file_store(store_uri)
    if _is_http_uri(store_uri):
        return _get_rest_store(store_uri)

    raise Exception("Tracking URI must be a local filesystem URI of the form '%s...' or a "
                    "remote URI of the form '%s...'. Update the tracking URI via "
                    "mlflow.set_tracking_uri" % (_LOCAL_FS_URI_PREFIX, _REMOTE_URI_PREFIX))


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


def _get_file_store(store_uri):
    path = urllib.parse.urlparse(store_uri).path
    return FileStore(path)


def _get_rest_store(store_uri):
    return RestStore({'hostname': store_uri})


def _get_databricks_rest_store(store_uri):
    parsed_uri = urllib.parse.urlparse(store_uri)

    profile = None
    if parsed_uri.scheme == 'databricks':
        profile = parsed_uri.hostname
    http_request_kwargs = rest_utils.get_databricks_http_request_kwargs_or_fail(profile)
    return DatabricksStore(http_request_kwargs)


def _get_model_log_dir(model_name, run_id):
    if not run_id:
        raise Exception("Must specify a run_id to get logging directory for a model.")
    store = _get_store()
    run = store.get_run(run_id)
    artifact_repo = ArtifactRepository.from_artifact_uri(run.info.artifact_uri, store)
    return artifact_repo.download_artifacts(model_name)


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
