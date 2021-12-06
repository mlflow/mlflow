import logging
import os
import re
import tempfile
import urllib.parse


from distutils import dir_util

import mlflow.utils
from mlflow.utils import databricks_utils
from mlflow.entities import SourceType, Param
from mlflow.exceptions import ExecutionException
from mlflow.projects import _project_spec
from mlflow import tracking
from mlflow.tracking.context.git_context import _get_git_commit
from mlflow.tracking import fluent
from mlflow.tracking.context.default_context import _get_user
from mlflow.utils.mlflow_tags import (
    MLFLOW_USER,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_REPO_URL,
    MLFLOW_GIT_BRANCH,
    LEGACY_MLFLOW_GIT_REPO_URL,
    LEGACY_MLFLOW_GIT_BRANCH_NAME,
    MLFLOW_PROJECT_ENTRY_POINT,
    MLFLOW_PARENT_RUN_ID,
)
from mlflow.utils.rest_utils import augmented_raise_for_status

# TODO: this should be restricted to just Git repos and not S3 and stuff like that
_GIT_URI_REGEX = re.compile(r"^[^/]*:")
_FILE_URI_REGEX = re.compile(r"^file://.+")
_ZIP_URI_REGEX = re.compile(r".+\.zip$")
MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG = "_mlflow_local_backend_run_id"
MLFLOW_DOCKER_WORKDIR_PATH = "/mlflow/projects/code/"

PROJECT_USE_CONDA = "USE_CONDA"
PROJECT_SYNCHRONOUS = "SYNCHRONOUS"
PROJECT_DOCKER_ARGS = "DOCKER_ARGS"
PROJECT_STORAGE_DIR = "STORAGE_DIR"
GIT_FETCH_DEPTH = 1


_logger = logging.getLogger(__name__)


def _parse_subdirectory(uri):
    # Parses a uri and returns the uri and subdirectory as separate values.
    # Uses '#' as a delimiter.
    unquoted_uri = _strip_quotes(uri)
    subdirectory = ""
    parsed_uri = unquoted_uri
    if "#" in unquoted_uri:
        subdirectory = unquoted_uri[unquoted_uri.find("#") + 1 :]
        parsed_uri = unquoted_uri[: unquoted_uri.find("#")]
    if subdirectory and "." in subdirectory:
        raise ExecutionException("'.' is not allowed in project subdirectory paths.")
    return parsed_uri, subdirectory


def _strip_quotes(uri):
    return uri.strip("'\"")


def _get_storage_dir(storage_dir):
    if storage_dir is not None and not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    return tempfile.mkdtemp(dir=storage_dir)


def _get_git_repo_url(work_dir):
    from git import Repo
    from git.exc import GitCommandError, InvalidGitRepositoryError

    try:
        repo = Repo(work_dir, search_parent_directories=True)
        remote_urls = [remote.url for remote in repo.remotes]
        if len(remote_urls) == 0:
            return None
    except GitCommandError:
        return None
    except InvalidGitRepositoryError:
        return None
    return remote_urls[0]


def _expand_uri(uri):
    if _is_local_uri(uri):
        return os.path.abspath(uri)
    return uri


def _is_file_uri(uri):
    """Returns True if the passed-in URI is a file:// URI."""
    return _FILE_URI_REGEX.match(uri)


def _is_local_uri(uri):
    """Returns True if the passed-in URI should be interpreted as a path on the local filesystem."""
    return not _GIT_URI_REGEX.match(uri)


def _is_zip_uri(uri):
    """Returns True if the passed-in URI points to a ZIP file."""
    return _ZIP_URI_REGEX.match(uri)


def _is_valid_branch_name(work_dir, version):
    """
    Returns True if the ``version`` is the name of a branch in a Git project.
    ``work_dir`` must be the working directory in a git repo.
    """
    if version is not None:
        from git import Repo
        from git.exc import GitCommandError

        repo = Repo(work_dir, search_parent_directories=True)
        try:
            return repo.git.rev_parse("--verify", "refs/heads/%s" % version) != ""
        except GitCommandError:
            return False
    return False


def fetch_and_validate_project(uri, version, entry_point, parameters):
    parameters = parameters or {}
    work_dir = _fetch_project(uri=uri, version=version)
    project = _project_spec.load_project(work_dir)
    project.get_entry_point(entry_point)._validate_parameters(parameters)
    return work_dir


def load_project(work_dir):
    return _project_spec.load_project(work_dir)


def _fetch_project(uri, version=None):
    """
    Fetch a project into a local directory, returning the path to the local project directory.
    """
    parsed_uri, subdirectory = _parse_subdirectory(uri)
    use_temp_dst_dir = _is_zip_uri(parsed_uri) or not _is_local_uri(parsed_uri)
    dst_dir = tempfile.mkdtemp() if use_temp_dst_dir else parsed_uri
    if use_temp_dst_dir:
        _logger.info("=== Fetching project from %s into %s ===", uri, dst_dir)
    if _is_zip_uri(parsed_uri):
        if _is_file_uri(parsed_uri):
            parsed_file_uri = urllib.parse.urlparse(urllib.parse.unquote(parsed_uri))
            parsed_uri = os.path.join(parsed_file_uri.netloc, parsed_file_uri.path)
        _unzip_repo(
            zip_file=(parsed_uri if _is_local_uri(parsed_uri) else _fetch_zip_repo(parsed_uri)),
            dst_dir=dst_dir,
        )
    elif _is_local_uri(uri):
        if version is not None:
            raise ExecutionException("Setting a version is only supported for Git project URIs")
        if use_temp_dst_dir:
            dir_util.copy_tree(src=parsed_uri, dst=dst_dir)
    else:
        assert _GIT_URI_REGEX.match(parsed_uri), "Non-local URI %s should be a Git URI" % parsed_uri
        _fetch_git_repo(parsed_uri, version, dst_dir)
    res = os.path.abspath(os.path.join(dst_dir, subdirectory))
    if not os.path.exists(res):
        raise ExecutionException("Could not find subdirectory %s of %s" % (subdirectory, dst_dir))
    return res


def _unzip_repo(zip_file, dst_dir):
    import zipfile

    with zipfile.ZipFile(zip_file) as zip_in:
        zip_in.extractall(dst_dir)


def _fetch_git_repo(uri, version, dst_dir):
    """
    Clone the git repo at ``uri`` into ``dst_dir``, checking out commit ``version`` (or defaulting
    to the head commit of the repository's master branch if version is unspecified).
    Assumes authentication parameters are specified by the environment, e.g. by a Git credential
    helper.
    """
    # We defer importing git until the last moment, because the import requires that the git
    # executable is availble on the PATH, so we only want to fail if we actually need it.
    import git

    repo = git.Repo.init(dst_dir)
    origin = repo.create_remote("origin", uri)
    origin.fetch(depth=GIT_FETCH_DEPTH)
    if version is not None:
        try:
            repo.git.checkout(version)
        except git.exc.GitCommandError as e:
            raise ExecutionException(
                "Unable to checkout version '%s' of git repo %s"
                "- please ensure that the version exists in the repo. "
                "Error: %s" % (version, uri, e)
            )
    else:
        repo.create_head("master", origin.refs.master)
        repo.heads.master.checkout()
    repo.submodule_update(init=True, recursive=True)


def _fetch_zip_repo(uri):
    import requests
    from io import BytesIO

    # TODO (dbczumar): Replace HTTP resolution via ``requests.get`` with an invocation of
    # ```mlflow.data.download_uri()`` when the API supports the same set of available stores as
    # the artifact repository (Azure, FTP, etc). See the following issue:
    # https://github.com/mlflow/mlflow/issues/763.
    response = requests.get(uri)
    try:
        augmented_raise_for_status(response)
    except requests.HTTPError as error:
        raise ExecutionException("Unable to retrieve ZIP file. Reason: %s" % str(error))
    return BytesIO(response.content)


def get_or_create_run(run_id, uri, experiment_id, work_dir, version, entry_point, parameters):
    if run_id:
        return tracking.MlflowClient().get_run(run_id)
    else:
        return _create_run(uri, experiment_id, work_dir, version, entry_point, parameters)


def _create_run(uri, experiment_id, work_dir, version, entry_point, parameters):
    """
    Create a ``Run`` against the current MLflow tracking server, logging metadata (e.g. the URI,
    entry point, and parameters of the project) about the run. Return an ``ActiveRun`` that can be
    used to report additional data about the run (metrics/params) to the tracking server.
    """
    if _is_local_uri(uri):
        source_name = tracking._tracking_service.utils._get_git_url_if_present(_expand_uri(uri))
    else:
        source_name = _expand_uri(uri)
    source_version = _get_git_commit(work_dir)
    existing_run = fluent.active_run()
    if existing_run:
        parent_run_id = existing_run.info.run_id
    else:
        parent_run_id = None

    tags = {
        MLFLOW_USER: _get_user(),
        MLFLOW_SOURCE_NAME: source_name,
        MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.PROJECT),
        MLFLOW_PROJECT_ENTRY_POINT: entry_point,
    }
    if source_version is not None:
        tags[MLFLOW_GIT_COMMIT] = source_version
    if parent_run_id is not None:
        tags[MLFLOW_PARENT_RUN_ID] = parent_run_id

    repo_url = _get_git_repo_url(work_dir)
    if repo_url is not None:
        tags[MLFLOW_GIT_REPO_URL] = repo_url
        tags[LEGACY_MLFLOW_GIT_REPO_URL] = repo_url

    # Add branch name tag if a branch is specified through -version
    if _is_valid_branch_name(work_dir, version):
        tags[MLFLOW_GIT_BRANCH] = version
        tags[LEGACY_MLFLOW_GIT_BRANCH_NAME] = version
    active_run = tracking.MlflowClient().create_run(experiment_id=experiment_id, tags=tags)

    project = _project_spec.load_project(work_dir)
    # Consolidate parameters for logging.
    # `storage_dir` is `None` since we want to log actual path not downloaded local path
    entry_point_obj = project.get_entry_point(entry_point)
    final_params, extra_params = entry_point_obj.compute_parameters(parameters, storage_dir=None)
    params_list = [
        Param(key, value) for key, value in list(final_params.items()) + list(extra_params.items())
    ]
    tracking.MlflowClient().log_batch(active_run.info.run_id, params=params_list)
    return active_run


def get_entry_point_command(project, entry_point, parameters, storage_dir):
    """
    Returns the shell command to execute in order to run the specified entry point.
    :param project: Project containing the target entry point
    :param entry_point: Entry point to run
    :param parameters: Parameters (dictionary) for the entry point command
    :param storage_dir: Base local directory to use for downloading remote artifacts passed to
                        arguments of type 'path'. If None, a temporary base directory is used.
    """
    storage_dir_for_run = _get_storage_dir(storage_dir)
    _logger.info(
        "=== Created directory %s for downloading remote URIs passed to arguments of"
        " type 'path' ===",
        storage_dir_for_run,
    )
    commands = []
    commands.append(
        project.get_entry_point(entry_point).compute_command(parameters, storage_dir_for_run)
    )
    return commands


def get_run_env_vars(run_id, experiment_id):
    """
    Returns a dictionary of environment variable key-value pairs to set in subprocess launched
    to run MLflow projects.
    """
    return {
        tracking._RUN_ID_ENV_VAR: run_id,
        tracking._TRACKING_URI_ENV_VAR: tracking.get_tracking_uri(),
        tracking._EXPERIMENT_ID_ENV_VAR: str(experiment_id),
    }


def get_databricks_env_vars(tracking_uri):
    if not mlflow.utils.uri.is_databricks_uri(tracking_uri):
        return {}

    config = databricks_utils.get_databricks_host_creds(tracking_uri)
    # We set these via environment variables so that only the current profile is exposed, rather
    # than all profiles in ~/.databrickscfg; maybe better would be to mount the necessary
    # part of ~/.databrickscfg into the container
    env_vars = {}
    env_vars[tracking._TRACKING_URI_ENV_VAR] = "databricks"
    env_vars["DATABRICKS_HOST"] = config.host
    if config.username:
        env_vars["DATABRICKS_USERNAME"] = config.username
    if config.password:
        env_vars["DATABRICKS_PASSWORD"] = config.password
    if config.token:
        env_vars["DATABRICKS_TOKEN"] = config.token
    if config.ignore_tls_verification:
        env_vars["DATABRICKS_INSECURE"] = str(config.ignore_tls_verification)
    return env_vars
