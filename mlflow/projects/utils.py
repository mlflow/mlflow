import logging
import os
import pathlib
import re
import shutil
import tempfile
import urllib.parse
import zipfile
from io import BytesIO

from mlflow import tracking
from mlflow.entities import Param, SourceType
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID, MLFLOW_RUN_ID, MLFLOW_TRACKING_URI
from mlflow.exceptions import ExecutionException
from mlflow.projects import _project_spec
from mlflow.tracking import fluent
from mlflow.tracking.context.default_context import _get_user
from mlflow.utils.git_utils import get_git_commit, get_git_repo_url
from mlflow.utils.mlflow_tags import (
    LEGACY_MLFLOW_GIT_BRANCH_NAME,
    LEGACY_MLFLOW_GIT_REPO_URL,
    MLFLOW_GIT_BRANCH,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_REPO_URL,
    MLFLOW_PARENT_RUN_ID,
    MLFLOW_PROJECT_ENTRY_POINT,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_USER,
)
from mlflow.utils.rest_utils import augmented_raise_for_status

_FILE_URI_REGEX = re.compile(r"^file://.+")
_ZIP_URI_REGEX = re.compile(r".+\.zip$")
MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG = "_mlflow_local_backend_run_id"
MLFLOW_DOCKER_WORKDIR_PATH = "/mlflow/projects/code/"

PROJECT_ENV_MANAGER = "ENV_MANAGER"
PROJECT_SYNCHRONOUS = "SYNCHRONOUS"
PROJECT_DOCKER_ARGS = "DOCKER_ARGS"
PROJECT_STORAGE_DIR = "STORAGE_DIR"
PROJECT_BUILD_IMAGE = "build_image"
PROJECT_DOCKER_AUTH = "docker_auth"
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


def _expand_uri(uri):
    if _is_local_uri(uri):
        return os.path.abspath(uri)
    return uri


def _is_file_uri(uri):
    """Returns True if the passed-in URI is a file:// URI."""
    return _FILE_URI_REGEX.match(uri)


def _is_git_repo(path) -> bool:
    """Returns True if passed-in path is a valid git repository"""
    import git

    try:
        repo = git.Repo(path)
        if len(repo.branches) > 0:
            return True
    except git.exc.InvalidGitRepositoryError:
        pass
    return False


def _parse_file_uri(uri: str) -> str:
    """Converts file URIs to filesystem paths"""
    if _is_file_uri(uri):
        parsed_file_uri = urllib.parse.urlparse(uri)
        return str(
            pathlib.Path(parsed_file_uri.netloc, parsed_file_uri.path, parsed_file_uri.fragment)
        )
    return uri


def _is_local_uri(uri: str) -> bool:
    """Returns True if passed-in URI should be interpreted as a folder on the local filesystem."""
    resolved_uri = pathlib.Path(_parse_file_uri(uri)).resolve()
    return resolved_uri.exists()


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
            return repo.git.rev_parse("--verify", f"refs/heads/{version}") != ""
        except GitCommandError:
            return False
    return False


def fetch_and_validate_project(uri, version, entry_point, parameters):
    parameters = parameters or {}
    work_dir = _fetch_project(uri=uri, version=version)
    project = _project_spec.load_project(work_dir)
    if entry_point_obj := project.get_entry_point(entry_point):
        entry_point_obj._validate_parameters(parameters)
    return work_dir


def load_project(work_dir):
    return _project_spec.load_project(work_dir)


def _fetch_project(uri, version=None):
    """
    Fetch a project into a local directory, returning the path to the local project directory.
    """
    parsed_uri, subdirectory = _parse_subdirectory(uri)
    use_temp_dst_dir = _is_zip_uri(parsed_uri) or not _is_local_uri(parsed_uri)
    dst_dir = tempfile.mkdtemp() if use_temp_dst_dir else _parse_file_uri(parsed_uri)

    if use_temp_dst_dir:
        _logger.info("=== Fetching project from %s into %s ===", uri, dst_dir)
    if _is_zip_uri(parsed_uri):
        parsed_uri = _parse_file_uri(parsed_uri)
        _unzip_repo(
            zip_file=(parsed_uri if _is_local_uri(parsed_uri) else _fetch_zip_repo(parsed_uri)),
            dst_dir=dst_dir,
        )
    elif _is_local_uri(parsed_uri):
        if use_temp_dst_dir:
            shutil.copytree(parsed_uri, dst_dir, dirs_exist_ok=True)
        if version is not None:
            if not _is_git_repo(_parse_file_uri(parsed_uri)):
                raise ExecutionException("Setting a version is only supported for Git project URIs")
            _fetch_git_repo(parsed_uri, version, dst_dir)
    else:
        _fetch_git_repo(parsed_uri, version, dst_dir)
    res = os.path.abspath(os.path.join(dst_dir, subdirectory))
    if not os.path.exists(res):
        raise ExecutionException(f"Could not find subdirectory {subdirectory} of {dst_dir}")
    return res


def _unzip_repo(zip_file, dst_dir):
    with zipfile.ZipFile(zip_file) as zip_in:
        zip_in.extractall(dst_dir)


_HEAD_BRANCH_REGEX = re.compile(r"^\s*HEAD branch:\s+(?P<branch>\S+)")


def _get_head_branch(remote_show_output):
    for line in remote_show_output.splitlines():
        match = _HEAD_BRANCH_REGEX.match(line)
        if match:
            return match.group("branch")


def _fetch_git_repo(uri, version, dst_dir):
    """
    Clone the git repo at ``uri`` into ``dst_dir``, checking out commit ``version`` (or defaulting
    to the head commit of the repository's master branch if version is unspecified).
    Assumes authentication parameters are specified by the environment, e.g. by a Git credential
    helper.
    """
    # We defer importing git until the last moment, because the import requires that the git
    # executable is available on the PATH, so we only want to fail if we actually need it.
    import git

    repo = git.Repo.init(dst_dir)
    origin = next((remote for remote in repo.remotes), None)
    if origin is None:
        origin = repo.create_remote("origin", uri)
    if version is not None:
        try:
            origin.fetch(refspec=version, depth=GIT_FETCH_DEPTH, tags=True)
            repo.git.checkout(version)
        except git.exc.GitCommandError as e:
            raise ExecutionException(
                f"Unable to checkout version '{version}' of git repo {uri}"
                "- please ensure that the version exists in the repo. "
                f"Error: {e}"
            )
    else:
        g = git.cmd.Git(dst_dir)
        cmd = ["git", "remote", "show", "origin"]
        output = g.execute(cmd)
        head_branch = _get_head_branch(output)
        if head_branch is None:
            raise ExecutionException(
                "Failed to find HEAD branch. Output of `{cmd}`:\n{output}".format(
                    cmd=" ".join(cmd), output=output
                )
            )
        origin.fetch(head_branch, depth=GIT_FETCH_DEPTH)
        ref = origin.refs[0]
        _logger.info("Fetched '%s' branch", head_branch)
        repo.create_head(head_branch, ref)
        repo.heads[head_branch].checkout()
    repo.git.execute(command=["git", "submodule", "update", "--init", "--recursive"])


def _fetch_zip_repo(uri):
    import requests

    # TODO (dbczumar): Replace HTTP resolution via ``requests.get`` with an invocation of
    # ```mlflow.data.download_uri()`` when the API supports the same set of available stores as
    # the artifact repository (Azure, FTP, etc). See the following issue:
    # https://github.com/mlflow/mlflow/issues/763.
    response = requests.get(uri)
    try:
        augmented_raise_for_status(response)
    except requests.HTTPError as error:
        raise ExecutionException(f"Unable to retrieve ZIP file. Reason: {error!s}")
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
    source_version = get_git_commit(work_dir)
    existing_run = fluent.active_run()
    parent_run_id = existing_run.info.run_id if existing_run else None

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

    repo_url = get_git_repo_url(work_dir)
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
    if entry_point_obj:
        final_params, extra_params = entry_point_obj.compute_parameters(
            parameters, storage_dir=None
        )
        params_list = [
            Param(key, value)
            for key, value in list(final_params.items()) + list(extra_params.items())
        ]
        tracking.MlflowClient().log_batch(active_run.info.run_id, params=params_list)
    return active_run


def get_entry_point_command(project, entry_point, parameters, storage_dir):
    """
    Returns the shell command to execute in order to run the specified entry point.

    Args:
        project: Project containing the target entry point.
        entry_point: Entry point to run.
        parameters: Parameters (dictionary) for the entry point command.
        storage_dir: Base local directory to use for downloading remote artifacts passed to
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
        MLFLOW_RUN_ID.name: run_id,
        MLFLOW_TRACKING_URI.name: tracking.get_tracking_uri(),
        MLFLOW_EXPERIMENT_ID.name: str(experiment_id),
    }
