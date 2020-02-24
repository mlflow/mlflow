import logging
import os
import re
import tempfile
import collections

from distutils import dir_util
from six.moves import urllib

from mlflow.entities import SourceType
from mlflow.exceptions import ExecutionException
from mlflow.projects import _project_spec
from mlflow import tracking
from mlflow.tracking.context.git_context import _get_git_commit
from mlflow.tracking import fluent
from mlflow.tracking.context.default_context import _get_user
from mlflow.utils.mlflow_tags import (
    MLFLOW_USER, MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE, MLFLOW_GIT_COMMIT, MLFLOW_GIT_REPO_URL,
    MLFLOW_GIT_BRANCH, LEGACY_MLFLOW_GIT_REPO_URL, LEGACY_MLFLOW_GIT_BRANCH_NAME,
    MLFLOW_PROJECT_ENTRY_POINT, MLFLOW_PARENT_RUN_ID,
)


# TODO: this should be restricted to just Git repos and not S3 and stuff like that
_GIT_URI_REGEX = re.compile(r"^[^/]*:")
_FILE_URI_REGEX = re.compile(r"^file://.+")
_ZIP_URI_REGEX = re.compile(r".+\.zip$")

_logger = logging.getLogger(__name__)


def _parse_subdirectory(uri):
    # Parses a uri and returns the uri and subdirectory as separate values.
    # Uses '#' as a delimiter.
    subdirectory = ''
    parsed_uri = uri
    if '#' in uri:
        subdirectory = uri[uri.find('#') + 1:]
        parsed_uri = uri[:uri.find('#')]
    if subdirectory and '.' in subdirectory:
        raise ExecutionException("'.' is not allowed in project subdirectory paths.")
    return parsed_uri, subdirectory


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
            return repo.git.rev_parse("--verify", "refs/heads/%s" % version) != ''
        except GitCommandError:
            return False
    return False


def generate_env_vars_to_attach_to_run(run):
    """
    Returns a dictionary of environment variable key-value pairs to set in subprocess launched
    to run MLflow projects.
    """
    return {
        tracking._RUN_ID_ENV_VAR: run.info.run_id,
        tracking._TRACKING_URI_ENV_VAR: tracking.get_tracking_uri(),
        tracking._EXPERIMENT_ID_ENV_VAR: str(run.info.experiment_id),
    }


def fetch_and_validate_project(uri, version, entry_point, parameters):
    parameters = parameters or {}
    work_dir = fetch_project(uri=uri, force_tempdir=False, version=version)
    project = _project_spec.load_project(work_dir)
    project.get_entry_point(entry_point)._validate_parameters(parameters)
    return project, work_dir


def fetch_project(uri, force_tempdir, version=None):
    """
    Fetch a project into a local directory, returning the path to the local project directory.
    :param force_tempdir: If True, will fetch the project into a temporary directory. Otherwise,
                          will fetch ZIP or Git projects into a temporary directory but simply
                          return the path of local projects (i.e. perform a no-op for local
                          projects).
    """
    parsed_uri, subdirectory = _parse_subdirectory(uri)
    use_temp_dst_dir = force_tempdir or _is_zip_uri(parsed_uri) or not _is_local_uri(parsed_uri)
    dst_dir = tempfile.mkdtemp() if use_temp_dst_dir else parsed_uri
    if use_temp_dst_dir:
        _logger.info("=== Fetching project from %s into %s ===", uri, dst_dir)
    if _is_zip_uri(parsed_uri):
        if _is_file_uri(parsed_uri):
            parsed_file_uri = urllib.parse.urlparse(urllib.parse.unquote(parsed_uri))
            parsed_uri = os.path.join(parsed_file_uri.netloc, parsed_file_uri.path)
        _unzip_repo(zip_file=(
            parsed_uri if _is_local_uri(parsed_uri) else _fetch_zip_repo(parsed_uri)),
            dst_dir=dst_dir)
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
    origin.fetch()
    if version is not None:
        try:
            repo.git.checkout(version)
        except git.exc.GitCommandError as e:
            raise ExecutionException("Unable to checkout version '%s' of git repo %s"
                                     "- please ensure that the version exists in the repo. "
                                     "Error: %s" % (version, uri, e))
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
        response.raise_for_status()
    except requests.HTTPError as error:
        raise ExecutionException("Unable to retrieve ZIP file. Reason: %s" % str(error))
    return BytesIO(response.content)


def get_or_create_run(run_id, uri, experiment_id, work_dir, entry_point):
    if run_id:
        return tracking.MlflowClient().get_run(run_id)
    else:
        return _create_run(uri, experiment_id, work_dir, entry_point)


def _create_run(uri, experiment_id, work_dir, entry_point):
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
        MLFLOW_PROJECT_ENTRY_POINT: entry_point
    }
    if source_version is not None:
        tags[MLFLOW_GIT_COMMIT] = source_version
    if parent_run_id is not None:
        tags[MLFLOW_PARENT_RUN_ID] = parent_run_id

    active_run = tracking.MlflowClient().create_run(experiment_id=experiment_id, tags=tags)
    return active_run


def log_project_params_and_tags(run, project, work_dir, entry_point, parameters, version):
    # Consolidate parameters for logging.
    # `storage_dir` is `None` since we want to log actual path not downloaded local path
    entry_point_obj = project.get_entry_point(entry_point)
    final_params, extra_params = entry_point_obj.compute_parameters(parameters, storage_dir=None)
    for key, value in (list(final_params.items()) + list(extra_params.items())):
        tracking.MlflowClient().log_param(run.info.run_id, key, value)

    repo_url = _get_git_repo_url(work_dir)
    if repo_url is not None:
        for tag in [MLFLOW_GIT_REPO_URL, LEGACY_MLFLOW_GIT_REPO_URL]:
            tracking.MlflowClient().set_tag(run.info.run_id, tag, repo_url)

    # Add branch name tag if a branch is specified through -version
    if _is_valid_branch_name(work_dir, version):
        for tag in [MLFLOW_GIT_BRANCH, LEGACY_MLFLOW_GIT_BRANCH_NAME]:
            tracking.MlflowClient().set_tag(run.info.run_id, tag, version)


FetchedProject = collections.namedtuple("FetchedProject", ["project", "work_dir", "run"])


def prepare_project_and_run(project_uri, experiment_id, run_id, entry_point, parameters, version):
    """
    Fetch project, read MLProject, create run  and set all needed information in the run

    :param project_uri: URI to the project (could be a local or git URI).
                    This is the parameter given to mlflow run command.
    :param experiment_id: experiment id
    :param run_id: if not None, run_id to use
    :param entry_point: entry point name in MLProject file
    :param parameters: parameters to pass to the command
    :param version: for Git-based projects, either a commit hash or a branch name.

    :return: A NamedTuple FetchedProject that contains an object describing the MLProject file,
             path where the project has been fetched and a mlflow run object
    """
    project, work_dir = fetch_and_validate_project(project_uri, version, entry_point, parameters)
    active_run = get_or_create_run(run_id, project_uri, experiment_id, work_dir, entry_point)
    log_project_params_and_tags(run_id, project, work_dir, entry_point, parameters, version)
    return FetchedProject(project, work_dir, active_run)
