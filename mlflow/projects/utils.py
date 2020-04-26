"""
Exposes developer APIs for implementing custom project execution backend plugins. For common
use cases like running MLflow projects, see :py:mod:`mlflow.projects`. For more information on
writing a plugin for a custom project execution backend, see `MLflow Plugins <../../plugins.html>`_.
"""
import logging
import os
import re
import tempfile


from distutils import dir_util
from six.moves import urllib
import yaml

from mlflow.entities import SourceType, Param
from mlflow.exceptions import ExecutionException
from mlflow.projects.entities import Project, EntryPoint
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
_MLPROJECT_FILE_NAME = "mlproject"
_DEFAULT_CONDA_FILE_NAME = "conda.yaml"


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


def _find_mlproject(directory):
    filenames = os.listdir(directory)
    for filename in filenames:
        if filename.lower() == _MLPROJECT_FILE_NAME:
            return os.path.join(directory, filename)
    return None


def load_project(directory):
    """
    Loads a :py:class:`mlflow.projects.entities.Project` from the specified directory

    :param directory: Directory from which to load project config. Config will be loaded from a
                      file named "MLproject" in the specified directory, if one exists.
    :return: A :py:class:`mlflow.projects.entities.Project`
    """
    mlproject_path = _find_mlproject(directory)

    # TODO: Validate structure of YAML loaded from the file
    yaml_obj = {}
    if mlproject_path is not None:
        with open(mlproject_path) as mlproject_file:
            yaml_obj = yaml.safe_load(mlproject_file)

    project_name = yaml_obj.get("name")

    # Validate config if docker_env parameter is present
    docker_env = yaml_obj.get("docker_env")
    if docker_env:
        if not docker_env.get("image"):
            raise ExecutionException("Project configuration (MLproject file) was invalid: Docker "
                                     "environment specified but no image attribute found.")
        if docker_env.get("volumes"):
            if not (isinstance(docker_env["volumes"], list)
                    and all([isinstance(i, str) for i in docker_env["volumes"]])):
                raise ExecutionException("Project configuration (MLproject file) was invalid: "
                                         "Docker volumes must be a list of strings, "
                                         """e.g.: '["/path1/:/path1", "/path2/:/path2"])""")
        if docker_env.get("environment"):
            if not (isinstance(docker_env["environment"], list)
                    and all([isinstance(i, list) or isinstance(i, str)
                             for i in docker_env["environment"]])):
                raise ExecutionException(
                    "Project configuration (MLproject file) was invalid: "
                    "environment must be a list containing either strings (to copy environment "
                    "variables from host system) or lists of string pairs (to define new "
                    "environment variables)."
                    """E.g.: '[["NEW_VAR", "new_value"], "VAR_TO_COPY_FROM_HOST"])""")

    # Validate config if conda_env parameter is present
    conda_path = yaml_obj.get("conda_env")
    if conda_path and docker_env:
        raise ExecutionException("Project cannot contain both a docker and "
                                 "conda environment.")

    # Parse entry points
    entry_points = {}
    for name, entry_point_yaml in yaml_obj.get("entry_points", {}).items():
        parameters = entry_point_yaml.get("parameters", {})
        command = entry_point_yaml.get("command")
        entry_points[name] = EntryPoint(name, parameters, command)

    if conda_path:
        conda_env_path = os.path.join(directory, conda_path)
        if not os.path.exists(conda_env_path):
            raise ExecutionException("Project specified conda environment file %s, but no such "
                                     "file was found." % conda_env_path)
        return Project(entry_points=entry_points, name=project_name,
                       conda_env_path=conda_env_path, docker_env=docker_env)

    default_conda_path = os.path.join(directory, _DEFAULT_CONDA_FILE_NAME)
    if os.path.exists(default_conda_path):
        return Project(entry_points=entry_points, name=project_name,
                       conda_env_path=default_conda_path, docker_env=docker_env)

    return Project(entry_points=entry_points, name=project_name, conda_env_path=None,
                   docker_env=docker_env)


def fetch_project(uri, version=None):
    """
    Fetch a project into a local directory, returning the path to the local project directory. If
    provided URI is a remote URI (e.g. the URI of a remote git repo like
    http://github.com/mlflow/mlflow-example.git), the project will be fetched to a local temporary
    directory.

    :param uri: URI of project to run. A local filesystem path
                or a Git repository URI (e.g. https://github.com/mlflow/mlflow-example)
                pointing to a project directory containing an MLproject file.
    :param version: For Git-based projects, either a commit hash or a branch name.
    :return: String path of directory containing fetched project
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


def create_run(uri, experiment_id, work_dir, version, entry_point, parameters):
    """
    Creates and returns an :py:class:`mlflow.entities.Run` against the current MLflow tracking
    server, logging metadata (e.g. the URI, entry point, and parameters of the project) about the
    run.
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

    repo_url = _get_git_repo_url(work_dir)
    if repo_url is not None:
        tags[MLFLOW_GIT_REPO_URL] = repo_url
        tags[LEGACY_MLFLOW_GIT_REPO_URL] = repo_url

    # Add branch name tag if a branch is specified through -version
    if _is_valid_branch_name(work_dir, version):
        tags[MLFLOW_GIT_BRANCH] = version
        tags[LEGACY_MLFLOW_GIT_BRANCH_NAME] = version
    active_run = tracking.MlflowClient().create_run(experiment_id=experiment_id, tags=tags)

    project = load_project(work_dir)
    # Consolidate parameters for logging.
    # `storage_dir` is `None` since we want to log actual path not downloaded local path
    entry_point_obj = project.get_entry_point(entry_point)
    final_params, extra_params = entry_point_obj.compute_parameters(parameters, storage_dir=None)
    params_list = [Param(key, value) for key, value in
                   list(final_params.items()) + list(extra_params.items())]
    tracking.MlflowClient().log_batch(active_run.info.run_id, params=params_list)
    return active_run
