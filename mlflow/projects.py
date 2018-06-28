from __future__ import print_function

import hashlib
import json
import os
import re
import shutil
import tempfile

from distutils import dir_util

from six.moves import shlex_quote
from databricks_cli.configure import provider

from mlflow.version import VERSION
from mlflow.entities.source_type import SourceType
from mlflow.entities.param import Param
from mlflow import data
import mlflow.tracking as tracking

from mlflow.utils import file_utils, process, rest_utils
from mlflow.utils.logging_utils import eprint


class ExecutionException(Exception):
    pass


class Project(object):
    """A project specification loaded from an MLproject file."""
    def __init__(self, uri, yaml_obj):
        self.uri = uri
        self.name = os.path.splitext(os.path.basename(os.path.abspath(uri)))[0]
        self.conda_env = yaml_obj.get("conda_env")
        self.entry_points = {}
        for name, entry_point_yaml in yaml_obj.get("entry_points", {}).items():
            parameters = entry_point_yaml.get("parameters", {})
            command = entry_point_yaml.get("command")
            self.entry_points[name] = EntryPoint(name, parameters, command)
        # TODO: validate the spec, e.g. make sure paths in it are fine

    def get_entry_point(self, entry_point):
        if entry_point in self.entry_points:
            return self.entry_points[entry_point]
        _, file_extension = os.path.splitext(entry_point)
        ext_to_cmd = {".py": "python", ".sh": os.environ.get("SHELL", "bash")}
        if file_extension in ext_to_cmd:
            command = "%s %s" % (ext_to_cmd[file_extension], shlex_quote(entry_point))
            return EntryPoint(name=entry_point, parameters={}, command=command)
        raise ExecutionException("Could not find {0} among entry points {1} or interpret {0} as a "
                                 "runnable script. Supported script file extensions: "
                                 "{2}".format(entry_point, list(self.entry_points.keys()),
                                              list(ext_to_cmd.keys())))


class EntryPoint(object):
    """An entry point in an MLproject specification."""
    def __init__(self, name, parameters, command):
        self.name = name
        self.parameters = {k: Parameter(k, v) for (k, v) in parameters.items()}
        self.command = command
        assert isinstance(self.command, str)

    def _validate_parameters(self, user_parameters):
        missing_params = []
        for name in self.parameters:
            if name not in user_parameters and self.parameters[name].default is None:
                missing_params.append(name)
        if len(missing_params) == 1:
            raise ExecutionException(
                "No value given for missing parameter: '%s'" % missing_params[0])
        elif len(missing_params) > 1:
            raise ExecutionException(
                "No value given for missing parameters: %s" %
                ", ".join(["'%s'" % name for name in missing_params]))

    def compute_parameters(self, user_parameters, storage_dir):
        """
        Given a dict mapping user-specified param names to values, computes parameters to
        substitute into the command for this entry point. Returns a tuple (params, extra_params)
        where `params` contains key-value pairs for parameters specified in the entry point
        definition, and `extra_params` contains key-value pairs for additional parameters passed
        by the user.

        Note that resolving parameter values can be a heavy operation, e.g. if a remote URI is
        passed for a parameter of type `path`, we download the URI to a local path within
        `storage_dir` and substitute in the local path as the parameter value.
        """
        if user_parameters is None:
            user_parameters = {}
        # Validate params before attempting to resolve parameter values
        self._validate_parameters(user_parameters)
        final_params = {}
        extra_params = {}

        for name, param_obj in self.parameters.items():
            if name in user_parameters:
                final_params[name] = param_obj.compute_value(user_parameters[name], storage_dir)
            else:
                final_params[name] = self.parameters[name].default
        for name in user_parameters:
            if name not in final_params:
                extra_params[name] = user_parameters[name]
        return _sanitize_param_dict(final_params), _sanitize_param_dict(extra_params)

    def compute_command(self, user_parameters, storage_dir):
        params, extra_params = self.compute_parameters(user_parameters, storage_dir)
        command_with_params = self.command.format(**params)
        command_arr = [command_with_params]
        command_arr.extend(["--%s %s" % (key, value) for key, value in extra_params.items()])
        return " ".join(command_arr)


class Parameter(object):
    """A parameter in an MLproject entry point."""
    def __init__(self, name, yaml_obj):
        self.name = name
        if isinstance(yaml_obj, str):
            self.type = yaml_obj
            self.default = None
        else:
            self.type = yaml_obj.get("type", "string")
            self.default = yaml_obj.get("default")

    def _compute_uri_value(self, user_param_value):
        if not data.is_uri(user_param_value):
            raise ExecutionException("Expected URI for parameter %s but got "
                                     "%s" % (self.name, user_param_value))
        return user_param_value

    def _compute_path_value(self, user_param_value, storage_dir):
        if not data.is_uri(user_param_value):
            if not os.path.exists(user_param_value):
                raise ExecutionException("Got value %s for parameter %s, but no such file or "
                                         "directory was found." % (user_param_value, self.name))
            return os.path.abspath(user_param_value)
        basename = os.path.basename(user_param_value)
        dest_path = os.path.join(storage_dir, basename)
        if dest_path != user_param_value:
            data.download_uri(uri=user_param_value, output_path=dest_path)
        return os.path.abspath(dest_path)

    def compute_value(self, user_param_value, storage_dir):
        if self.type != "path" and self.type != "uri":
            return user_param_value
        if self.type == "uri":
            return self._compute_uri_value(user_param_value)
        return self._compute_path_value(user_param_value, storage_dir)


def _sanitize_param_dict(param_dict):
    return {str(key): shlex_quote(str(value)) for key, value in param_dict.items()}


def _get_databricks_run_cmd(uri, entry_point, version, parameters):
    """
    Generates MLflow CLI command to run on Databricks cluster in order to launch a run on Databricks
    """
    mlflow_run_cmd = ["mlflow", "run", uri, "--entry-point", entry_point]
    if version is not None:
        mlflow_run_cmd.extend(["--version", version])
    if parameters is not None:
        for key, value in parameters.items():
            mlflow_run_cmd.extend(["-P", "%s=%s" % (key, value)])
    mlflow_run_str = " ".join(map(shlex_quote, mlflow_run_cmd))
    return ["bash", "-c", "export PATH=$PATH:$DB_HOME/python/bin:/$DB_HOME/conda/bin && %s"
            % mlflow_run_str]



def _get_db_hostname_and_auth():
    """
    Reads the hostname & auth token to use for running on Databricks from the config file created
    by the Databricks CLI.
    """
    home_dir = os.path.expanduser("~")
    cfg_file = os.path.join(home_dir, ".databrickscfg")
    if not os.path.exists(cfg_file):
        raise ExecutionException("Could not find profile for Databricks CLI in %s. Make sure the "
                                 "the Databricks CLI is installed and that credentials have been "
                                 "configured as described in "
                                 "https://github.com/databricks/databricks-cli" % cfg_file)
    else:
        config = provider.get_config_for_profile(provider.DEFAULT_SECTION)
        return config.host, config.token, config.username, config.password


def _run_databricks(uri, entry_point, version, parameters, experiment_id, cluster_spec,
                    git_username, git_password):
    hostname, token, username, password, = _get_db_hostname_and_auth()
    auth = (username, password) if username is not None and password is not None else None
    # Read cluster spec from file
    with open(cluster_spec, 'r') as handle:
        cluster_spec = json.load(handle)
    # Make jobs API request to launch run.
    env_vars = {"MLFLOW_GIT_URI": uri}
    if git_username is not None:
        env_vars["MLFLOW_GIT_USERNAME"] = git_username
    if git_password is not None:
        env_vars["MLFLOW_GIT_PASSWORD"] = git_password
    # Pass experiment ID to shell job on Databricks as an environment variable.
    if experiment_id is not None:
        eprint("=== Using experiment ID %s ===" % experiment_id)
        env_vars[tracking._EXPERIMENT_ID_ENV_VAR] = experiment_id
    req_body_json = {
        'run_name': 'MLflow Job Run for %s' % uri,
        'new_cluster': cluster_spec,
        'shell_command_task': {
            'command': _get_databricks_run_cmd(uri, entry_point, version, parameters),
            "env_vars": env_vars
        },
        "libraries": [{"pypi": {"package": "mlflow==%s" % VERSION}}]
    }
    eprint("=== Running entry point %s of project %s on Databricks. ===" % (entry_point, uri))
    run_submit_res = rest_utils.databricks_api_request(
        hostname=hostname, endpoint="jobs/runs/submit", token=token, auth=auth, method="POST",
        req_body_json=req_body_json)
    run_id = run_submit_res["run_id"]
    eprint("=== Launched MLflow run as Databricks job run with ID %s. Getting run status "
           "page URL... ===" % run_id)
    run_info = rest_utils.databricks_api_request(
        hostname=hostname, endpoint="jobs/runs/get", token=token, auth=auth, method="GET",
        params={"run_id": run_id})
    jobs_page_url = run_info["run_page_url"]
    eprint("=== Check the run's status at %s ===" % jobs_page_url)


def _run_local(uri, entry_point, version, parameters, experiment_id, use_conda, use_temp_cwd,
               storage_dir, git_username, git_password):
    """
    Run an MLflow project from the given URI in a new directory.

    Supports downloading projects from Git URIs with a specified version, or copying them from
    the file system. For Git-based projects, a commit can be specified as the `version`.
    """
    eprint("=== Fetching project from %s ===" % uri)

    # Get the working directory to use for running the project & download it there
    work_dir = _get_work_dir(uri, use_temp_cwd)
    eprint("=== Work directory for this run: %s ===" % work_dir)
    expanded_uri = _expand_uri(uri)
    _fetch_project(expanded_uri, version, work_dir, git_username, git_password)

    # Load the MLproject file
    if not os.path.isfile(os.path.join(work_dir, "MLproject")):
        raise ExecutionException("No MLproject file found in %s" % uri)
    project = Project(expanded_uri, file_utils.read_yaml(work_dir, "MLproject"))
    _run_project(project, entry_point, work_dir, parameters, use_conda, storage_dir, experiment_id)


def run(uri, entry_point="main", version=None, parameters=None, experiment_id=None,
        mode=None, cluster_spec=None, git_username=None, git_password=None, use_conda=True,
        use_temp_cwd=False, storage_dir=None):
    """
    Run an MLflow project from the given URI in a new directory.

    Supports downloading projects from Git URIs with a specified version, or copying them from
    the file system. For Git-based projects, a commit can be specified as the `version`.

    :param entry_point: Entry point to run within the project. If no entry point with the specified
                        name is found, attempts to run the project file `entry_point` as a script,
                        using "python" to run .py files and the default shell (specified by
                        environment variable $SHELL) to run .sh files.
    :param experiment_id: ID of experiment under which to launch the run.
    :param mode: Execution mode for the run. Can be set to "databricks" or "local"
    :param cluster_spec: Path to JSON file describing the cluster to use when launching a run on
                         Databricks.
    :param git_username: Username for HTTP(S) authentication with Git.
    :param git_password: Password for HTTP(S) authentication with Git.
    :param use_conda: If True (the default), creates a new Conda environment for the run and
                      installs project dependencies within that environment. Otherwise, runs the
                      project in the current environment without installing any project
                      dependencies.
    :param use_temp_cwd: Only used if `mode` is "local" and `uri` is a local directory.
                         If True, copies project to a temporary working directory before running it.
                         Otherwise (the default), runs project using `uri` (the project's path) as
                         the working directory.
    :param storage_dir: Only used if `mode` is local. MLflow will download artifacts from
                        distributed URIs passed to parameters of type 'path' to subdirectories of
                        storage_dir.
    """
    if mode is None or mode == "local":
        _run_local(uri=uri, entry_point=entry_point, version=version, parameters=parameters,
                   experiment_id=experiment_id, use_conda=use_conda, use_temp_cwd=use_temp_cwd,
                   storage_dir=storage_dir, git_username=git_username, git_password=git_password)
    elif mode == "databricks":
        _run_databricks(uri=uri, entry_point=entry_point, version=version, parameters=parameters,
                        experiment_id=experiment_id, cluster_spec=cluster_spec,
                        git_username=git_username, git_password=git_password)
    else:
        supported_modes = ["local", "databricks"]
        raise ExecutionException("Got unsupported execution mode %s. Supported "
                                 "values: %s" % (mode, supported_modes))


# TODO: this should be restricted to just Git repos and not S3 and stuff like that
_GIT_URI_REGEX = re.compile(r"^[^/]*:")


def _get_work_dir(uri, use_temp_cwd):
    """
    Returns a working directory to use for fetching & running the project with the specified URI.
    :param use_temp_cwd: Only used if `uri` is a local directory. If True, returns a temporary
                         working directory.
    """
    if _GIT_URI_REGEX.match(uri) or use_temp_cwd:
        # Create a temp directory to download and run the project in
        return tempfile.mkdtemp(prefix="mlflow-")
    return os.path.abspath(uri)


def _get_storage_dir(storage_dir):
    if storage_dir is not None and not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    return tempfile.mkdtemp(dir=storage_dir)


def _expand_uri(uri):
    if _GIT_URI_REGEX.match(uri):
        return uri
    return os.path.abspath(uri)


def _fetch_project(uri, version, dst_dir, git_username, git_password):
    """Download a project to the target `dst_dir` from a Git URI or local path."""
    if _GIT_URI_REGEX.match(uri):
        # Use Git to clone the project
        _fetch_git_repo(uri, version, dst_dir, git_username, git_password)
    else:
        if version is not None:
            raise ExecutionException("Setting a version is only supported for Git project URIs")
        # TODO: don't copy mlruns directory here
        # Note: uri might be equal to dst_dir, e.g. if we're not using a temporary work dir
        if uri != dst_dir:
            dir_util.copy_tree(src=uri, dst=dst_dir)

    # Make sure they don't have an outputs or mlruns directory (will need to change if we change
    # how we log results locally)
    shutil.rmtree(os.path.join(dst_dir, "outputs"), ignore_errors=True)
    shutil.rmtree(os.path.join(dst_dir, "mlruns"), ignore_errors=True)


def _fetch_git_repo(uri, version, dst_dir, git_username, git_password):
    """
    Clones the git repo at `uri` into `dst_dir`, checking out commit `version` (or defaulting
    to the head commit of the repository's master branch if version is unspecified). If git_username
    and git_password are specified, uses them to authenticate while fetching the repo. Otherwise,
    assumes authentication parameters are specified by the environment, e.g. by a Git credential
    helper.
    """
    # We defer importing git until the last moment, because the import requires that the git
    # executable is availble on the PATH, so we only want to fail if we actually need it.
    import git
    repo = git.Repo.init(dst_dir)
    origin = repo.create_remote("origin", uri)
    git_args = [git_username, git_password]
    if not (all(arg is not None for arg in git_args) or all(arg is None for arg in git_args)):
        raise ExecutionException("Either both or neither of git_username and git_password must be "
                                 "specified.")
    if git_username:
        git_credentials = "url=%s\nusername=%s\npassword=%s" % (uri, git_username, git_password)
        repo.git.config("--local", "credential.helper", "cache")
        process.exec_cmd(cmd=["git", "credential-cache", "store"], cwd=dst_dir,
                         cmd_stdin=git_credentials)
    origin.fetch()
    if version is not None:
        repo.git.checkout(version)
    else:
        repo.create_head("master", origin.refs.master)
        repo.heads.master.checkout()


def _get_conda_env_name(conda_env_path):
    with open(conda_env_path) as conda_env_file:
        conda_env_hash = hashlib.sha1(conda_env_file.read().encode("utf-8")).hexdigest()
    return "mlflow-%s" % conda_env_hash


def _maybe_create_conda_env(conda_env_path):
    conda_env = _get_conda_env_name(conda_env_path)
    try:
        process.exec_cmd(["conda", "--help"], throw_on_error=False)
    except EnvironmentError:
        raise ExecutionException('conda is not installed properly. Please follow the instructions '
                                 'on https://conda.io/docs/user-guide/install/index.html')
    (_, stdout, _) = process.exec_cmd(["conda", "env", "list", "--json"])
    env_names = [os.path.basename(env) for env in json.loads(stdout)['envs']]

    conda_action = 'create'
    if conda_env not in env_names:
        eprint('=== Creating conda environment %s ===' % conda_env)
        process.exec_cmd(["conda", "env", conda_action, "-n", conda_env, "--file",
                          conda_env_path], stream_output=True)


def _run_project(project, entry_point, work_dir, parameters, use_conda, storage_dir, experiment_id):
    """Locally run a project that has been checked out in `work_dir`."""
    storage_dir_for_run = _get_storage_dir(storage_dir)
    eprint("=== Created directory %s for downloading remote URIs passed to arguments of "
           "type 'path' ===" % storage_dir_for_run)
    # Try to build the command first in case the user mis-specified parameters
    run_project_command = project.get_entry_point(entry_point)\
        .compute_command(parameters, storage_dir_for_run)
    commands = []
    if use_conda:
        conda_env_path = os.path.abspath(os.path.join(work_dir, project.conda_env))
        _maybe_create_conda_env(conda_env_path)
        commands.append("source activate %s" % _get_conda_env_name(conda_env_path))

    # Create a new run and log every provided parameter into it.
    active_run = tracking.start_run(experiment_id=experiment_id,
                                    source_name=project.uri,
                                    source_version=tracking._get_git_commit(work_dir),
                                    entry_point_name=entry_point,
                                    source_type=SourceType.PROJECT)
    if parameters is not None:
        for key, value in parameters.items():
            active_run.log_param(Param(key, value))
    # Add the run id into a magic environment variable that the subprocess will read,
    # causing it to reuse the run.
    exp_id = experiment_id or tracking._get_experiment_id()
    env_map = {
        tracking._RUN_NAME_ENV_VAR: active_run.run_info.run_uuid,
        tracking._TRACKING_URI_ENV_VAR: tracking.get_tracking_uri(),
        tracking._EXPERIMENT_ID_ENV_VAR: str(exp_id),
    }

    commands.append(run_project_command)
    command = " && ".join(commands)
    eprint("=== Running command: %s ===" % command)
    try:
        process.exec_cmd([os.environ.get("SHELL", "bash"), "-c", command], cwd=work_dir,
                         stream_output=True, env=env_map)
        tracking.end_run()
        eprint("=== Run succeeded ===")
    except process.ShellCommandException:
        tracking.end_run("FAILED")
        eprint("=== Run failed ===")
