import logging
import os
import platform
import posixpath
import subprocess
import sys
from pathlib import Path

import mlflow
from mlflow import tracking
from mlflow.environment_variables import (
    MLFLOW_KERBEROS_TICKET_CACHE,
    MLFLOW_KERBEROS_USER,
    MLFLOW_PYARROW_EXTRA_CONF,
)
from mlflow.exceptions import MlflowException
from mlflow.projects import env_type
from mlflow.projects.backend.abstract_backend import AbstractBackend
from mlflow.projects.submitted_run import LocalSubmittedRun
from mlflow.projects.utils import (
    MLFLOW_DOCKER_WORKDIR_PATH,
    MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG,
    PROJECT_BUILD_IMAGE,
    PROJECT_DOCKER_ARGS,
    PROJECT_DOCKER_AUTH,
    PROJECT_ENV_MANAGER,
    PROJECT_STORAGE_DIR,
    PROJECT_SYNCHRONOUS,
    fetch_and_validate_project,
    get_entry_point_command,
    get_or_create_run,
    get_run_env_vars,
    load_project,
)
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.azure_blob_artifact_repo import AzureBlobArtifactRepository
from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository
from mlflow.store.artifact.hdfs_artifact_repo import HdfsArtifactRepository
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.conda import get_or_create_conda_env
from mlflow.utils.databricks_utils import get_databricks_env_vars, is_in_databricks_runtime
from mlflow.utils.environment import _PythonEnv
from mlflow.utils.file_utils import get_or_create_nfs_tmp_dir
from mlflow.utils.mlflow_tags import MLFLOW_PROJECT_ENV
from mlflow.utils.os import is_windows
from mlflow.utils.virtualenv import (
    _PYENV_ROOT_DIR,
    _VIRTUALENV_ENVS_DIR,
    _create_virtualenv,
    _get_mlflow_virtualenv_root,
    _get_virtualenv_extra_env_vars,
    _get_virtualenv_name,
)

_logger = logging.getLogger(__name__)


def _env_type_to_env_manager(env_typ):
    if env_typ == env_type.CONDA:
        return _EnvManager.CONDA
    elif env_typ == env_type.PYTHON:
        return _EnvManager.VIRTUALENV
    elif env_typ == env_type.DOCKER:
        return _EnvManager.LOCAL


class LocalBackend(AbstractBackend):
    def run(
        self,
        project_uri,
        entry_point,
        params,
        version,
        backend_config,
        tracking_uri,
        experiment_id,
    ):
        work_dir = fetch_and_validate_project(project_uri, version, entry_point, params)
        project = load_project(work_dir)
        if MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG in backend_config:
            run_id = backend_config[MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG]
        else:
            run_id = None
        active_run = get_or_create_run(
            run_id, project_uri, experiment_id, work_dir, version, entry_point, params
        )
        command_args = []
        command_separator = " "
        env_manager = backend_config[PROJECT_ENV_MANAGER]
        synchronous = backend_config[PROJECT_SYNCHRONOUS]
        docker_args = backend_config[PROJECT_DOCKER_ARGS]
        storage_dir = backend_config[PROJECT_STORAGE_DIR]
        build_image = backend_config[PROJECT_BUILD_IMAGE]
        docker_auth = backend_config[PROJECT_DOCKER_AUTH]
        # Select an appropriate env manager for the project env type
        if env_manager is None:
            env_manager = _env_type_to_env_manager(project.env_type)
        else:
            if project.env_type == env_type.PYTHON and env_manager == _EnvManager.CONDA:
                raise MlflowException.invalid_parameter_value(
                    "python_env project cannot be executed using conda. Set `--env-manager` to "
                    "'virtualenv' or 'local' to execute this project."
                )

        # If a docker_env attribute is defined in MLproject then it takes precedence over conda yaml
        # environments, so the project will be executed inside a docker container.
        if project.docker_env:
            from mlflow.projects.docker import (
                build_docker_image,
                validate_docker_env,
                validate_docker_installation,
            )

            tracking.MlflowClient().set_tag(active_run.info.run_id, MLFLOW_PROJECT_ENV, "docker")
            validate_docker_env(project)
            validate_docker_installation()
            image = build_docker_image(
                work_dir=work_dir,
                repository_uri=project.name,
                base_image=project.docker_env.get("image"),
                run_id=active_run.info.run_id,
                build_image=build_image,
                docker_auth=docker_auth,
            )
            command_args += _get_docker_command(
                image=image,
                active_run=active_run,
                docker_args=docker_args,
                volumes=project.docker_env.get("volumes"),
                user_env_vars=project.docker_env.get("environment"),
            )
        # Synchronously create a conda environment (even though this may take some time)
        # to avoid failures due to multiple concurrent attempts to create the same conda env.
        elif env_manager == _EnvManager.VIRTUALENV:
            tracking.MlflowClient().set_tag(
                active_run.info.run_id, MLFLOW_PROJECT_ENV, "virtualenv"
            )
            command_separator = " && "
            if project.env_type == env_type.CONDA:
                python_env = _PythonEnv.from_conda_yaml(project.env_config_path)
            else:
                python_env = (
                    _PythonEnv.from_yaml(project.env_config_path)
                    if project.env_config_path
                    else _PythonEnv()
                )

            if is_in_databricks_runtime():
                nfs_tmp_dir = get_or_create_nfs_tmp_dir()
                env_root = Path(nfs_tmp_dir) / "envs"
                pyenv_root_dir = str(env_root / _PYENV_ROOT_DIR)
                virtualenv_root = env_root / _VIRTUALENV_ENVS_DIR
                env_vars = _get_virtualenv_extra_env_vars(str(env_root))
            else:
                pyenv_root_dir = None
                virtualenv_root = Path(_get_mlflow_virtualenv_root())
                env_vars = None
            work_dir_path = Path(work_dir)
            env_name = _get_virtualenv_name(python_env, work_dir_path)
            env_dir = virtualenv_root / env_name
            activate_cmd = _create_virtualenv(
                local_model_path=work_dir_path,
                python_env=python_env,
                env_dir=env_dir,
                pyenv_root_dir=pyenv_root_dir,
                env_manager=env_manager,
                extra_env=env_vars,
            )
            command_args += [activate_cmd]
        elif env_manager == _EnvManager.CONDA:
            tracking.MlflowClient().set_tag(active_run.info.run_id, MLFLOW_PROJECT_ENV, "conda")
            command_separator = " && "
            conda_env = get_or_create_conda_env(project.env_config_path)
            command_args += conda_env.get_activate_command()

        # In synchronous mode, run the entry point command in a blocking fashion, sending status
        # updates to the tracking server when finished. Note that the run state may not be
        # persisted to the tracking server if interrupted
        if synchronous:
            command_args += get_entry_point_command(project, entry_point, params, storage_dir)
            command_str = command_separator.join(command_args)
            return _run_entry_point(
                command_str, work_dir, experiment_id, run_id=active_run.info.run_id
            )
        # Otherwise, invoke `mlflow run` in a subprocess
        return _invoke_mlflow_run_subprocess(
            work_dir=work_dir,
            entry_point=entry_point,
            parameters=params,
            experiment_id=experiment_id,
            env_manager=env_manager,
            docker_args=docker_args,
            storage_dir=storage_dir,
            run_id=active_run.info.run_id,
        )


def _invoke_mlflow_run_subprocess(
    work_dir, entry_point, parameters, experiment_id, env_manager, docker_args, storage_dir, run_id
):
    """
    Run an MLflow project asynchronously by invoking ``mlflow run`` in a subprocess, returning
    a SubmittedRun that can be used to query run status.
    """
    _logger.info("=== Asynchronously launching MLflow run with ID %s ===", run_id)
    mlflow_run_arr = _build_mlflow_run_cmd(
        uri=work_dir,
        entry_point=entry_point,
        docker_args=docker_args,
        storage_dir=storage_dir,
        env_manager=env_manager,
        run_id=run_id,
        parameters=parameters,
    )
    env_vars = get_run_env_vars(run_id, experiment_id)
    env_vars.update(get_databricks_env_vars(mlflow.get_tracking_uri()))
    mlflow_run_subprocess = _run_mlflow_run_cmd(mlflow_run_arr, env_vars)
    return LocalSubmittedRun(run_id, mlflow_run_subprocess)


def _build_mlflow_run_cmd(
    uri, entry_point, docker_args, storage_dir, env_manager, run_id, parameters
):
    """
    Build and return an array containing an ``mlflow run`` command that can be invoked to locally
    run the project at the specified URI.
    """
    mlflow_run_arr = ["mlflow", "run", uri, "-e", entry_point, "--run-id", run_id]
    if docker_args is not None:
        for key, value in docker_args.items():
            args = key if isinstance(value, bool) else f"{key}={value}"
            mlflow_run_arr.extend(["--docker-args", args])
    if storage_dir is not None:
        mlflow_run_arr.extend(["--storage-dir", storage_dir])
    mlflow_run_arr.extend(["--env-manager", env_manager])
    for key, value in parameters.items():
        mlflow_run_arr.extend(["-P", f"{key}={value}"])
    return mlflow_run_arr


def _run_mlflow_run_cmd(mlflow_run_arr, env_map):
    """
    Invoke ``mlflow run`` in a subprocess, which in turn runs the entry point in a child process.
    Returns a handle to the subprocess. Popen launched to invoke ``mlflow run``.
    """
    final_env = os.environ.copy()
    final_env.update(env_map)
    # Launch `mlflow run` command as the leader of its own process group so that we can do a
    # best-effort cleanup of all its descendant processes if needed
    if sys.platform == "win32":
        return subprocess.Popen(
            mlflow_run_arr,
            env=final_env,
            text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        return subprocess.Popen(mlflow_run_arr, env=final_env, text=True, preexec_fn=os.setsid)


def _run_entry_point(command, work_dir, experiment_id, run_id):  # noqa: D417
    """
    Run an entry point command in a subprocess, returning a SubmittedRun that can be used to
    query the run's status.

    Args:
        command: Entry point command to run
        work_dir: Working directory in which to run the command
        run_id: MLflow run ID associated with the entry point execution.
    """
    env = os.environ.copy()
    env.update(get_run_env_vars(run_id, experiment_id))
    env.update(get_databricks_env_vars(tracking_uri=mlflow.get_tracking_uri()))
    _logger.info("=== Running command '%s' in run with ID '%s' === ", command, run_id)
    # in case os name is not 'nt', we are not running on windows. It introduces
    # bash command otherwise.
    if not is_windows():
        process = subprocess.Popen(["bash", "-c", command], close_fds=True, cwd=work_dir, env=env)
    else:
        # process = subprocess.Popen(command, close_fds=True, cwd=work_dir, env=env)
        process = subprocess.Popen(["cmd", "/c", command], close_fds=True, cwd=work_dir, env=env)
    return LocalSubmittedRun(run_id, process)


def _get_docker_command(image, active_run, docker_args=None, volumes=None, user_env_vars=None):
    from mlflow.projects.docker import get_docker_tracking_cmd_and_envs

    docker_path = "docker"
    cmd = [docker_path, "run", "--rm"]

    if docker_args:
        for name, value in docker_args.items():
            # Passed just the name as boolean flag
            if isinstance(value, bool) and value:
                if len(name) == 1:
                    cmd += ["-" + name]
                else:
                    cmd += ["--" + name]
            else:
                # Passed name=value
                if len(name) == 1:
                    cmd += ["-" + name, value]
                else:
                    cmd += ["--" + name, value]

    env_vars = get_run_env_vars(
        run_id=active_run.info.run_id, experiment_id=active_run.info.experiment_id
    )
    tracking_uri = tracking.get_tracking_uri()
    tracking_cmds, tracking_envs = get_docker_tracking_cmd_and_envs(tracking_uri)
    artifact_cmds, artifact_envs = _get_docker_artifact_storage_cmd_and_envs(
        active_run.info.artifact_uri
    )

    cmd += tracking_cmds + artifact_cmds
    env_vars.update(tracking_envs)
    env_vars.update(artifact_envs)
    if user_env_vars is not None:
        for user_entry in user_env_vars:
            if isinstance(user_entry, list):
                # User has defined a new environment variable for the docker environment
                env_vars[user_entry[0]] = user_entry[1]
            else:
                # User wants to copy an environment variable from system environment
                system_var = os.environ.get(user_entry)
                if system_var is None:
                    raise MlflowException(
                        "This project expects the {} environment variables to "
                        "be set on the machine running the project, but {} was "
                        "not set. Please ensure all expected environment variables "
                        "are set".format(", ".join(user_env_vars), user_entry)
                    )
                env_vars[user_entry] = system_var

    if volumes is not None:
        for v in volumes:
            cmd += ["-v", v]

    for key, value in env_vars.items():
        cmd += ["-e", f"{key}={value}"]
    cmd += [image.tags[0]]
    return cmd


def _get_local_artifact_cmd_and_envs(artifact_repo):
    artifact_dir = artifact_repo.artifact_dir
    container_path = artifact_dir
    if not os.path.isabs(container_path):
        container_path = os.path.join(MLFLOW_DOCKER_WORKDIR_PATH, container_path)
        container_path = os.path.normpath(container_path)
    abs_artifact_dir = os.path.abspath(artifact_dir)
    return ["-v", f"{abs_artifact_dir}:{container_path}"], {}


def _get_s3_artifact_cmd_and_envs(artifact_repo):
    if platform.system() == "Windows":
        win_user_dir = os.environ["USERPROFILE"]
        aws_path = os.path.join(win_user_dir, ".aws")
    else:
        aws_path = posixpath.expanduser("~/.aws")

    volumes = []
    if posixpath.exists(aws_path):
        volumes = ["-v", "{}:{}".format(str(aws_path), "/.aws")]
    envs = {
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
        "MLFLOW_S3_ENDPOINT_URL": os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
        "MLFLOW_S3_IGNORE_TLS": os.environ.get("MLFLOW_S3_IGNORE_TLS"),
    }
    envs = {k: v for k, v in envs.items() if v is not None}
    return volumes, envs


def _get_azure_blob_artifact_cmd_and_envs(artifact_repo):
    envs = {
        "AZURE_STORAGE_CONNECTION_STRING": os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
        "AZURE_STORAGE_ACCESS_KEY": os.environ.get("AZURE_STORAGE_ACCESS_KEY"),
    }
    envs = {k: v for k, v in envs.items() if v is not None}
    return [], envs


def _get_gcs_artifact_cmd_and_envs(artifact_repo):
    cmds = []
    envs = {}

    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        credentials_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        cmds = ["-v", f"{credentials_path}:/.gcs"]
        envs["GOOGLE_APPLICATION_CREDENTIALS"] = "/.gcs"
    return cmds, envs


def _get_hdfs_artifact_cmd_and_envs(artifact_repo):
    cmds = []
    envs = {
        "MLFLOW_KERBEROS_TICKET_CACHE": MLFLOW_KERBEROS_TICKET_CACHE.get(),
        "MLFLOW_KERBEROS_USER": MLFLOW_KERBEROS_USER.get(),
        "MLFLOW_PYARROW_EXTRA_CONF": MLFLOW_PYARROW_EXTRA_CONF.get(),
    }
    envs = {k: v for k, v in envs.items() if v is not None}

    if "MLFLOW_KERBEROS_TICKET_CACHE" in envs:
        ticket_cache = envs["MLFLOW_KERBEROS_TICKET_CACHE"]
        cmds = ["-v", f"{ticket_cache}:{ticket_cache}"]
    return cmds, envs


_artifact_storages = {
    LocalArtifactRepository: _get_local_artifact_cmd_and_envs,
    S3ArtifactRepository: _get_s3_artifact_cmd_and_envs,
    AzureBlobArtifactRepository: _get_azure_blob_artifact_cmd_and_envs,
    HdfsArtifactRepository: _get_hdfs_artifact_cmd_and_envs,
    GCSArtifactRepository: _get_gcs_artifact_cmd_and_envs,
}


def _get_docker_artifact_storage_cmd_and_envs(artifact_uri):
    artifact_repo = get_artifact_repository(artifact_uri)
    _get_cmd_and_envs = _artifact_storages.get(type(artifact_repo))
    if _get_cmd_and_envs is not None:
        return _get_cmd_and_envs(artifact_repo)
    else:
        return [], {}
