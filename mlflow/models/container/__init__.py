"""
Initialize the environment and start model serving in a Docker container.

To be executed only during the model deployment.

"""

import logging
import multiprocessing
import os
import shutil
import signal
import sys
from pathlib import Path
from subprocess import Popen, check_call

import mlflow
import mlflow.version
from mlflow import pyfunc
from mlflow.environment_variables import MLFLOW_DISABLE_ENV_CREATION
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.pyfunc import _extract_conda_env, mlserver, scoring_server
from mlflow.store.artifact.models_artifact_repo import REGISTERED_MODEL_META_FILE_NAME
from mlflow.utils import env_manager as em
from mlflow.utils.environment import _PythonEnv
from mlflow.utils.virtualenv import _get_or_create_virtualenv
from mlflow.utils.yaml_utils import read_yaml
from mlflow.version import VERSION as MLFLOW_VERSION

MODEL_PATH = "/opt/ml/model"


DEFAULT_SAGEMAKER_SERVER_PORT = 8080
DEFAULT_INFERENCE_SERVER_PORT = 8000
DEFAULT_NGINX_SERVER_PORT = 8080
DEFAULT_MLSERVER_PORT = 8080

SUPPORTED_FLAVORS = [pyfunc.FLAVOR_NAME]

DISABLE_NGINX = "DISABLE_NGINX"
ENABLE_MLSERVER = "ENABLE_MLSERVER"

SERVING_ENVIRONMENT = "SERVING_ENVIRONMENT"


_logger = logging.getLogger(__name__)


def _init(cmd, env_manager):
    """
    Initialize the container and execute command.

    Args:
        cmd: Command param passed by Sagemaker. Can be "serve" or "train" (unimplemented).
    """
    if cmd == "serve":
        _serve(env_manager)
    elif cmd == "train":
        _train()
    else:
        raise Exception(f"Unrecognized command {cmd}, full args = {sys.argv}")


def _serve(env_manager):
    """
    Serve the model.

    Read the MLmodel config, initialize the Conda environment if needed and start python server.
    """
    model_config_path = os.path.join(MODEL_PATH, MLMODEL_FILE_NAME)
    m = Model.load(model_config_path)

    if pyfunc.FLAVOR_NAME in m.flavors:
        _serve_pyfunc(m, env_manager)
    else:
        raise Exception("This container only supports models with the PyFunc flavors.")


def _install_pyfunc_deps(
    model_path=None, install_mlflow=False, enable_mlserver=False, env_manager=em.VIRTUALENV
):
    """
    Creates a conda env for serving the model at the specified path and installs almost all serving
    dependencies into the environment - MLflow is not installed as it's not available via conda.
    """
    activate_cmd = _install_model_dependencies_to_env(model_path, env_manager) if model_path else []

    # NB: install gunicorn[gevent] from pip rather than from conda because gunicorn is already
    # dependency of mlflow on pip and we expect mlflow to be part of the environment.
    server_deps = ["gunicorn[gevent]"]
    if enable_mlserver:
        server_deps = [
            "'mlserver>=1.2.0,!=1.3.1,<2.0.0'",
            "'mlserver-mlflow>=1.2.0,!=1.3.1,<2.0.0'",
        ]

    install_server_deps = [f"pip install {' '.join(server_deps)}"]
    if Popen(["bash", "-c", " && ".join(activate_cmd + install_server_deps)]).wait() != 0:
        raise Exception("Failed to install serving dependencies into the model environment.")

    # NB: If we don't use virtualenv or conda env, we don't need to install mlflow here as
    # it's already installed in the container.
    if len(activate_cmd):
        if _container_includes_mlflow_source():
            # If the MLflow source code is copied to the container,
            # we always need to run `pip install /opt/mlflow` otherwise
            # the MLflow dependencies are not installed.
            install_mlflow_cmd = ["pip install /opt/mlflow/."]
        elif install_mlflow:
            install_mlflow_cmd = [f"pip install mlflow=={MLFLOW_VERSION}"]
        else:
            install_mlflow_cmd = []

        if install_mlflow_cmd:
            if Popen(["bash", "-c", " && ".join(activate_cmd + install_mlflow_cmd)]).wait() != 0:
                raise Exception("Failed to install mlflow into the model environment.")
    return activate_cmd


def _install_model_dependencies_to_env(model_path, env_manager) -> list[str]:
    """:
    Installs model dependencies to the specified environment, which can be either a local
    environment, a conda environment, or a virtualenv.

    Returns:
        Empty list if local environment, otherwise a list of bash commands to activate the
        virtualenv or conda environment.
    """
    model_config_path = os.path.join(model_path, MLMODEL_FILE_NAME)
    model = Model.load(model_config_path)

    conf = model.flavors.get(pyfunc.FLAVOR_NAME, {})
    if pyfunc.ENV not in conf:
        return []
    env_conf = conf[mlflow.pyfunc.ENV]

    if env_manager == em.LOCAL:
        # Install pip dependencies directly into the local environment
        python_env_config_path = os.path.join(model_path, env_conf[em.VIRTUALENV])
        python_env = _PythonEnv.from_yaml(python_env_config_path)
        deps = " ".join(python_env.build_dependencies + python_env.dependencies)
        deps = deps.replace("requirements.txt", os.path.join(model_path, "requirements.txt"))
        if Popen(["bash", "-c", f"python -m pip install {deps}"]).wait() != 0:
            raise Exception("Failed to install model dependencies.")
        return []

    _logger.info("creating and activating custom environment")

    env = _extract_conda_env(env_conf)
    env_path_dst = os.path.join("/opt/mlflow/", env)
    env_path_dst_dir = os.path.dirname(env_path_dst)
    if not os.path.exists(env_path_dst_dir):
        os.makedirs(env_path_dst_dir)
    shutil.copy2(os.path.join(MODEL_PATH, env), env_path_dst)

    if env_manager == em.CONDA:
        conda_create_model_env = f"conda env create -n custom_env -f {env_path_dst}"
        if Popen(["bash", "-c", conda_create_model_env]).wait() != 0:
            raise Exception("Failed to create model environment.")
        activate_cmd = ["source /miniconda/bin/activate custom_env"]

    elif env_manager == em.VIRTUALENV:
        env_activate_cmd = _get_or_create_virtualenv(model_path, env_manager=env_manager)
        path = env_activate_cmd.split(" ")[-1]
        os.symlink(path, "/opt/activate")
        activate_cmd = [env_activate_cmd]

    return activate_cmd


def _serve_pyfunc(model, env_manager):
    # option to disable manually nginx. The default behavior is to enable nginx.
    disable_nginx = os.getenv(DISABLE_NGINX, "false").lower() == "true"
    enable_mlserver = os.getenv(ENABLE_MLSERVER, "false").lower() == "true"
    disable_env_creation = MLFLOW_DISABLE_ENV_CREATION.get()

    conf = model.flavors[pyfunc.FLAVOR_NAME]
    bash_cmds = []
    if pyfunc.ENV in conf:
        # NB: MLFLOW_DISABLE_ENV_CREATION is False only for SageMaker deployment, where the model
        # files are loaded into the container at runtime rather than build time. In this case,
        # we need to create a virtual environment and install the model dependencies into it when
        # starting the container.
        if not disable_env_creation:
            _install_pyfunc_deps(
                MODEL_PATH,
                install_mlflow=True,
                enable_mlserver=enable_mlserver,
                env_manager=env_manager,
            )
        if env_manager == em.CONDA:
            bash_cmds.append("source /miniconda/bin/activate custom_env")
        elif env_manager == em.VIRTUALENV:
            bash_cmds.append("source /opt/activate")
    procs = []

    start_nginx = True
    if disable_nginx or enable_mlserver:
        start_nginx = False

    if start_nginx:
        nginx_conf = Path(mlflow.models.__file__).parent.joinpath(
            "container", "scoring_server", "nginx.conf"
        )

        nginx = Popen(["nginx", "-c", nginx_conf]) if start_nginx else None

        # link the log streams to stdout/err so they will be logged to the container logs.
        # Default behavior is to do the redirection unless explicitly specified
        # by environment variable.
        check_call(["ln", "-sf", "/dev/stdout", "/var/log/nginx/access.log"])
        check_call(["ln", "-sf", "/dev/stderr", "/var/log/nginx/error.log"])

        procs.append(nginx)

    cpu_count = multiprocessing.cpu_count()
    inference_server_kwargs = {}
    if enable_mlserver:
        inference_server = mlserver
        # Allows users to choose the number of workers using MLServer var env settings.
        # Default to cpu count
        nworkers = int(os.getenv("MLSERVER_INFER_WORKERS", cpu_count))
        # Since MLServer will run without NGINX, expose the server in the `8080`
        # port, which is the assumed "public" port.
        port = DEFAULT_MLSERVER_PORT

        model_meta = _read_registered_model_meta(MODEL_PATH)
        model_dict = model.to_dict()
        inference_server_kwargs = {
            "model_name": model_meta.get("model_name"),
            "model_version": model_meta.get(
                "model_version", model_dict.get("run_id", model_dict.get("model_uuid"))
            ),
        }
    else:
        inference_server = scoring_server
        nworkers = int(os.getenv("MLFLOW_MODELS_WORKERS", cpu_count))
        port = DEFAULT_INFERENCE_SERVER_PORT

    cmd, cmd_env = inference_server.get_cmd(
        model_uri=MODEL_PATH, nworkers=nworkers, port=port, **inference_server_kwargs
    )

    bash_cmds.append(cmd)
    inference_server_process = Popen(["/bin/bash", "-c", " && ".join(bash_cmds)], env=cmd_env)
    procs.append(inference_server_process)

    signal.signal(signal.SIGTERM, lambda a, b: _sigterm_handler(pids=[p.pid for p in procs]))
    # If either subprocess exits, so do we.
    awaited_pids = _await_subprocess_exit_any(procs=procs)
    _sigterm_handler(awaited_pids)


def _read_registered_model_meta(model_path):
    model_meta = {}
    if os.path.isfile(os.path.join(model_path, REGISTERED_MODEL_META_FILE_NAME)):
        model_meta = read_yaml(model_path, REGISTERED_MODEL_META_FILE_NAME)

    return model_meta


def _container_includes_mlflow_source():
    return os.path.exists("/opt/mlflow/pyproject.toml")


def _train():
    raise Exception("Train is not implemented.")


def _await_subprocess_exit_any(procs):
    pids = [proc.pid for proc in procs]
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break
    return pids


def _sigterm_handler(pids):
    """
    Cleanup when terminating.

    Attempt to kill all launched processes and exit.

    """
    _logger.info("Got sigterm signal, exiting.")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

    sys.exit(0)
