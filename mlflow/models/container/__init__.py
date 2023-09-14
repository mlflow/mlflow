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
from mlflow import mleap, pyfunc
from mlflow.environment_variables import MLFLOW_DEPLOYMENT_FLAVOR_NAME, MLFLOW_DISABLE_ENV_CREATION
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.pyfunc import _extract_conda_env, mlserver, scoring_server
from mlflow.utils import env_manager as em
from mlflow.utils.virtualenv import _get_or_create_virtualenv
from mlflow.version import VERSION as MLFLOW_VERSION

MODEL_PATH = "/opt/ml/model"


DEFAULT_SAGEMAKER_SERVER_PORT = 8080
DEFAULT_INFERENCE_SERVER_PORT = 8000
DEFAULT_NGINX_SERVER_PORT = 8080
DEFAULT_MLSERVER_PORT = 8080

SUPPORTED_FLAVORS = [pyfunc.FLAVOR_NAME, mleap.FLAVOR_NAME]

DISABLE_NGINX = "DISABLE_NGINX"
ENABLE_MLSERVER = "ENABLE_MLSERVER"

SERVING_ENVIRONMENT = "SERVING_ENVIRONMENT"


_logger = logging.getLogger(__name__)


def _init(cmd, env_manager):
    """
    Initialize the container and execute command.

    :param cmd: Command param passed by Sagemaker. Can be  "serve" or "train" (unimplemented).
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

    # Older versions of mlflow may not specify a deployment configuration
    serving_flavor = MLFLOW_DEPLOYMENT_FLAVOR_NAME.get() or pyfunc.FLAVOR_NAME

    if serving_flavor == mleap.FLAVOR_NAME:
        _serve_mleap()
    elif pyfunc.FLAVOR_NAME in m.flavors:
        _serve_pyfunc(m, env_manager)
    else:
        raise Exception("This container only supports models with the MLeap or PyFunc flavors.")


def _install_pyfunc_deps(
    model_path=None, install_mlflow=False, enable_mlserver=False, env_manager=em.VIRTUALENV
):
    """
    Creates a conda env for serving the model at the specified path and installs almost all serving
    dependencies into the environment - MLflow is not installed as it's not available via conda.
    """
    # If model is a pyfunc model, create its conda env (even if it also has mleap flavor)
    activate_cmd = []
    if model_path:
        model_config_path = os.path.join(model_path, MLMODEL_FILE_NAME)
        model = Model.load(model_config_path)
        # NOTE: this differs from _serve cause we always activate the env even if you're serving
        # an mleap model
        if pyfunc.FLAVOR_NAME not in model.flavors:
            return
        conf = model.flavors[pyfunc.FLAVOR_NAME]
        if pyfunc.ENV in conf:
            _logger.info("creating and activating custom environment")
            env = _extract_conda_env(conf[pyfunc.ENV])
            env_path_dst = os.path.join("/opt/mlflow/", env)
            env_path_dst_dir = os.path.dirname(env_path_dst)
            if not os.path.exists(env_path_dst_dir):
                os.makedirs(env_path_dst_dir)
            shutil.copyfile(os.path.join(MODEL_PATH, env), env_path_dst)
            if env_manager == em.CONDA:
                conda_create_model_env = f"conda env create -n custom_env -f {env_path_dst}"
                if Popen(["bash", "-c", conda_create_model_env]).wait() != 0:
                    raise Exception("Failed to create model environment.")
                activate_cmd = ["source /miniconda/bin/activate custom_env"]
            elif env_manager == em.VIRTUALENV:
                env_activate_cmd = _get_or_create_virtualenv(model_path)
                path = env_activate_cmd.split(" ")[-1]
                os.symlink(path, "/opt/activate")
                activate_cmd = [env_activate_cmd]

    # NB: install gunicorn[gevent] from pip rather than from conda because gunicorn is already
    # dependency of mlflow on pip and we expect mlflow to be part of the environment.
    server_deps = ["gunicorn[gevent]"]
    if enable_mlserver:
        server_deps = ["'mlserver>=1.2.0,!=1.3.1'", "'mlserver-mlflow>=1.2.0,!=1.3.1'"]

    install_server_deps = [f"pip install {' '.join(server_deps)}"]
    if Popen(["bash", "-c", " && ".join(activate_cmd + install_server_deps)]).wait() != 0:
        raise Exception("Failed to install serving dependencies into the model environment.")

    if len(activate_cmd) and install_mlflow:
        install_mlflow_cmd = [
            "pip install /opt/mlflow/."
            if _container_includes_mlflow_source()
            else f"pip install mlflow=={MLFLOW_VERSION}"
        ]
        if Popen(["bash", "-c", " && ".join(activate_cmd + install_mlflow_cmd)]).wait() != 0:
            raise Exception("Failed to install mlflow into the model environment.")
    return activate_cmd


def _serve_pyfunc(model, env_manager):
    # option to disable manually nginx. The default behavior is to enable nginx.
    disable_nginx = os.getenv(DISABLE_NGINX, "false").lower() == "true"
    enable_mlserver = os.getenv(ENABLE_MLSERVER, "false").lower() == "true"
    disable_env_creation = MLFLOW_DISABLE_ENV_CREATION.get()

    conf = model.flavors[pyfunc.FLAVOR_NAME]
    bash_cmds = []
    if pyfunc.ENV in conf:
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
    if enable_mlserver:
        inference_server = mlserver
        # Allows users to choose the number of workers using MLServer var env settings.
        # Default to cpu count
        nworkers = int(os.getenv("MLSERVER_INFER_WORKERS", cpu_count))
        # Since MLServer will run without NGINX, expose the server in the `8080`
        # port, which is the assumed "public" port.
        port = DEFAULT_MLSERVER_PORT
    else:
        inference_server = scoring_server
        # users can use GUNICORN_CMD_ARGS="--workers=3" var env to override the number of workers
        nworkers = cpu_count
        port = DEFAULT_INFERENCE_SERVER_PORT

    cmd, cmd_env = inference_server.get_cmd(model_uri=MODEL_PATH, nworkers=nworkers, port=port)

    bash_cmds.append(cmd)
    inference_server_process = Popen(["/bin/bash", "-c", " && ".join(bash_cmds)], env=cmd_env)
    procs.append(inference_server_process)

    signal.signal(signal.SIGTERM, lambda a, b: _sigterm_handler(pids=[p.pid for p in procs]))
    # If either subprocess exits, so do we.
    awaited_pids = _await_subprocess_exit_any(procs=procs)
    _sigterm_handler(awaited_pids)


def _serve_mleap():
    serve_cmd = [
        "java",
        "-cp",
        '"/opt/java/jars/*"',
        "org.mlflow.sagemaker.ScoringServer",
        MODEL_PATH,
        str(DEFAULT_SAGEMAKER_SERVER_PORT),
    ]
    # Invoke `Popen` with a single string command in the shell to support wildcard usage
    # with the mlflow jar version.
    serve_cmd = " ".join(serve_cmd)
    mleap = Popen(serve_cmd, shell=True)
    signal.signal(signal.SIGTERM, lambda a, b: _sigterm_handler(pids=[mleap.pid]))
    awaited_pids = _await_subprocess_exit_any(procs=[mleap])
    _sigterm_handler(awaited_pids)


def _container_includes_mlflow_source():
    return os.path.exists("/opt/mlflow/setup.py")


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
