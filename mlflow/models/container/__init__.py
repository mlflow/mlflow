"""
Initialize the environment and start model serving in a Docker container.

To be executed only during the model deployment.

"""
import multiprocessing
import os
import signal
import shutil
from subprocess import check_call, Popen
import sys

from pkg_resources import resource_filename

import mlflow
import mlflow.version

from mlflow import pyfunc, mleap
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.docker_utils import DISABLE_ENV_CREATION
from mlflow.pyfunc import scoring_server, mlserver
from mlflow.version import VERSION as MLFLOW_VERSION

MODEL_PATH = "/opt/ml/model"


DEPLOYMENT_CONFIG_KEY_FLAVOR_NAME = "MLFLOW_DEPLOYMENT_FLAVOR_NAME"

DEFAULT_SAGEMAKER_SERVER_PORT = 8080
DEFAULT_INFERENCE_SERVER_PORT = 8000
DEFAULT_NGINX_SERVER_PORT = 8080
DEFAULT_MLSERVER_PORT = 8080

SUPPORTED_FLAVORS = [pyfunc.FLAVOR_NAME, mleap.FLAVOR_NAME]

DISABLE_NGINX = "DISABLE_NGINX"
ENABLE_MLSERVER = "ENABLE_MLSERVER"

SERVING_ENVIRONMENT = "SERVING_ENVIRONMENT"


def _init(cmd):
    """
    Initialize the container and execute command.

    :param cmd: Command param passed by Sagemaker. Can be  "serve" or "train" (unimplemented).
    """
    if cmd == "serve":
        _serve()
    elif cmd == "train":
        _train()
    else:
        raise Exception(
            "Unrecognized command {cmd}, full args = {args}".format(cmd=cmd, args=str(sys.argv))
        )


def _serve():
    """
    Serve the model.

    Read the MLmodel config, initialize the Conda environment if needed and start python server.
    """
    model_config_path = os.path.join(MODEL_PATH, MLMODEL_FILE_NAME)
    m = Model.load(model_config_path)

    if DEPLOYMENT_CONFIG_KEY_FLAVOR_NAME in os.environ:
        serving_flavor = os.environ[DEPLOYMENT_CONFIG_KEY_FLAVOR_NAME]
    else:
        # Older versions of mlflow may not specify a deployment configuration
        serving_flavor = pyfunc.FLAVOR_NAME

    if serving_flavor == mleap.FLAVOR_NAME:
        _serve_mleap()
    elif pyfunc.FLAVOR_NAME in m.flavors:
        _serve_pyfunc(m)
    else:
        raise Exception("This container only supports models with the MLeap or PyFunc flavors.")


def _install_pyfunc_deps(model_path=None, install_mlflow=False, enable_mlserver=False):
    """
    Creates a conda env for serving the model at the specified path and installs almost all serving
    dependencies into the environment - MLflow is not installed as it's not available via conda.
    """
    # If model is a pyfunc model, create its conda env (even if it also has mleap flavor)
    has_env = False
    if model_path:
        model_config_path = os.path.join(model_path, MLMODEL_FILE_NAME)
        model = Model.load(model_config_path)
        # NOTE: this differs from _serve cause we always activate the env even if you're serving
        # an mleap model
        if pyfunc.FLAVOR_NAME not in model.flavors:
            return
        conf = model.flavors[pyfunc.FLAVOR_NAME]
        if pyfunc.ENV in conf:
            print("creating and activating custom environment")
            env = conf[pyfunc.ENV]
            env_path_dst = os.path.join("/opt/mlflow/", env)
            env_path_dst_dir = os.path.dirname(env_path_dst)
            if not os.path.exists(env_path_dst_dir):
                os.makedirs(env_path_dst_dir)
            shutil.copyfile(os.path.join(MODEL_PATH, env), env_path_dst)
            conda_create_model_env = "conda env create -n custom_env -f {}".format(env_path_dst)
            if Popen(["bash", "-c", conda_create_model_env]).wait() != 0:
                raise Exception("Failed to create model environment.")
            has_env = True
    activate_cmd = ["source /miniconda/bin/activate custom_env"] if has_env else []
    # NB: install gunicorn[gevent] from pip rather than from conda because gunicorn is already
    # dependency of mlflow on pip and we expect mlflow to be part of the environment.
    server_deps = ["gunicorn[gevent]"]
    if enable_mlserver:
        server_deps = ["mlserver", "mlserver-mlflow"]

    install_server_deps = [f"pip install {' '.join(server_deps)}"]
    if Popen(["bash", "-c", " && ".join(activate_cmd + install_server_deps)]).wait() != 0:
        raise Exception("Failed to install serving dependencies into the model environment.")

    if has_env and install_mlflow:
        install_mlflow_cmd = [
            "pip install /opt/mlflow/."
            if _container_includes_mlflow_source()
            else "pip install mlflow=={}".format(MLFLOW_VERSION)
        ]
        if Popen(["bash", "-c", " && ".join(activate_cmd + install_mlflow_cmd)]).wait() != 0:
            raise Exception("Failed to install mlflow into the model environment.")


def _serve_pyfunc(model):
    # option to disable manually nginx. The default behavior is to enable nginx.
    disable_nginx = os.getenv(DISABLE_NGINX, "false").lower() == "true"
    enable_mlserver = os.getenv(ENABLE_MLSERVER, "false").lower() == "true"
    disable_env_creation = os.environ.get(DISABLE_ENV_CREATION) == "true"

    conf = model.flavors[pyfunc.FLAVOR_NAME]
    bash_cmds = []
    if pyfunc.ENV in conf:
        if not disable_env_creation:
            _install_pyfunc_deps(MODEL_PATH, install_mlflow=True, enable_mlserver=enable_mlserver)
        bash_cmds += ["source /miniconda/bin/activate custom_env"]

    procs = []

    start_nginx = True
    if disable_nginx or enable_mlserver:
        start_nginx = False

    if start_nginx:
        nginx_conf = resource_filename(
            mlflow.models.__name__, "container/scoring_server/nginx.conf"
        )

        nginx = Popen(["nginx", "-c", nginx_conf]) if start_nginx else None

        # link the log streams to stdout/err so they will be logged to the container logs.
        # Default behavior is to do the redirection unless explicitly specified
        # by environment variable.
        check_call(["ln", "-sf", "/dev/stdout", "/var/log/nginx/access.log"])
        check_call(["ln", "-sf", "/dev/stderr", "/var/log/nginx/error.log"])

        procs.append(nginx)

    cpu_count = multiprocessing.cpu_count()
    os.system("pip -V")
    os.system("python -V")
    os.system('python -c"from mlflow.version import VERSION as V; print(V)"')

    inference_server = mlserver if enable_mlserver else scoring_server
    # Since MLServer will run without NGINX, expose the server in the `8080`
    # port, which is the assumed "public" port.
    port = DEFAULT_MLSERVER_PORT if enable_mlserver else DEFAULT_INFERENCE_SERVER_PORT
    cmd, cmd_env = inference_server.get_cmd(model_uri=MODEL_PATH, nworkers=cpu_count, port=port)

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
    print("Got sigterm signal, exiting.")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

    sys.exit(0)
