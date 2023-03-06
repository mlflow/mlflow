"""
Initialize the environment and start model serving in a Docker container.

To be executed only during the model deployment.

"""
import multiprocessing
import os
import signal
import shutil
import string
from subprocess import check_call, Popen
import sys
import tempfile
import typing
import logging

from pkg_resources import resource_filename

import mlflow
import mlflow.version

from mlflow import pyfunc, mleap
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.docker_utils import DISABLE_ENV_CREATION
from mlflow.pyfunc import scoring_server, mlserver, _extract_conda_env
from mlflow.version import VERSION as MLFLOW_VERSION
from mlflow.utils import env_manager as em
from mlflow.utils.virtualenv import _get_or_create_virtualenv

MODEL_PATH = "/opt/ml/model"


DEPLOYMENT_CONFIG_KEY_FLAVOR_NAME = "MLFLOW_DEPLOYMENT_FLAVOR_NAME"

PRIMARY_HOST_ADDRESS = "0.0.0.0"
UPSTREAM_HOST_ADDRESS = "127.0.0.1"

DEFAULT_PRIMARY_PORT = 8080
DEFAULT_UPSTREAM_PORT = 8000

SAGEMAKER_BIND_TO_PORT = "SAGEMAKER_BIND_TO_PORT"
SAGEMAKER_SAFE_PORT_RANGE = "SAGEMAKER_SAFE_PORT_RANGE"

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
        raise Exception(
            "Unrecognized command {cmd}, full args = {args}".format(cmd=cmd, args=str(sys.argv))
        )


def _serve(env_manager):
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
        # NOTE: Pin version until the next MLflow release (otherwise it won't
        # pick up the requirements on the current's `setup.py`)
        server_deps = ["'mlserver>=1.2.0.dev13'", "'mlserver-mlflow>=1.2.0.dev13'"]

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
    disable_env_creation = os.environ.get(DISABLE_ENV_CREATION) == "true"

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

    upstream_host = PRIMARY_HOST_ADDRESS
    primary_port = _derive_primary_port(DEFAULT_PRIMARY_PORT)
    upstream_port = _derive_upstream_port(primary_port, DEFAULT_UPSTREAM_PORT)

    if start_nginx:
        nginx_conf = resource_filename(
            mlflow.models.__name__, "container/scoring_server/nginx.conf"
        )

        upstream_host = UPSTREAM_HOST_ADDRESS
        nginx_conf = _interpolate_nginx_config(
            nginx_conf,
            upstream_host,
            primary_port,
            upstream_port,
        )

        nginx = Popen(["nginx", "-c", nginx_conf]) if start_nginx else None

        # link the log streams to stdout/err so they will be logged to the container logs.
        # Default behavior is to do the redirection unless explicitly specified
        # by environment variable.
        check_call(["ln", "-sf", "/dev/stdout", "/var/log/nginx/access.log"])
        check_call(["ln", "-sf", "/dev/stderr", "/var/log/nginx/error.log"])

        procs.append(nginx)

    cpu_count = multiprocessing.cpu_count()

    inference_server = mlserver if enable_mlserver else scoring_server
    # Since MLServer will run without NGINX, expose the server in the `8080`
    # port, which is the assumed "public" port.
    port = primary_port if disable_nginx else upstream_port
    cmd, cmd_env = inference_server.get_cmd(
        model_uri=MODEL_PATH,
        nworkers=cpu_count,
        host=upstream_host,
        port=port,
    )

    bash_cmds.append(cmd)
    inference_server_process = Popen(["/bin/bash", "-c", " && ".join(bash_cmds)], env=cmd_env)
    procs.append(inference_server_process)

    signal.signal(signal.SIGTERM, lambda a, b: _sigterm_handler(pids=[p.pid for p in procs]))
    # If either subprocess exits, so do we.
    awaited_pids = _await_subprocess_exit_any(procs=procs)
    _sigterm_handler(awaited_pids)


def _serve_mleap():
    primary_port = _derive_primary_port(DEFAULT_PRIMARY_PORT)
    serve_cmd = [
        "java",
        "-cp",
        '"/opt/java/jars/*"',
        "org.mlflow.sagemaker.ScoringServer",
        MODEL_PATH,
        str(primary_port),
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


def _derive_primary_port(default_port: int) -> int:
    """
    Returns SAGEMAKER_BIND_TO_PORT environment variable value, if present, or a given default port.
    """
    return int(os.getenv(SAGEMAKER_BIND_TO_PORT, default_port))


def _derive_upstream_port(busy_port: int, default_port: int) -> int:
    """
    Returns port from SAGEMAKER_SAFE_PORT_RANGE environment variable value, if present,
    or a given default port. SAGEMAKER_SAFE_PORT_RANGE specifies the value as an inclusive
    range in the format "XXXX-YYYY".

    :param busy_port: denotes an already occupied primary port;
        the returned value must not coincide with this value.
    """
    port_range = os.getenv(SAGEMAKER_SAFE_PORT_RANGE)
    if port_range is None:
        return default_port
    else:
        lower_bound, upper_bound = _parse_sagemaker_safe_port_range(port_range)
        port = _select_port_from_range(lower_bound, upper_bound, busy_port)
        return port


def _select_port_from_range(lower_bound: int, upper_bound: int, busy_port: int) -> int:
    """
    Returns a port value within lower and upper bounds excluding a given busy port.
    """
    port = lower_bound
    if port != busy_port:
        return port
    else:
        port += 1
        if port > upper_bound:
            # in case when lower_bound == busy_port AND lower_bound == upper_bound
            raise ValueError(
                f"Could not find a vacant port within an inclusive range"
                f" '{lower_bound}'-'{upper_bound}' and a busy port '{busy_port}'."
            )
        return port


def _parse_sagemaker_safe_port_range(port_range: str) -> typing.Tuple[int, int]:
    """
    Parses values range string in "XXXX-YYYY" format (SAGEMAKER_SAFE_PORT_RANGE environment
    variable, if present) and returns a lower bound (XXXX) and an upper bound (YYYY) values.
    """
    lower_bound, upper_bound = port_range.split("-", maxsplit=1)
    return int(lower_bound), int(upper_bound)


def _interpolate_nginx_config(
    config_path: str,
    upstream_host: str,
    primary_port: int,
    upstream_port: int,
) -> str:
    """
    Reads the original nginx config file, given as config_path, interpolates
    upstream_host, primary_port, upstream_port values, writes an updated
    nginx config file to a temporary destination, and returns its path.
    """
    with open(config_path) as file:
        nginx_conf_content = file.read()

    nginx_conf_content = string.Template(nginx_conf_content).safe_substitute(
        upstream_host=upstream_host,
        primary_port=primary_port,
        upstream_port=upstream_port,
    )

    with tempfile.NamedTemporaryFile("w+", delete=False) as fp:
        fp.write(nginx_conf_content)

    return fp.name
