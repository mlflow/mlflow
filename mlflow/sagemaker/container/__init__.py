"""
Initialize the environment and start model serving on Sagemaker or local Docker container.

To be executed only during the model deployment.

"""
from __future__ import print_function

import multiprocessing
import os
import signal
from subprocess import check_call, Popen
import sys

from pkg_resources import resource_filename

import mlflow
import mlflow.version

from mlflow import pyfunc, mleap
from mlflow.models import Model
from mlflow.models.docker_utils import DISABLE_ENV_CREATION
from mlflow.version import VERSION as MLFLOW_VERSION

MODEL_PATH = "/opt/ml/model"

DEPLOYMENT_CONFIG_KEY_FLAVOR_NAME = "MLFLOW_DEPLOYMENT_FLAVOR_NAME"

DEFAULT_SAGEMAKER_SERVER_PORT = 8080

SUPPORTED_FLAVORS = [
    pyfunc.FLAVOR_NAME,
    mleap.FLAVOR_NAME
]


def _init(cmd):
    """
    Initialize the container and execute command.

    :param cmd: Command param passed by Sagemaker. Can be  "serve" or "train" (unimplemented).
    """
    if cmd == 'serve':
        _serve()
    elif cmd == 'train':
        _train()
    else:
        raise Exception("Unrecognized command {cmd}, full args = {args}".format(cmd=cmd,
                                                                                args=str(sys.argv)))


def _install_base_deps(model_path):
    """
    Creates a conda env for serving the model at the specified path and installs almost all serving
    dependencies into the environment - MLflow is not installed as it's not available via conda.
    """
    # If model is a pyfunc model, create its conda env (even if it also has mleap flavor)
    model_config_path = os.path.join(model_path, "MLmodel")
    model = Model.load(model_config_path)
    # NOTE: this differs from _serve cause we always activate the env even if you're serving
    # an mleap model
    if pyfunc.FLAVOR_NAME not in model.flavors:
        return
    conf = model.flavors[pyfunc.FLAVOR_NAME]
    print("creating and activating custom environment")
    env = conf[pyfunc.ENV]
    env_path = os.path.join(model_path, env)
    conda_create_and_install_deps = "conda env create -n custom_env -f {} && " \
                                    "conda install -n custom_env gunicorn gevent".format(env_path)
    install_deps_proc = Popen(["bash", "-c", conda_create_and_install_deps])
    if install_deps_proc.wait() != 0:
        raise Exception("Failed to install server dependencies")


def _install_mlflow_cmds():
    """
    Return commands needed to install MLflow
    :return:
    """
    return [
        "pip install /opt/mlflow/." if _container_includes_mlflow_source()
        else "pip install mlflow=={}".format(MLFLOW_VERSION)
    ]


def _serve():
    """
    Serve the model.

    Read the MLmodel config, initialize the Conda environment if needed and start python server.
    """
    model_config_path = os.path.join(MODEL_PATH, "MLmodel")
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


def _serve_pyfunc(model):
    conf = model.flavors[pyfunc.FLAVOR_NAME]
    bash_cmds = []
    if pyfunc.ENV in conf:
        if not os.environ.get(DISABLE_ENV_CREATION) == "true":
            _install_base_deps(model)
        # TODO don't unconditionally activate env (e.g. for non-pyfunc models)? We might already
        # do this anyways so it could be ok
        bash_cmds += ["source /miniconda/bin/activate custom_env"]
        bash_cmds += _install_mlflow_cmds()
    nginx_conf = resource_filename(mlflow.sagemaker.__name__, "container/scoring_server/nginx.conf")
    nginx = Popen(['nginx', '-c', nginx_conf])
    # link the log streams to stdout/err so they will be logged to the container logs
    check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
    check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])
    cpu_count = multiprocessing.cpu_count()
    os.system("pip -V")
    os.system("python -V")
    os.system('python -c"from mlflow.version import VERSION as V; print(V)"')
    cmd = ("gunicorn --timeout 60 -k gevent -b unix:/tmp/gunicorn.sock -w {nworkers} " +
           "mlflow.sagemaker.container.scoring_server.wsgi:app").format(nworkers=cpu_count)
    bash_cmds.append(cmd)
    gunicorn = Popen(["/bin/bash", "-c", " && ".join(bash_cmds)])
    signal.signal(signal.SIGTERM, lambda a, b: _sigterm_handler(pids=[nginx.pid, gunicorn.pid]))
    # If either subprocess exits, so do we.
    awaited_pids = _await_subprocess_exit_any(procs=[nginx, gunicorn])
    _sigterm_handler(awaited_pids)


def _serve_mleap():
    serve_cmd = ["java", "-cp", "\"/opt/java/jars/*\"", "org.mlflow.sagemaker.ScoringServer",
                 MODEL_PATH, str(DEFAULT_SAGEMAKER_SERVER_PORT)]
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
