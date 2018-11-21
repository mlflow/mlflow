"""
Initialize the environment and start model serving on Sagemaker or local Docker container.

To be executed only during the model deployment.

"""
from __future__ import print_function

import multiprocessing
import os
import shutil
import signal
from subprocess import check_call, Popen
import sys
import yaml

from pkg_resources import resource_filename

import mlflow
import mlflow.version

from mlflow import pyfunc, mleap
from mlflow.models import Model
from mlflow.version import VERSION as MLFLOW_VERSION

MODEL_PATH = "/opt/ml/model"

DEPLOYMENT_CONFIG_KEY_FLAVOR_NAME = "deployment_flavor_name"

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


def _server_dependencies_cmds():
    """
    Get commands required to install packages required to serve the model with MLflow. These are
    packages outside of the user-provided environment, except for the MLflow itself.

    :return: List of commands.
    """
    # TODO: Should we reinstall MLflow? What if there is MLflow in the user's conda environment?
    return ["conda install gunicorn", "conda install gevent",
            "pip install /opt/mlflow/." if _container_includes_mlflow_source()
            else "pip install mlflow=={}".format(MLFLOW_VERSION)]


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
        print("activating custom environment")
        env = conf[pyfunc.ENV]
        env_path_dst = os.path.join("/opt/mlflow/", env)
        env_path_dst_dir = os.path.dirname(env_path_dst)
        if not os.path.exists(env_path_dst_dir):
            os.makedirs(env_path_dst_dir)
        # TODO: should we test that the environment does not include any of the server dependencies?
        # Those are gonna be reinstalled. should probably test this on the client side
        shutil.copyfile(os.path.join(MODEL_PATH, env), env_path_dst)
        os.system("conda env create -n custom_env -f {}".format(env_path_dst))
        bash_cmds += ["source /miniconda/bin/activate custom_env"] + _server_dependencies_cmds()
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
