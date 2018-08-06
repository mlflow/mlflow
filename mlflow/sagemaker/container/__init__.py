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
from collections import namedtuple

from pkg_resources import resource_filename

import mlflow
import mlflow.version

from mlflow import pyfunc
from mlflow.models import Model
from mlflow.utils import PYTHON_VERSION
from mlflow.utils.logging_utils import print_flush
from mlflow.version import VERSION as MLFLOW_VERSION

# The default Anaconda environment is active when this module is imported,
# so `PYTHON_VERSION` is the correct version of the default environment
DEFAULT_CONDA_PYTHON_VERSION = PYTHON_VERSION

# Supported versions are listed at https://conda.io/docs/user-guide/tasks/manage-python.html
SUPPORTED_CONDA_MAJOR_PY_VERSIONS = ['2.7', '3.4', '3.5', '3.6']

CondaEnv = namedtuple("CondaEnv", ["name", "config", "py_version"])

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
    return ["conda install -c anaconda gunicorn", "conda install -c anaconda gevent",
            "pip install /opt/mlflow/." if os.path.isdir("/opt/mlflow")
            else "pip install mlflow=={}".format(MLFLOW_VERSION)]


def _serve():
    """
    Serve the model.

    Read the MLmodel config, initialize the Conda environment if needed and start python server.
    """
    m = Model.load("/opt/ml/model/MLmodel")
    if pyfunc.FLAVOR_NAME not in m.flavors:
        raise Exception("Only supports pyfunc models and this is not one.")
    model_conf = m.flavors[pyfunc.FLAVOR_NAME]

    bash_cmds = []
    conda_env = _get_conda_env(model_conf)
    if conda_env is not None:
        _create_conda_env(conda_env)
        bash_cmds += ["source /miniconda/bin/activate {en}".format(en=conda_env.name)] + \
                _server_dependencies_cmds()

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
    gunicorn = Popen(["/bin/bash", "-c", "; ".join(bash_cmds)])
    signal.signal(signal.SIGTERM, lambda a, b: _sigterm_handler(nginx.pid, gunicorn.pid))
    # If either subprocess exits, so do we.
    pids = set([nginx.pid, gunicorn.pid])
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break
    _sigterm_handler(nginx.pid, gunicorn.pid)


def _get_conda_env(model_config):
    model_py_version = None
    if pyfunc.PY_VERSION in model_config:
        model_py_version = model_config[pyfunc.PY_VERSION]

    if pyfunc.ENV in model_config:
        print_flush("Activating custom Anaconda environment")
        env = model_config[pyfunc.ENV]
        env_path_src = os.path.join("/opt/ml/model", env)
        _add_py_version_to_conda_env_if_necessary(env_path_src, model_py_version)
        env_path_dst = os.path.join("/opt/mlflow/", env)
        env_path_dst_dir = os.path.dirname(env_path_dst)
        if not os.path.exists(env_path_dst_dir):
            os.makedirs(env_path_dst_dir)
        # TODO: should we test that the environment does not include any of the server dependencies?
        # Those are gonna be reinstalled. should probably test this on the client side
        shutil.copyfile(env_path_src, env_path_dst)
        return CondaEnv(name="custom_env", config=env, py_version=None)
    elif model_py_version is None:
        print_flush("The model does not specify a Python version or a custom Anaconda environment"
                    " to use.")
        _warn_potentially_incompatible_conda_env()
        return None
    elif not _conda_supports_py_version(model_py_version):
        print_flush("The version of python used to serialize the model, Python {mpyv}, is not"
                     " supported by Anaconda.".format(mpyv=model_py_version))
        _warn_potentially_incompatible_conda_env()
        return None
    elif model_py_version == PYTHON_VERSION:
        print_flush("The model's Python version matches the default environment's"
                    " Python version: {dpyv}. Using the default environment.".format(
                        dpyv=DEFAULT_CONDA_PYTHON_VERSION))
        return None
    else:
        print_flush("The model's Python version is {mpyv}. Activating Anaconda environment with"
                    " Python version {mpyv}.".format(mpyv=model_py_version))
        return CondaEnv(name="py_env", config=None, py_version=model_py_version)


def _add_py_version_to_conda_env_if_necessary(env_path, model_py_version):
    try:
        with open(env_path, "r") as f:
            env_yaml = yaml.load(f)

        deps = env_yaml["dependencies"]
        env_py_versions = list(filter(
            lambda item : "=" in item and item.split("=")[0].lower() == "python", deps))
        env_contains_python = len(env_py_versions) > 0
        if env_contains_python:
            env_py_version = env_py_versions[-1].split("=")[1]
            print_flush("Using Python version specified by the custom Anaconda environment:" 
                        " Python {epyv}.".format(epyv=env_py_version))
        else:
            print_flush("The custom Anaconda environment does not specify a version of Python to use.")
            if model_py_version is not None:
                print_flush("Using the version of the Python associated with the model:" 
                            " Python {mpyv}".format(mpyv=model_py_version))
                deps.append("python={mpyv}".format(mpyv=model_py_version))
            else:
                deps.append("python={dpyv}".format(dpyv=DEFAULT_CONDA_PYTHON_VERSION))
                _warn_potentially_incompatible_conda_env()

        with open(env_path, "w") as out:
            yaml.dump(env_yaml, out, default_flow_style=False)
    except Exception as e:
        print(e)


def _create_conda_env(env):
    if env.config is not None:
        cmd = "conda env create -n {en} -f {cfg}".format(en=env.name, cfg=env.config)
    elif env.py_version is not None:
        cmd = "conda create -n {en} python={pyv}".format(en=env.name, pyv=env.py_version)
    os.system(cmd)


def _conda_supports_py_version(py_version):
    major_version = ".".join(py_version.split(".")[:2])
    return (major_version in SUPPORTED_CONDA_MAJOR_PY_VERSIONS)


def _warn_potentially_incompatible_conda_env():
    """
    Prints a warning message indicating that the default Anaconda environment,
    which may not be compatible with the model, will be used for serving.
    """
    print_flush("Using the default Anaconda environment with Python {dpyv}, which may not be"
                 " compatible with the model.".format(dpyv=DEFAULT_CONDA_PYTHON_VERSION))


def _train():
    raise Exception("Train is not implemented.")


def _sigterm_handler(nginx_pid, gunicorn_pid):
    """
    Cleanup when terminating.

    Attempt to kill all launched processes and exit.

    """
    print("Got sigterm signal, exiting.")
    try:
        os.kill(nginx_pid, signal.SIGQUIT)
    except OSError:
        pass
    try:
        os.kill(gunicorn_pid, signal.SIGTERM)
    except OSError:
        pass

    sys.exit(0)
