"""
Code to initializae the environment and start model serving on Sagemaker or local docker container.

This code is to be executed only during the model deployment.

"""
from __future__ import print_function

import multiprocessing
import os
import shutil
import signal
from subprocess import check_call, Popen
import sys


from pkg_resources import resource_filename

import mlflow
import mlflow.version

from mlflow import pyfunc
from mlflow.models import Model

_dev_flag = False


def _init(cmd):
    """
    Initialize the container and execute command.

    :param cmd: Command param passed by Sagemaker. Can be  "serve" or "train" (unimplemented).
    When running locally, it can have dev_ prefix which will reinstall mlflow from local directory.
    """
    if cmd == 'serve':
        _serve()
    elif cmd == 'train':
        _train()
    elif cmd.startswith("dev_"):
        # dev-mode: re-install mlflow to ensure we have the latest version
        os.system("pip install -e /opt/mlflow/.")
        global _dev_flag
        _dev_flag = True
        _init(cmd[4:])
    else:
        raise Exception("Unrecognized command {cmd}, full args = {args}".format(cmd=cmd,
                                                                                args=str(sys.argv)))


def _server_dependencies_cmds():
    return ["conda install -c anaconda gunicorn", "conda install -c anaconda gevent",
            "pip install -e /opt/mlflow/." if _dev_flag else
            "pip install mlflow=={}".format(mlflow.version.version)]


def _serve():
    m = Model.load("/opt/ml/model/MLmodel")
    if pyfunc.FLAVOR_NAME not in m.flavors:
        raise Exception("Currently can only deal with pyfunc models and this is not one.")
    conf = m.flavors[pyfunc.FLAVOR_NAME]
    bash_cmds = []
    if pyfunc.ENV in conf:
        env = conf[pyfunc.ENV]
        env_path_dst = os.path.join("/opt/mlflow/", env)
        # /opt/ml/ is read-only, we need to copy the env elsewhere before importing it
        shutil.copy(src=os.path.join("/opt/ml/model/", env), dst=env_path_dst)
        os.system("conda env create -n custom_env -f {}".format(env_path_dst))
        bash_cmds += ["source /miniconda/bin/activate custom_env"] + _server_dependencies_cmds()
    nginx_conf = resource_filename(mlflow.sagemaker.__name__, "container/scoring_server/nginx.conf")
    nginx = Popen(['nginx', '-c', nginx_conf])
    # link the log streams to stdout/err so they will be logged to the container logs
    check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
    check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])
    cpu_count = multiprocessing.cpu_count()
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


def _train():
    raise Exception("Train is not implemented.")


def _sigterm_handler(nginx_pid, gunicorn_pid):
    try:
        os.kill(nginx_pid, signal.SIGQUIT)
    except OSError:
        pass
    try:
        os.kill(gunicorn_pid, signal.SIGTERM)
    except OSError:
        pass

    sys.exit(0)
