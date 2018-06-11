from __future__ import print_function

import multiprocessing
import os
import shutil
import signal
import sys
import subprocess

from pkg_resources import resource_filename

import mlflow
from mlflow import pyfunc
from mlflow.models import Model


cpu_count = multiprocessing.cpu_count()
model_server_timeout = os.environ.get('MODEL_SERVER_TIMEOUT', 60)
model_server_workers = int(os.environ.get('MODEL_SERVER_WORKERS', cpu_count))


def _scoring_server_init(cmd):
    if cmd == 'serve':
        _serve()
    elif cmd == 'train':
        _train()
    elif cmd.startswith("dev_"):
        # dev-mode: re-install mlflow to ensure we have the latest version
        print("re-installing mlflow")
        os.system("pip install -e /opt/mlflow/.")
        _scoring_server_init(cmd[4:])
    else:
        raise Exception("Unrecognized command {cmd}, full args = {args}".format(cmd=cmd,
                                                                                args=str(sys.argv)))


def _install_server_dependencies():
    os.system("pip install pandas")
    os.system("pip install Flask")
    os.system("conda install - c anaconda gunicorn")
    os.system("conda install - c anaconda gevent")


def _serve():
    m = Model.load("/opt/ml/model/MLmodel")
    if pyfunc.FLAVOR_NAME not in m.flavors:
        raise Exception("Currently can only deal with pyfunc models and this is not one.")
    conf = m.flavors[pyfunc.FLAVOR_NAME]
    if pyfunc.ENV in conf:
        env = conf[pyfunc.ENV]
        print("activating conda environment {}".format(env))
        env_path_dst = os.path.join("/opt/mlflow/", env)
        shutil.copy(src=os.path.join("/opt/ml/model/", env), dst=env_path_dst)
        os.system("conda create -n custom_env -f {}".format(env_path_dst))
        os.system("source activate custom_env")
        # make sure we have all dependencies needed to run the server
        _install_server_dependencies()
    _start_server()


def _start_server():
    print('Starting the inference server with {} workers.'.format(model_server_workers))

    # link the log streams to stdout/err so they will be logged to the container logs
    subprocess.check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
    subprocess.check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])
    nginx_conf = resource_filename(mlflow.sagemaker.__name__, "container/scoring_server/nginx.conf")

    nginx = subprocess.Popen(['nginx', '-c', nginx_conf])
    gunicorn = subprocess.Popen(['gunicorn',
                                 '--timeout', str(model_server_timeout),
                                 '-k', 'gevent',
                                 '-b', 'unix:/tmp/gunicorn.sock',
                                 '-w', str(model_server_workers),
                                 'mlflow.sagemaker.container.scoring_server.wsgi:app'])

    signal.signal(signal.SIGTERM, lambda a, b: _sigterm_handler(nginx.pid, gunicorn.pid))

    # If either subprocess exits, so do we.
    pids = set([nginx.pid, gunicorn.pid])
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break

    _sigterm_handler(nginx.pid, gunicorn.pid)
    print('Inference server exiting')


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
