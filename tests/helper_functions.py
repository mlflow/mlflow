import os
import random
import re
import requests
import string
import time
import signal
from subprocess import Popen, PIPE, STDOUT

import pandas as pd

import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.pyfunc


def random_int(lo=1, hi=1e10):
    return random.randint(lo, hi)


def random_str(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def random_file(ext):
    return "temp_test_%d.%s" % (random_int(), ext)


def score_model_in_sagemaker_docker_container(
        model_path, data, content_type, flavor=mlflow.pyfunc.FLAVOR_NAME,
        activity_polling_timeout_seconds=500):
    """
    :param model_path: Path to the model to be served.
    :param data: The data to send to the docker container for testing. This is either a
                 Pandas dataframe or string of the format specified by `content_type`.
    :param content_type: The type of the data to send to the docker container for testing. This is
                         one of `mlflow.pyfunc.scoring_server.CONTENT_TYPES`.
    :param flavor: Model flavor to be deployed.
    :param activity_polling_timeout_seconds: The amount of time, in seconds, to wait before
                                             declaring the scoring process to have failed.
    """
    env = dict(os.environ)
    env.update(LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8")
    proc = _start_scoring_proc(
            cmd=['mlflow', 'sagemaker', 'run-local', '-m', model_path, '-p', "5000", "-f", flavor],
            env=env)
    return _evaluate_scoring_proc(proc, 5000, data, content_type, activity_polling_timeout_seconds)


def pyfunc_serve_and_score_model(
        model_path, data, content_type, activity_polling_timeout_seconds=500):
    """
    :param model_path: Path to the model to be served.
    :param data: The data to send to the pyfunc server for testing. This is either a
                 Pandas dataframe or string of the format specified by `content_type`.
    :param content_type: The type of the data to send to the pyfunc server for testing. This is
                         one of `mlflow.pyfunc.scoring_server.CONTENT_TYPES`.
    :param activity_polling_timeout_seconds: The amount of time, in seconds, to wait before
                                             declaring the scoring process to have failed.
    """
    env = dict(os.environ)
    env.update(LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8")
    proc = _start_scoring_proc(
            cmd=['mlflow', 'pyfunc', 'serve', '-m', model_path, "-p", "0"],
            env=env)
    for x in iter(proc.stdout.readline, ""):
        print(x)
        m = re.match(pattern=".*Running on http://127.0.0.1:(\\d+).*", string=x)
        if m:
            return _evaluate_scoring_proc(
                    proc, int(m.group(1)), data, content_type, activity_polling_timeout_seconds)

    raise Exception("Failed to start server")


def _start_scoring_proc(cmd, env):
    proc = Popen(cmd,
                 stdout=PIPE,
                 stderr=STDOUT,
                 universal_newlines=True,
                 env=env,
                 # Assign the scoring process to a process group. All child processes of the
                 # scoring process will be assigned to this group as well. This allows child
                 # processes of the scoring process to be terminated successfully
                 preexec_fn=os.setsid)
    return proc


def _evaluate_scoring_proc(proc, port, data, content_type, activity_polling_timeout_seconds=250):
    """
    :param activity_polling_timeout_seconds: The amount of time, in seconds, to wait before
                                             declaring the scoring process to have failed.
    """
    try:
        for i in range(0, int(activity_polling_timeout_seconds / 5)):
            assert proc.poll() is None, "scoring process died"
            time.sleep(5)
            # noinspection PyBroadException
            try:
                ping_status = requests.get(url='http://localhost:%d/ping' % port)
                print('connection attempt', i, "server is up! ping status", ping_status)
                if ping_status.status_code == 200:
                    break
            except Exception:  # pylint: disable=broad-except
                print('connection attempt', i, "failed, server is not up yet")

        assert proc.poll() is None, "scoring process died"
        ping_status = requests.get(url='http://localhost:%d/ping' % port)
        print("server up, ping status", ping_status)
        if ping_status.status_code != 200:
            raise Exception("ping failed, server is not happy")
        if type(data) == pd.DataFrame:
            if content_type == pyfunc_scoring_server.CONTENT_TYPE_JSON:
                data = data.to_json(orient="records")
            elif content_type == pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED:
                data = data.to_json(orient="split")
            elif content_type == pyfunc_scoring_server.CONTENT_TYPE_CSV:
                data = data.to_csv()
            else:
                raise Exception(
                        "Unexpected content type for Pandas dataframe input %s" % content_type)
        response = requests.post(url='http://localhost:%d/invocations' % port,
                                 data=data,
                                 headers={"Content-Type": content_type})
        return response
    finally:
        if proc.poll() is None:
            # Terminate the process group containing the scoring process.
            # This will terminate all child processes of the scoring process
            pgrp = os.getpgid(proc.pid)
            os.killpg(pgrp, signal.SIGTERM)
        print("captured output of the scoring process")
        print("-------------------------STDOUT------------------------------")
        print(proc.stdout.read())
        print("==============================================================")
