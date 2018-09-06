import os
import random

import re

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import requests
import string
from subprocess import Popen, PIPE, STDOUT
import time

import pandas as pd


def random_int(lo=1, hi=1e10):
    return random.randint(lo, hi)


def random_str(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def random_file(ext):
    return "temp_test_%d.%s" % (random_int(), ext)


def score_model_in_sagemaker_docker_container(model_path, data):
    """
    :param data: The data to send to the docker container for testing. This is either a
                 Pandas dataframe or a JSON-formatted string.
    """
    env = dict(os.environ)
    env.update(LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8")
    proc = Popen(['mlflow', 'sagemaker', 'run-local', '-m', model_path, '-p', "5000"],
                 stdout=PIPE,
                 stderr=STDOUT,
                 universal_newlines=True, env=env)
    r = _score_proc(proc, 5000, data, "json").content
    import json
    return json.loads(r)  # TODO: we should return pd.Dataframe the same as pyfunc serve


def pyfunc_serve_and_score_model(model_path, data):
    env = dict(os.environ)
    env.update(LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8")
    cmd = ['mlflow', 'pyfunc', 'serve', '-m', model_path, "-p", "0"]
    proc = Popen(cmd,
                 stdout=PIPE,
                 stderr=STDOUT,
                 universal_newlines=True,
                 env=env)
    for x in iter(proc.stdout.readline, ""):
        print(x)
        m = re.match(pattern=".*Running on http://127.0.0.1:(\\d+).*", string=x)
        if m:
            return pd.read_json(_score_proc(proc, int(m.group(1)), data, data_type="csv").content,
                                orient="records")

    raise Exception("Failed to start server")


def _score_proc(proc, port, data, data_type):
    try:
        for i in range(0, 50):
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
        x = StringIO()
        if data_type == "json":
            if type(data) == pd.DataFrame:
                data = data.to_dict(orient="records")
            requests.post(url='http://localhost:%d/invocations' % port,
                          json=data)
        elif data_type == "csv":
            data.to_csv(x, index=False, header=True)
            r = requests.request(method="post", url='http://localhost:%d/invocations' % port,
                                 data=x.getvalue(),
                                 headers={"Content-Type": "text/csv"})
        else:
            raise Exception("Unexpected data_type %s" % data_type)
        if r.status_code != 200:
            raise Exception("scoring failed, status code = {}".format(r.status_code))
        return r
    finally:
        if proc.poll() is None:
            proc.terminate()
        print("captured output of the scoring process")
        print("-------------------------STDOUT------------------------------")
        print(proc.stdout.read())
        print("==============================================================")
