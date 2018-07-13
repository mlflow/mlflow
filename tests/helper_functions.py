import os
import random
import requests
import string
from subprocess import Popen, PIPE, STDOUT
import time


def random_int(lo=1, hi=1e10):
    return random.randint(lo, hi)


def random_str(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def random_file(ext):
    return "temp_test_%d.%s" % (random_int(), ext)


def score_model_in_sagemaker_docker_container(model_path, data):
    env = dict(os.environ)
    env.update(LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8")
    proc = Popen(['mlflow', 'sagemaker', 'run-local', '-m', model_path], stdout=PIPE, stderr=STDOUT,
                 universal_newlines=True, env=env)
    try:
        for i in range(0, 50):
            assert proc.poll() is None, "scoring process died"
            time.sleep(5)
            # noinspection PyBroadException
            try:
                ping_status = requests.get(url='http://localhost:5000/ping')
                print('connection attempt', i, "server is up! ping status", ping_status)
                if ping_status.status_code == 200:
                    break
            except Exception:  # pylint: disable=broad-except
                print('connection attempt', i, "failed, server is not up yet")

        assert proc.poll() is None, "scoring process died"
        ping_status = requests.get(url='http://localhost:5000/ping')
        print("server up, ping status", ping_status)
        if ping_status.status_code != 200:
            raise Exception("ping failed, server is not happy")
        x = data.to_dict(orient='records')
        y = requests.post(url='http://localhost:5000/invocations', json=x)
        import json
        return json.loads(y.content)
    finally:
        if proc.poll() is None:
            proc.terminate()
        print("captured output of the scoring process")
        print(proc.stdout.read())
