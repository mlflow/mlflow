import os
import random
from unittest import mock

import requests
import string
import time
import signal
import socket
from subprocess import Popen
import uuid
import sys

import pandas as pd
import pytest

import mlflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.pyfunc
from mlflow.utils.file_utils import read_yaml, write_yaml

LOCALHOST = "127.0.0.1"


def get_safe_port():
    """Returns an ephemeral port that is very likely to be free to bind to."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((LOCALHOST, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def random_int(lo=1, hi=1e10):
    return random.randint(lo, hi)


def random_str(size=10, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def random_file(ext):
    return "temp_test_%d.%s" % (random_int(), ext)


def score_model_in_sagemaker_docker_container(
    model_uri,
    data,
    content_type,
    flavor=mlflow.pyfunc.FLAVOR_NAME,
    activity_polling_timeout_seconds=500,
):
    """
    :param model_uri: URI to the model to be served.
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
        cmd=["mlflow", "sagemaker", "run-local", "-m", model_uri, "-p", "5000", "-f", flavor],
        env=env,
    )
    return _evaluate_scoring_proc(proc, 5000, data, content_type, activity_polling_timeout_seconds)


def pyfunc_build_image(model_uri, extra_args=None):
    """
    Builds a docker image containing the specified model, returning the name of the image.
    :param model_uri: URI of model, e.g. runs:/some-run-id/run-relative/path/to/model
    :param extra_args: List of extra args to pass to `mlflow models build-docker` command
    """
    name = uuid.uuid4().hex
    cmd = ["mlflow", "models", "build-docker", "-m", model_uri, "-n", name]
    if extra_args:
        cmd += extra_args
    p = Popen(cmd,)
    assert p.wait() == 0, "Failed to build docker image to serve model from %s" % model_uri
    return name


def pyfunc_serve_from_docker_image(image_name, host_port, extra_args=None):
    """
    Serves a model from a docker container, exposing it as an endpoint at the specified port
    on the host machine. Returns a handle (Popen object) to the server process.
    """
    env = dict(os.environ)
    env.update(LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8")
    scoring_cmd = ["docker", "run", "-p", "%s:8080" % host_port, image_name]
    if extra_args is not None:
        scoring_cmd += extra_args
    return _start_scoring_proc(cmd=scoring_cmd, env=env)


def pyfunc_serve_from_docker_image_with_env_override(
    image_name, host_port, gunicorn_opts, extra_args=None
):
    """
    Serves a model from a docker container, exposing it as an endpoint at the specified port
    on the host machine. Returns a handle (Popen object) to the server process.
    """
    env = dict(os.environ)
    env.update(LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8")
    scoring_cmd = [
        "docker",
        "run",
        "-e",
        "GUNICORN_CMD_ARGS=%s" % gunicorn_opts,
        "-p",
        "%s:8080" % host_port,
        image_name,
    ]
    if extra_args is not None:
        scoring_cmd += extra_args
    return _start_scoring_proc(cmd=scoring_cmd, env=env)


def pyfunc_serve_and_score_model(
    model_uri,
    data,
    content_type,
    activity_polling_timeout_seconds=500,
    extra_args=None,
    stdout=sys.stdout,
):
    """
    :param model_uri: URI to the model to be served.
    :param data: The data to send to the pyfunc server for testing. This is either a
                 Pandas dataframe or string of the format specified by `content_type`.
    :param content_type: The type of the data to send to the pyfunc server for testing. This is
                         one of `mlflow.pyfunc.scoring_server.CONTENT_TYPES`.
    :param activity_polling_timeout_seconds: The amount of time, in seconds, to wait before
                                             declaring the scoring process to have failed.
    :param extra_args: A list of extra arguments to pass to the pyfunc scoring server command. For
                       example, passing ``extra_args=["--no-conda"]`` will pass the ``--no-conda``
                       flag to the scoring server to ensure that conda environment activation
                       is skipped.
    """
    env = dict(os.environ)
    env.update(LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8")
    env.update(MLFLOW_TRACKING_URI=mlflow.get_tracking_uri())
    env.update(MLFLOW_HOME=_get_mlflow_home())
    port = get_safe_port()
    scoring_cmd = [
        "mlflow",
        "models",
        "serve",
        "-m",
        model_uri,
        "-p",
        str(port),
        "--install-mlflow",
    ]
    if extra_args is not None:
        scoring_cmd += extra_args
    proc = _start_scoring_proc(cmd=scoring_cmd, env=env, stdout=stdout, stderr=stdout)
    return _evaluate_scoring_proc(proc, port, data, content_type, activity_polling_timeout_seconds)


def _get_mlflow_home():
    """
    :return: The path to the MLflow installation root directory
    """
    mlflow_module_path = os.path.dirname(os.path.abspath(mlflow.__file__))
    # The MLflow root directory is one level about the mlflow module location
    return os.path.join(mlflow_module_path, os.pardir)


def _start_scoring_proc(cmd, env, stdout=sys.stdout, stderr=sys.stderr):
    proc = Popen(
        cmd,
        stdout=stdout,
        stderr=stderr,
        universal_newlines=True,
        env=env,
        # Assign the scoring process to a process group. All child processes of the
        # scoring process will be assigned to this group as well. This allows child
        # processes of the scoring process to be terminated successfully
        preexec_fn=os.setsid,
    )
    return proc


class RestEndpoint:
    def __init__(self, proc, port, activity_polling_timeout_seconds=250):
        self._proc = proc
        self._port = port
        self._activity_polling_timeout_seconds = activity_polling_timeout_seconds

    def __enter__(self):
        for i in range(0, int(self._activity_polling_timeout_seconds / 5)):
            assert self._proc.poll() is None, "scoring process died"
            time.sleep(5)
            # noinspection PyBroadException
            try:
                ping_status = requests.get(url="http://localhost:%d/ping" % self._port)
                print("connection attempt", i, "server is up! ping status", ping_status)
                if ping_status.status_code == 200:
                    break
            except Exception:
                print("connection attempt", i, "failed, server is not up yet")
        if ping_status.status_code != 200:
            raise Exception("ping failed, server is not happy")
        print("server up, ping status", ping_status)
        return self

    def __exit__(self, tp, val, traceback):
        if self._proc.poll() is None:
            # Terminate the process group containing the scoring process.
            # This will terminate all child processes of the scoring process
            pgrp = os.getpgid(self._proc.pid)
            os.killpg(pgrp, signal.SIGTERM)

    def invoke(self, data, content_type):
        if type(data) == pd.DataFrame:
            if content_type == pyfunc_scoring_server.CONTENT_TYPE_JSON_RECORDS_ORIENTED:
                data = data.to_json(orient="records")
            elif (
                content_type == pyfunc_scoring_server.CONTENT_TYPE_JSON
                or content_type == pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED
            ):
                data = data.to_json(orient="split")
            elif content_type == pyfunc_scoring_server.CONTENT_TYPE_CSV:
                data = data.to_csv(index=False)
            else:
                raise Exception(
                    "Unexpected content type for Pandas dataframe input %s" % content_type
                )
        response = requests.post(
            url="http://localhost:%d/invocations" % self._port,
            data=data,
            headers={"Content-Type": content_type},
        )
        return response


def _evaluate_scoring_proc(proc, port, data, content_type, activity_polling_timeout_seconds=250):
    """
    :param activity_polling_timeout_seconds: The amount of time, in seconds, to wait before
                                             declaring the scoring process to have failed.
    """
    with RestEndpoint(proc, port, activity_polling_timeout_seconds) as endpoint:
        return endpoint.invoke(data, content_type)


@pytest.fixture(scope="module", autouse=True)
def set_boto_credentials():
    os.environ["AWS_ACCESS_KEY_ID"] = "NotARealAccessKey"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "NotARealSecretAccessKey"
    os.environ["AWS_SESSION_TOKEN"] = "NotARealSessionToken"


@pytest.fixture
def mock_s3_bucket():
    """
    Creates a mock S3 bucket using moto

    :return: The name of the mock bucket
    """
    import boto3
    import moto

    with moto.mock_s3():
        bucket_name = "mock-bucket"
        s3_client = boto3.client("s3")
        s3_client.create_bucket(Bucket=bucket_name)
        yield bucket_name


class safe_edit_yaml(object):
    def __init__(self, root, file_name, edit_func):
        self._root = root
        self._file_name = file_name
        self._edit_func = edit_func
        self._original = read_yaml(root, file_name)

    def __enter__(self):
        new_dict = self._edit_func(self._original.copy())
        write_yaml(self._root, self._file_name, new_dict, overwrite=True)

    def __exit__(self, *args):
        write_yaml(self._root, self._file_name, self._original, overwrite=True)


def create_mock_response(status_code, text):
    """
    Create a mock resposne object with the status_code and text

    :param: status_code int HTTP status code
    :param: text message from the response
    :reutrn: mock HTTP Response
    """
    response = mock.MagicMock()
    response.status_code = status_code
    response.text = text
    return response
