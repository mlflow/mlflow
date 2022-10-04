import os
import random
import functools
from unittest import mock
from contextlib import ExitStack, contextmanager

import logging
import requests
import time
import signal
import socket
import subprocess
import uuid
import sys
import yaml

import pytest

import mlflow
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import read_yaml, write_yaml
from mlflow.utils.environment import (
    _get_pip_deps,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
)

AWS_METADATA_IP = "169.254.169.254"  # Used to fetch AWS Instance and User metadata.
LOCALHOST = "127.0.0.1"
PROTOBUF_REQUIREMENT = "protobuf<4.0.0"

_logger = logging.getLogger(__name__)


def get_safe_port():
    """Returns an ephemeral port that is very likely to be free to bind to."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((LOCALHOST, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def random_int(lo=1, hi=1e10):
    return random.randint(lo, hi)


def random_str(size=10):
    msg = (
        "UUID4 generated strings have a high potential for collision at small sizes."
        "10 is set as the lower bounds for random string generation to prevent non-deterministic"
        "test failures."
    )
    assert size >= 10, msg
    return uuid.uuid4().hex[:size]


def random_file(ext):
    return "temp_test_%d.%s" % (random_int(), ext)


def score_model_in_sagemaker_docker_container(
    model_uri,
    data,
    content_type,
    flavor="python_function",
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
    return _evaluate_scoring_proc(
        proc, 5000, data, content_type, activity_polling_timeout_seconds, False
    )


def pyfunc_generate_dockerfile(output_directory, model_uri=None, extra_args=None):
    """
    Builds a dockerfile for the specified model.
    :param model_uri: URI of model, e.g. runs:/some-run-id/run-relative/path/to/model
    :param extra_args: List of extra args to pass to `mlflow models build-docker` command
    :param output_directory: Output directory to generate Dockerfile and model artifacts
    """
    cmd = [
        "mlflow",
        "models",
        "generate-dockerfile",
        *(["-m", model_uri] if model_uri else []),
        "-d",
        output_directory,
    ]
    mlflow_home = os.environ.get("MLFLOW_HOME")
    if mlflow_home:
        cmd += ["--mlflow-home", mlflow_home]
    if extra_args:
        cmd += extra_args
    subprocess.run(cmd, check=True)


def pyfunc_build_image(model_uri=None, extra_args=None):
    """
    Builds a docker image containing the specified model, returning the name of the image.
    :param model_uri: URI of model, e.g. runs:/some-run-id/run-relative/path/to/model
    :param extra_args: List of extra args to pass to `mlflow models build-docker` command
    """
    name = uuid.uuid4().hex
    cmd = [
        "mlflow",
        "models",
        "build-docker",
        *(["-m", model_uri] if model_uri else []),
        "-n",
        name,
    ]
    mlflow_home = os.environ.get("MLFLOW_HOME")
    if mlflow_home:
        cmd += ["--mlflow-home", mlflow_home]
    if extra_args:
        cmd += extra_args
    p = subprocess.Popen(cmd)
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
    image_name, host_port, gunicorn_opts, extra_args=None, extra_docker_run_options=None
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
        *(extra_docker_run_options or []),
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
                       example, passing ``extra_args=["--env-manager", "local"]`` will pass the
                       ``--env-manager local`` flag to the scoring server to ensure that conda
                       environment activation is skipped.
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
    validate_version = True
    if extra_args is not None:
        scoring_cmd += extra_args
        validate_version = "--enable-mlserver" not in extra_args
    proc = _start_scoring_proc(cmd=scoring_cmd, env=env, stdout=stdout, stderr=stdout)
    return _evaluate_scoring_proc(
        proc, port, data, content_type, activity_polling_timeout_seconds, validate_version
    )


def _get_mlflow_home():
    """
    :return: The path to the MLflow installation root directory
    """
    mlflow_module_path = os.path.dirname(os.path.abspath(mlflow.__file__))
    # The MLflow root directory is one level about the mlflow module location
    return os.path.join(mlflow_module_path, os.pardir)


def _start_scoring_proc(cmd, env, stdout=sys.stdout, stderr=sys.stderr):
    if os.name != "nt":
        return subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            text=True,
            env=env,
            # Assign the scoring process to a process group. All child processes of the
            # scoring process will be assigned to this group as well. This allows child
            # processes of the scoring process to be terminated successfully
            preexec_fn=os.setsid,
        )
    else:
        return subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            text=True,
            env=env,
            # On Windows, `os.setsid` and `preexec_fn` are unavailable
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )


class RestEndpoint:
    def __init__(self, proc, port, activity_polling_timeout_seconds=250, validate_version=True):
        self._proc = proc
        self._port = port
        self._activity_polling_timeout_seconds = activity_polling_timeout_seconds
        self._validate_version = validate_version

    def __enter__(self):
        ping_status = None
        for i in range(self._activity_polling_timeout_seconds):
            assert self._proc.poll() is None, "scoring process died"
            time.sleep(1)
            # noinspection PyBroadException
            try:
                ping_status = requests.get(url="http://localhost:%d/ping" % self._port)
                _logger.info(f"connection attempt {i} server is up! ping status {ping_status}")
                if ping_status.status_code == 200:
                    break
            except Exception:
                _logger.info(f"connection attempt {i} failed, server is not up yet")
        if ping_status is None or ping_status.status_code != 200:
            raise Exception("ping failed, server is not happy")
        _logger.info(f"server up, ping status {ping_status}")

        if self._validate_version:
            resp_status = requests.get(url="http://localhost:%d/version" % self._port)
            version = resp_status.text
            _logger.info(f"mlflow server version {version}")
            if version != mlflow.__version__:
                raise Exception("version path is not returning correct mlflow version")
        return self

    def __exit__(self, tp, val, traceback):
        if self._proc.poll() is None:
            # Terminate the process group containing the scoring process.
            # This will terminate all child processes of the scoring process
            if os.name != "nt":
                pgrp = os.getpgid(self._proc.pid)
                os.killpg(pgrp, signal.SIGTERM)
            else:
                # https://stackoverflow.com/questions/47016723/windows-equivalent-for-spawning-and-killing-separate-process-group-in-python-3
                self._proc.send_signal(signal.CTRL_BREAK_EVENT)
                self._proc.kill()

    def invoke(self, data, content_type):
        import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
        import pandas as pd

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


def _evaluate_scoring_proc(
    proc, port, data, content_type, activity_polling_timeout_seconds=250, validate_version=True
):
    """
    :param activity_polling_timeout_seconds: The amount of time, in seconds, to wait before
                                             declaring the scoring process to have failed.
    """
    with RestEndpoint(
        proc, port, activity_polling_timeout_seconds, validate_version=validate_version
    ) as endpoint:
        return endpoint.invoke(data, content_type)


@pytest.fixture(scope="function", autouse=True)
def set_boto_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "NotARealAccessKey")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "NotARealSecretAccessKey")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "NotARealSessionToken")


class safe_edit_yaml:
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
    Create a mock response object with the status_code and text

    :param: status_code int HTTP status code
    :param: text message from the response
    :return: mock HTTP Response
    """
    response = mock.MagicMock()
    response.status_code = status_code
    response.text = text
    return response


def _read_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _read_lines(path):
    with open(path, "r") as f:
        return f.read().splitlines()


def _compare_logged_code_paths(code_path, model_path, flavor_name):
    import mlflow.pyfunc
    from mlflow.utils.model_utils import _get_flavor_configuration, FLAVOR_CONFIG_CODE

    pyfunc_conf = _get_flavor_configuration(
        model_path=model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME
    )
    flavor_conf = _get_flavor_configuration(model_path, flavor_name=flavor_name)
    assert pyfunc_conf[mlflow.pyfunc.CODE] == flavor_conf[FLAVOR_CONFIG_CODE]
    saved_code_path = os.path.join(model_path, pyfunc_conf[mlflow.pyfunc.CODE])
    assert os.path.exists(saved_code_path)

    with open(os.path.join(saved_code_path, os.path.basename(code_path)), "r") as f1:
        with open(code_path, "r") as f2:
            assert f1.read() == f2.read()


def _compare_conda_env_requirements(env_path, req_path):
    assert os.path.exists(req_path)
    custom_env_parsed = _read_yaml(env_path)
    requirements = _read_lines(req_path)
    assert _get_pip_deps(custom_env_parsed) == requirements


def _assert_pip_requirements(model_uri, requirements, constraints=None, strict=False):
    """
    Loads the pip requirements (and optionally constraints) from `model_uri` and compares them
    to `requirements` (and `constraints`).

    If `strict` is True, evaluate `set(requirements) == set(loaded_requirements)`.
    Otherwise, evaluate `set(requirements) <= set(loaded_requirements)`.
    """
    local_path = _download_artifact_from_uri(model_uri)
    txt_reqs = _read_lines(os.path.join(local_path, _REQUIREMENTS_FILE_NAME))
    conda_reqs = _get_pip_deps(_read_yaml(os.path.join(local_path, _CONDA_ENV_FILE_NAME)))
    compare_func = set.__eq__ if strict else set.__le__
    requirements = set(requirements)
    assert compare_func(requirements, set(txt_reqs))
    assert compare_func(requirements, set(conda_reqs))

    if constraints is not None:
        assert f"-c {_CONSTRAINTS_FILE_NAME}" in txt_reqs
        assert f"-c {_CONSTRAINTS_FILE_NAME}" in conda_reqs
        cons = _read_lines(os.path.join(local_path, _CONSTRAINTS_FILE_NAME))
        assert compare_func(set(constraints), set(cons))


def _is_available_on_pypi(package, version=None, module=None):
    """
    Returns True if the specified package version is available on PyPI.

    :param package: The name of the package.
    :param version: The version of the package. If None, defaults to the installed version.
    :param module: The name of the top-level module provided by the package . For example,
                   if `package` is 'scikit-learn', `module` should be 'sklearn'. If None, defaults
                   to `package`.
    """
    from mlflow.utils.requirements_utils import _get_installed_version

    resp = requests.get("https://pypi.python.org/pypi/{}/json".format(package))
    if not resp.ok:
        return False

    version = version or _get_installed_version(module or package)
    dist_files = resp.json()["releases"].get(version)
    return (
        dist_files is not None  # specified version exists
        and (len(dist_files) > 0)  # at least one distribution file exists
        and not dist_files[0].get("yanked", False)  # specified version is not yanked
    )


def _is_importable(module_name):
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def allow_infer_pip_requirements_fallback_if(condition):
    def decorator(f):
        return pytest.mark.allow_infer_pip_requirements_fallback(f) if condition else f

    return decorator


def mock_method_chain(mock_obj, methods, return_value=None, side_effect=None):
    """
    Mock a chain of methods.

    Examples
    --------
    >>> from unittest import mock
    >>> m = mock.MagicMock()
    >>> mock_method_chain(m, ["a", "b"], return_value=0)
    >>> m.a().b()
    0
    >>> mock_method_chain(m, ["c.d", "e"], return_value=1)
    >>> m.c.d().e()
    1
    >>> mock_method_chain(m, ["f"], side_effect=Exception("side_effect"))
    >>> m.f()
    Traceback (most recent call last):
      ...
    Exception: side_effect
    """
    length = len(methods)
    for idx, method in enumerate(methods):
        mock_obj = functools.reduce(getattr, method.split("."), mock_obj)
        if idx != length - 1:
            mock_obj = mock_obj.return_value
        else:
            mock_obj.return_value = return_value
            mock_obj.side_effect = side_effect


@contextmanager
def multi_context(*cms):
    with ExitStack() as stack:
        yield list(map(stack.enter_context, cms))


class StartsWithMatcher:
    def __init__(self, prefix):
        self.prefix = prefix

    def __eq__(self, other):
        return isinstance(other, str) and other.startswith(self.prefix)


class AnyStringWith(str):
    def __eq__(self, other):
        return self in other
