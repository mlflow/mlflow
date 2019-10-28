"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures we can use the tracking API to communicate with it.
"""

import mock
from subprocess import Popen
import os
import sys
import posixpath
import pytest
from six.moves import urllib
import socket
import shutil
from threading import Thread
import time
import tempfile

import mlflow.experiments
from mlflow.entities.model_registry import RegisteredModelDetailed, RegisteredModel
from mlflow.exceptions import MlflowException
from mlflow.entities import RunStatus, Metric, Param, RunTag, ViewType
from mlflow.server import BACKEND_STORE_URI_ENV_VAR, ARTIFACT_ROOT_ENV_VAR
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_USER, MLFLOW_RUN_NAME, MLFLOW_PARENT_RUN_ID, \
    MLFLOW_SOURCE_TYPE, MLFLOW_SOURCE_NAME, MLFLOW_PROJECT_ENTRY_POINT, MLFLOW_GIT_COMMIT
from mlflow.utils.file_utils import path_to_local_file_uri, local_file_uri_to_path
from tests.integration.utils import invoke_cli_runner

from tests.helper_functions import LOCALHOST, get_safe_port


def _await_server_up_or_die(port, timeout=60):
    """Waits until the local flask server is listening on the given port."""
    print('Awaiting server to be up on %s:%s' % (LOCALHOST, port))
    start_time = time.time()
    connected = False
    while not connected and time.time() - start_time < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((LOCALHOST, port))
        if result == 0:
            connected = True
        else:
            print('Server not yet up, waiting...')
            time.sleep(0.5)
    if not connected:
        raise Exception('Failed to connect on %s:%s after %s seconds' % (LOCALHOST, port, timeout))
    print('Server is up on %s:%s!' % (LOCALHOST, port))


# NB: We explicitly wait and timeout on server shutdown in order to ensure that pytest output
# reveals the cause in the event of a test hang due to the subprocess not exiting.
def _await_server_down_or_die(process, timeout=60):
    """Waits until the local flask server process is terminated."""
    print('Awaiting termination of server process...')
    start_time = time.time()

    def wait():
        process.wait()

    Thread(target=wait).start()
    while process.returncode is None and time.time() - start_time < timeout:
        time.sleep(0.5)
    if process.returncode is None:
        raise Exception('Server failed to shutdown after %s seconds' % timeout)


def _init_server(backend_uri, root_artifact_uri):
    """
    Launch a new REST server using the tracking store specified by backend_uri and root artifact
    directory specified by root_artifact_uri.
    :returns A tuple (url, process) containing the string URL of the server and a handle to the
             server process (a multiprocessing.Process object).
    """
    mlflow.set_tracking_uri(None)
    server_port = get_safe_port()
    env = {
        BACKEND_STORE_URI_ENV_VAR: backend_uri,
        ARTIFACT_ROOT_ENV_VAR: path_to_local_file_uri(
            tempfile.mkdtemp(dir=local_file_uri_to_path(root_artifact_uri))),
    }
    with mock.patch.dict(os.environ, env):
        cmd = ["python",
               "-c",
               'from mlflow.server import app; app.run("{hostname}", {port})'.format(
                   hostname=LOCALHOST, port=server_port)]
        process = Popen(cmd)

    _await_server_up_or_die(server_port)
    url = "http://{hostname}:{port}".format(hostname=LOCALHOST, port=server_port)
    print("Launching tracking server against backend URI %s. Server URL: %s" % (backend_uri, url))
    return url, process


# Root directory for all stores (backend or artifact stores) created during this suite
SUITE_ROOT_DIR = tempfile.mkdtemp("test_rest_tracking")
# Root directory for all artifact stores created during this suite
SUITE_ARTIFACT_ROOT_DIR = tempfile.mkdtemp(suffix="artifacts", dir=SUITE_ROOT_DIR)


def _get_sqlite_uri():
    path = path_to_local_file_uri(os.path.join(SUITE_ROOT_DIR, "test-database.bd"))
    path = path[len("file://"):]

    # NB: It looks like windows and posix have different requirements on number of slashes for
    # whatever reason. Windows needs uri like 'sqlite:///C:/path/to/my/file' whereas posix expects
    # sqlite://///path/to/my/file
    prefix = "sqlite://" if sys.platform == "win32" else "sqlite:////"
    return prefix + path


# Backend store URIs to test against
BACKEND_URIS = [
    _get_sqlite_uri(),  # SqlAlchemy
]

# Map of backend URI to tuple (server URL, Process). We populate this map by constructing
# a server per backend URI
BACKEND_URI_TO_SERVER_URL_AND_PROC = {
    uri: _init_server(backend_uri=uri,
                      root_artifact_uri=SUITE_ARTIFACT_ROOT_DIR)
    for uri in BACKEND_URIS
}


def pytest_generate_tests(metafunc):
    """
    Automatically parametrize each each fixture/test that depends on `backend_store_uri` with the
    list of backend store URIs.
    """
    if 'backend_store_uri' in metafunc.fixturenames:
        metafunc.parametrize('backend_store_uri', BACKEND_URIS)


@pytest.fixture(scope="module", autouse=True)
def server_urls():
    """
    Clean up all servers created for testing in `pytest_generate_tests`
    """
    yield
    for server_url, process in BACKEND_URI_TO_SERVER_URL_AND_PROC.values():
        print("Terminating server at %s..." % (server_url))
        print("type = ", type(process))
        process.terminate()
        _await_server_down_or_die(process)
    shutil.rmtree(SUITE_ROOT_DIR)


@pytest.fixture()
def tracking_server_uri(backend_store_uri):
    url, _ = BACKEND_URI_TO_SERVER_URL_AND_PROC[backend_store_uri]
    return url


@pytest.fixture()
def mlflow_client(tracking_server_uri):
    """Provides an MLflow Tracking API client pointed at the local tracking server."""
    return mock.Mock(wraps=MlflowClient(tracking_server_uri))


@pytest.fixture()
def cli_env(tracking_server_uri):
    """Provides an environment for the MLflow CLI pointed at the local tracking server."""
    cli_env = {
        "LC_ALL": "en_US.UTF-8",
        "LANG": "en_US.UTF-8",
        "MLFLOW_TRACKING_URI": tracking_server_uri,
    }
    return cli_env


def assert_is_between(start_time, end_time, expected_time):
    assert expected_time >= start_time
    assert expected_time <= end_time


def now():
    return int(time.time() * 1000)


def test_create_registered_model(mlflow_client, backend_store_uri):
    name = 'CreateRMTest'
    start_time = now()
    registered_model = mlflow_client.create_registered_model(name)
    end_time = now()
    assert isinstance(registered_model, RegisteredModel)
    assert registered_model.name == name
    registered_model_detailed = mlflow_client.get_registered_model_details(name)
    assert isinstance(registered_model_detailed, RegisteredModelDetailed)
    assert registered_model_detailed.name == name
    assert str(registered_model_detailed.description) is ""
    assert registered_model_detailed.latest_versions == []
    assert_is_between(start_time, end_time, registered_model_detailed.creation_timestamp)
    assert_is_between(start_time, end_time, registered_model_detailed.last_updated_timestamp)
    assert [name] == [rm.name for rm in mlflow_client.list_registered_models() if rm.name == name]


def test_update_registered_model(mlflow_client, backend_store_uri):
    name = 'UpdateRMTest'
    start_time_1 = now()
    mlflow_client.create_registered_model(name)
    end_time_1 = now()
    registered_model_detailed_1 = mlflow_client.get_registered_model_details(name)
    assert registered_model_detailed_1.name == name
    assert str(registered_model_detailed_1.description) is ""
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_1.creation_timestamp)
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_1.last_updated_timestamp)

    # update with no args is an error
    with pytest.raises(MlflowException) as e:
        mlflow_client.update_registered_model(name=name, new_name=None, description=None)

    # update name
    new_name = "UpdateRMTest 2"
    start_time_2 = now()
    mlflow_client.update_registered_model(name=name, new_name=new_name)
    end_time_2 = now()
    with pytest.raises(MlflowException):
        mlflow_client.get_registered_model_details(name)
    registered_model_detailed_2 = mlflow_client.get_registered_model_details(new_name)
    assert registered_model_detailed_2.name == new_name
    assert str(registered_model_detailed_2.description) is ""
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_2.creation_timestamp)
    assert_is_between(start_time_2, end_time_2, registered_model_detailed_2.last_updated_timestamp)

    # update description
    start_time_3 = now()
    mlflow_client.update_registered_model(name=new_name, description="This is a test")
    end_time_3 = now()
    registered_model_detailed_3 = mlflow_client.get_registered_model_details(new_name)
    assert registered_model_detailed_3.name == new_name
    assert registered_model_detailed_3.description == "This is a test"
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_3.creation_timestamp)
    assert_is_between(start_time_3, end_time_3, registered_model_detailed_3.last_updated_timestamp)

    # update name and description
    another_new = "UpdateRMTest 4"
    start_time_4 = now()
    mlflow_client.update_registered_model(new_name, another_new, "4th update")
    end_time_4 = now()
    registered_model_detailed_4 = mlflow_client.get_registered_model_details(another_new)
    assert registered_model_detailed_4.name == another_new
    assert registered_model_detailed_4.description == "4th update"
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_4.creation_timestamp)
    assert_is_between(start_time_4, end_time_4, registered_model_detailed_4.last_updated_timestamp)

    # old named models are not accessible
    for old_name in [name, new_name]:
        with pytest.raises(MlflowException) as e:
            mlflow_client.get_registered_model_details(old_name)


def test_delete_registered_model(mlflow_client, backend_store_uri):
    name = 'UpdateRMTest'
    start_time_1 = now()
    mlflow_client.create_registered_model(name)
    end_time_1 = now()
    registered_model_detailed_1 = mlflow_client.get_registered_model_details(name)
    assert registered_model_detailed_1.name == name
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_1.creation_timestamp)
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_1.last_updated_timestamp)

    assert [name] == [rm.name for rm in mlflow_client.list_registered_models() if rm.name == name]

    # cannot create a model with same name
    with pytest.raises(MlflowException) as e:
        mlflow_client.create_registered_model(name)

    mlflow_client.delete_registered_model(name)

    # cannot get a deleted model
    with pytest.raises(MlflowException) as e:
        mlflow_client.get_registered_model_details(name)

    # cannot update a deleted model
    with pytest.raises(MlflowException) as e:
        mlflow_client.update_registered_model(name=name, new_name="something else")

    # list does not include deleted model
    assert [] == [rm.name for rm in mlflow_client.list_registered_models() if rm.name == name]

    # recreate model with same name
    start_time_2 = now()
    mlflow_client.create_registered_model(name)
    end_time_2 = now()
    registered_model_detailed_2 = mlflow_client.get_registered_model_details(name)
    assert registered_model_detailed_2.name == name
    assert_is_between(start_time_2, end_time_2, registered_model_detailed_2.creation_timestamp)
    assert_is_between(start_time_2, end_time_2, registered_model_detailed_2.last_updated_timestamp)

    assert [name] == [rm.name for rm in mlflow_client.list_registered_models() if rm.name == name]
