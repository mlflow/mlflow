"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures we can use the tracking API to communicate with it.
"""

import mock
from subprocess import Popen, PIPE, STDOUT
import os
import sys
import pytest
import socket
import shutil
from threading import Thread
import time
import tempfile

import mlflow.experiments
from mlflow.entities import RunStatus, Metric, Param, RunTag
from mlflow.protos.service_pb2 import LOCAL as SOURCE_TYPE_LOCAL
from mlflow.server import app, BACKEND_STORE_URI_ENV_VAR, ARTIFACT_ROOT_ENV_VAR
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_PARENT_RUN_ID, MLFLOW_SOURCE_TYPE, \
    MLFLOW_SOURCE_NAME, MLFLOW_PROJECT_ENTRY_POINT, MLFLOW_GIT_COMMIT
from mlflow.utils.file_utils import path_to_local_file_uri, local_file_uri_to_path
from tests.integration.utils import invoke_cli_runner

LOCALHOST = '127.0.0.1'


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
    server_port = _get_safe_port()
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


def _get_safe_port():
    """Returns an ephemeral port that is very likely to be free to bind to."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((LOCALHOST, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


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
    path_to_local_file_uri(os.path.join(SUITE_ROOT_DIR, "file_store_root")),  # FileStore
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
    return MlflowClient(tracking_server_uri)


@pytest.fixture()
def cli_env(tracking_server_uri):
    """Provides an environment for the MLflow CLI pointed at the local tracking server."""
    cli_env = {
        "LC_ALL": "en_US.UTF-8",
        "LANG": "en_US.UTF-8",
        "MLFLOW_TRACKING_URI": tracking_server_uri,
    }
    return cli_env


def test_create_get_list_experiment(mlflow_client):
    experiment_id = mlflow_client.create_experiment('My Experiment',
                                                    artifact_location='my_location')
    exp = mlflow_client.get_experiment(experiment_id)
    assert exp.name == 'My Experiment'
    assert exp.artifact_location == 'my_location'

    experiments = mlflow_client.list_experiments()
    assert set([e.name for e in experiments]) == {'My Experiment', 'Default'}


def test_delete_restore_experiment(mlflow_client):
    experiment_id = mlflow_client.create_experiment('Deleterious')
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == 'active'
    mlflow_client.delete_experiment(experiment_id)
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == 'deleted'
    mlflow_client.restore_experiment(experiment_id)
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == 'active'


def test_delete_restore_experiment_cli(mlflow_client, cli_env):
    experiment_name = "DeleteriousCLI"
    invoke_cli_runner(mlflow.experiments.commands, ['create', experiment_name], env=cli_env)
    experiment_id = mlflow_client.get_experiment_by_name(experiment_name).experiment_id
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == 'active'
    invoke_cli_runner(mlflow.experiments.commands, ['delete', str(experiment_id)], env=cli_env)
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == 'deleted'
    invoke_cli_runner(mlflow.experiments.commands, ['restore', str(experiment_id)], env=cli_env)
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == 'active'


def test_rename_experiment(mlflow_client):
    experiment_id = mlflow_client.create_experiment('BadName')
    assert mlflow_client.get_experiment(experiment_id).name == 'BadName'
    mlflow_client.rename_experiment(experiment_id, 'GoodName')
    assert mlflow_client.get_experiment(experiment_id).name == 'GoodName'


def test_rename_experiment_cli(mlflow_client, cli_env):
    bad_experiment_name = "CLIBadName"
    good_experiment_name = "CLIGoodName"

    invoke_cli_runner(mlflow.experiments.commands, ['create', bad_experiment_name], env=cli_env)
    experiment_id = mlflow_client.get_experiment_by_name(bad_experiment_name).experiment_id
    assert mlflow_client.get_experiment(experiment_id).name == bad_experiment_name
    invoke_cli_runner(
        mlflow.experiments.commands,
        ['rename', str(experiment_id), good_experiment_name], env=cli_env)
    assert mlflow_client.get_experiment(experiment_id).name == good_experiment_name


@pytest.mark.parametrize("parent_run_id_kwarg", [None, "my-parent-id"])
def test_create_run_all_args(mlflow_client, parent_run_id_kwarg):
    source_name = "Hello"
    entry_point = "entry"
    source_version = "abc"
    create_run_kwargs = {
        "user_id": "123",
        "run_name": "My name",
        "start_time": 456,
        "tags": {
            MLFLOW_SOURCE_TYPE: "LOCAL",
            MLFLOW_SOURCE_NAME: source_name,
            MLFLOW_PROJECT_ENTRY_POINT: entry_point,
            MLFLOW_GIT_COMMIT: source_version,
            MLFLOW_PARENT_RUN_ID: "7",
            "my": "tag",
            "other": "tag",
        },
        "parent_run_id": parent_run_id_kwarg
    }
    experiment_id = mlflow_client.create_experiment('Run A Lot (parent_run_id=%s)'
                                                    % (parent_run_id_kwarg))
    created_run = mlflow_client.create_run(experiment_id, **create_run_kwargs)
    run_id = created_run.info.run_id
    print("Run id=%s" % run_id)
    run = mlflow_client.get_run(run_id)
    assert run.info.run_id == run_id
    assert run.info.run_uuid == run_id
    assert run.info.experiment_id == experiment_id
    assert run.info.user_id == create_run_kwargs["user_id"]
    assert run.info.source_type == SOURCE_TYPE_LOCAL
    assert run.info.source_name == source_name
    assert run.info.entry_point_name == entry_point
    assert run.info.start_time == create_run_kwargs["start_time"]
    assert run.info.source_version == source_version
    for tag in create_run_kwargs["tags"]:
        assert tag in run.data.tags
    assert run.data.tags.get(MLFLOW_RUN_NAME) == create_run_kwargs["run_name"]
    assert run.data.tags.get(MLFLOW_PARENT_RUN_ID) == parent_run_id_kwarg or "7"
    assert mlflow_client.list_run_infos(experiment_id) == [run.info]


def test_create_run_defaults(mlflow_client):
    experiment_id = mlflow_client.create_experiment('Run A Little')
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    run = mlflow_client.get_run(run_id)
    assert run.info.run_id == run_id
    assert run.info.experiment_id == experiment_id
    assert run.info.user_id is not None  # we should pick some default


def test_log_metrics_params_tags(mlflow_client, backend_store_uri):
    experiment_id = mlflow_client.create_experiment('Oh My')
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    mlflow_client.log_metric(run_id, key='metric', value=123.456, timestamp=789, step=2)
    mlflow_client.log_metric(run_id, key='stepless-metric', value=987.654, timestamp=321)
    mlflow_client.log_param(run_id, 'param', 'value')
    mlflow_client.set_tag(run_id, 'taggity', 'do-dah')
    run = mlflow_client.get_run(run_id)
    assert run.data.metrics.get('metric') == 123.456
    assert run.data.metrics.get('stepless-metric') == 987.654
    assert run.data.params.get('param') == 'value'
    assert run.data.tags.get('taggity') == 'do-dah'
    metric_history0 = mlflow_client.get_metric_history(run_id, "metric")
    assert len(metric_history0) == 1
    metric0 = metric_history0[0]
    assert metric0.key == "metric"
    assert metric0.value == 123.456
    assert metric0.timestamp == 789
    assert metric0.step == 2
    metric_history1 = mlflow_client.get_metric_history(run_id, "stepless-metric")
    assert len(metric_history1) == 1
    metric1 = metric_history1[0]
    assert metric1.key == "stepless-metric"
    assert metric1.value == 987.654
    assert metric1.timestamp == 321
    assert metric1.step == 0


def test_log_batch(mlflow_client, backend_store_uri):
    experiment_id = mlflow_client.create_experiment('Batch em up')
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    mlflow_client.log_batch(
        run_id=run_id,
        metrics=[Metric("metric", 123.456, 789, 3)], params=[Param("param", "value")],
        tags=[RunTag("taggity", "do-dah")])
    run = mlflow_client.get_run(run_id)
    assert run.data.metrics.get('metric') == 123.456
    assert run.data.params.get('param') == 'value'
    assert run.data.tags.get('taggity') == 'do-dah'
    metric_history = mlflow_client.get_metric_history(run_id, "metric")
    assert len(metric_history) == 1
    metric = metric_history[0]
    assert metric.key == "metric"
    assert metric.value == 123.456
    assert metric.timestamp == 789
    assert metric.step == 3


def test_set_terminated_defaults(mlflow_client):
    experiment_id = mlflow_client.create_experiment('Terminator 1')
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    assert RunStatus.to_string(mlflow_client.get_run(run_id).info.status) == 'RUNNING'
    assert mlflow_client.get_run(run_id).info.end_time is None
    mlflow_client.set_terminated(run_id)
    assert RunStatus.to_string(mlflow_client.get_run(run_id).info.status) == 'FINISHED'
    assert mlflow_client.get_run(run_id).info.end_time <= int(time.time() * 1000)


def test_set_terminated_status(mlflow_client):
    experiment_id = mlflow_client.create_experiment('Terminator 2')
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    assert RunStatus.to_string(mlflow_client.get_run(run_id).info.status) == 'RUNNING'
    assert mlflow_client.get_run(run_id).info.end_time is None
    mlflow_client.set_terminated(run_id, 'FAILED')
    assert RunStatus.to_string(mlflow_client.get_run(run_id).info.status) == 'FAILED'
    assert mlflow_client.get_run(run_id).info.end_time <= int(time.time() * 1000)


def test_artifacts(mlflow_client):
    experiment_id = mlflow_client.create_experiment('Art In Fact')
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    src_dir = tempfile.mkdtemp('test_artifacts_src')
    src_file = os.path.join(src_dir, 'my.file')
    with open(src_file, 'w') as f:
        f.write('Hello, World!')
    mlflow_client.log_artifact(run_id, src_file, None)
    mlflow_client.log_artifacts(run_id, src_dir, 'dir')

    root_artifacts_list = mlflow_client.list_artifacts(run_id)
    assert set([a.path for a in root_artifacts_list]) == {'my.file', 'dir'}

    dir_artifacts_list = mlflow_client.list_artifacts(run_id, 'dir')
    assert set([a.path for a in dir_artifacts_list]) == {'dir/my.file'}

    all_artifacts = mlflow_client.download_artifacts(run_id, '.')
    assert open('%s/my.file' % all_artifacts, 'r').read() == 'Hello, World!'
    assert open('%s/dir/my.file' % all_artifacts, 'r').read() == 'Hello, World!'

    dir_artifacts = mlflow_client.download_artifacts(run_id, 'dir')
    assert open('%s/my.file' % dir_artifacts, 'r').read() == 'Hello, World!'
