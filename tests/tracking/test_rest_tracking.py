"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures we can use the tracking API to communicate with it.
"""

import mock
from multiprocessing import Process
import os
import pytest
import socket
import time
import tempfile

from click.testing import CliRunner

import mlflow.experiments
from mlflow.entities import RunStatus
from mlflow.protos.service_pb2 import LOCAL as SOURCE_TYPE_LOCAL
from mlflow.server import app, BACKEND_STORE_URI_ENV_VAR
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_PARENT_RUN_ID, MLFLOW_SOURCE_TYPE, \
    MLFLOW_SOURCE_NAME, MLFLOW_PROJECT_ENTRY_POINT, MLFLOW_GIT_COMMIT


LOCALHOST = '127.0.0.1'
SERVER_PORT = 0


def _get_safe_port():
    """Returns an ephemeral port that is very likely to be free to bind to."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((LOCALHOST, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


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
    while process.is_alive() and time.time() - start_time < timeout:
        time.sleep(0.5)
    if process.is_alive():
        raise Exception('Server failed to shutdown after %s seconds' % timeout)


@pytest.fixture(scope="module", autouse=True)
def init_and_tear_down_server(request):
    """
    Once per run of the entire set of tests, we create a new server, and
    clean it up at the end.
    """
    mlflow.set_tracking_uri(None)
    global SERVER_PORT
    SERVER_PORT = _get_safe_port()
    file_store_path = tempfile.mkdtemp("test_rest_tracking_file_store")
    env = {BACKEND_STORE_URI_ENV_VAR: file_store_path}
    with mock.patch.dict(os.environ, env):
        process = Process(target=lambda: app.run(LOCALHOST, SERVER_PORT))
        process.start()
    _await_server_up_or_die(SERVER_PORT)

    # Yielding here causes pytest to resume execution at the end of all tests.
    yield

    print("Terminating server...")
    process.terminate()
    _await_server_down_or_die(process)


@pytest.fixture()
def tracking_server_uri():
    """Provides a tracking URI for communicating with the local tracking server."""
    return "http://{hostname}:{port}".format(hostname=LOCALHOST, port=SERVER_PORT)


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
    assert set([e.name for e in experiments]) == {'My Experiment'}


def test_delete_restore_experiment(mlflow_client):
    experiment_id = mlflow_client.create_experiment('Deleterious')
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == 'active'
    mlflow_client.delete_experiment(experiment_id)
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == 'deleted'
    mlflow_client.restore_experiment(experiment_id)
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == 'active'


def test_delete_restore_experiment_cli(mlflow_client, cli_env):
    experiment_name = "DeleteriousCLI"
    CliRunner(env=cli_env).invoke(mlflow.experiments.commands, ['create', experiment_name])
    experiment_id = mlflow_client.get_experiment_by_name(experiment_name).experiment_id
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == 'active'
    CliRunner(env=cli_env).invoke(mlflow.experiments.commands, ['delete', str(experiment_id)])
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == 'deleted'
    CliRunner(env=cli_env).invoke(mlflow.experiments.commands, ['restore', str(experiment_id)])
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == 'active'


def test_rename_experiment(mlflow_client):
    experiment_id = mlflow_client.create_experiment('BadName')
    assert mlflow_client.get_experiment(experiment_id).name == 'BadName'
    mlflow_client.rename_experiment(experiment_id, 'GoodName')
    assert mlflow_client.get_experiment(experiment_id).name == 'GoodName'


def test_rename_experiment_cli(mlflow_client, cli_env):
    bad_experiment_name = "BadName"
    good_experiment_name = "GoodName"

    CliRunner(env=cli_env).invoke(mlflow.experiments.commands, ['create', bad_experiment_name])
    experiment_id = mlflow_client.get_experiment_by_name(bad_experiment_name).experiment_id
    assert mlflow_client.get_experiment(experiment_id).name == bad_experiment_name
    CliRunner(env=cli_env).invoke(
            mlflow.experiments.commands,
            ['rename', str(experiment_id), good_experiment_name])
    assert mlflow_client.get_experiment(experiment_id).name == good_experiment_name


def test_create_run_all_args(mlflow_client):
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
        }
    }
    experiment_id = mlflow_client.create_experiment('Run A Lot')
    created_run = mlflow_client.create_run(experiment_id, **create_run_kwargs)
    run_id = created_run.info.run_uuid
    print("Run id=%s" % run_id)

    run = mlflow_client.get_run(run_id)
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

    assert mlflow_client.list_run_infos(experiment_id) == [run.info]


def test_create_run_defaults(mlflow_client):
    experiment_id = mlflow_client.create_experiment('Run A Little')
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_uuid
    run = mlflow_client.get_run(run_id)
    assert run.info.run_uuid == run_id
    assert run.info.experiment_id == experiment_id
    assert run.info.user_id is not None  # we should pick some default


def test_log_metrics_params_tags(mlflow_client):
    experiment_id = mlflow_client.create_experiment('Oh My')
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_uuid
    mlflow_client.log_metric(run_id, 'metric', 123.456)
    mlflow_client.log_param(run_id, 'param', 'value')
    mlflow_client.set_tag(run_id, 'taggity', 'do-dah')
    run = mlflow_client.get_run(run_id)
    assert run.data.metrics.get('metric') == 123.456
    assert run.data.params.get('param') == 'value'
    assert run.data.tags.get('taggity') == 'do-dah'


def test_set_terminated_defaults(mlflow_client):
    experiment_id = mlflow_client.create_experiment('Terminator 1')
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_uuid
    assert RunStatus.to_string(mlflow_client.get_run(run_id).info.status) == 'RUNNING'
    assert mlflow_client.get_run(run_id).info.end_time is None
    mlflow_client.set_terminated(run_id)
    assert RunStatus.to_string(mlflow_client.get_run(run_id).info.status) == 'FINISHED'
    assert mlflow_client.get_run(run_id).info.end_time <= int(time.time() * 1000)


def test_set_terminated_status(mlflow_client):
    experiment_id = mlflow_client.create_experiment('Terminator 2')
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_uuid
    assert RunStatus.to_string(mlflow_client.get_run(run_id).info.status) == 'RUNNING'
    assert mlflow_client.get_run(run_id).info.end_time is None
    mlflow_client.set_terminated(run_id, 'FAILED')
    assert RunStatus.to_string(mlflow_client.get_run(run_id).info.status) == 'FAILED'
    assert mlflow_client.get_run(run_id).info.end_time <= int(time.time() * 1000)


def test_artifacts(mlflow_client):
    experiment_id = mlflow_client.create_experiment('Art In Fact')
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_uuid
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
