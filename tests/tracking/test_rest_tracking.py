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
import unittest

from mlflow.server import app, FILE_STORE_ENV_VAR
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_PARENT_RUN_ID


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
    global SERVER_PORT
    SERVER_PORT = _get_safe_port()
    file_store_path = tempfile.mkdtemp("test_rest_tracking_file_store")
    env = {FILE_STORE_ENV_VAR: file_store_path}
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
def mlflow_client():
    """Provides an MLflow Tracking API client pointed at the local server."""
    return MlflowClient("%s:%s" % (LOCALHOST, SERVER_PORT))


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


def test_rename_experiment(mlflow_client):
    experiment_id = mlflow_client.create_experiment('BadName')
    assert mlflow_client.get_experiment(experiment_id).name == 'BadName'
    mlflow_client.rename_experiment(experiment_id, 'GoodName')
    assert mlflow_client.get_experiment(experiment_id).name == 'GoodName'


def test_create_run_all_args(mlflow_client):
    experiment_id = mlflow_client.create_experiment('Run A Lot')
    expected_tags = {'my': 'tag', 'other': 'tag'}
    created_run = mlflow_client.create_run(
        experiment_id, user_id=123, run_name='My name', source_type='LOCAL',
        source_name='Hello', entry_point_name='entry', start_time=456,
        source_version='abc', tags=expected_tags, parent_run_id=7)
    run_id = created_run.info.run_uuid
    print("Run id=%s" % run_id)
    run = mlflow_client.get_run(run_id)
    assert run.info.run_uuid == run_id
    assert run.info.experiment_id == experiment_id
    assert run.info.user_id == 123
    assert run.info.source_type == 'LOCAL'
    assert run.info.source_name == 'Hello'
    assert run.info.entry_point_name == 'entry'
    assert run.info.start_time == 456
    assert run.info.source_version == 'abc'
    actual_tags = {t.key: t.value for t in run.data.tags}
    for tag in expected_tags:
        assert tag in actual_tags
    assert actual_tags.get(MLFLOW_RUN_NAME) == 'My name'
    assert actual_tags.get(MLFLOW_PARENT_RUN_ID) == '7'

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
    metrics = {t.key: t.value for t in run.data.metrics}
    params = {t.key: t.value for t in run.data.params}
    tags = {t.key: t.value for t in run.data.tags}
    assert metrics.get('metric') == 123.456
    assert params.get('param') == 'value'
    assert tags.get('taggity') == 'do-dah'


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
