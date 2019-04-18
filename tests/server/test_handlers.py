import json

import mock
import pytest

import mlflow
from mlflow.entities import ViewType, Metric, RunTag, Param
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.server.handlers import get_endpoints, _create_experiment, _get_request_message, \
    _search_runs, _log_batch, catch_mlflow_exception
from mlflow.protos.service_pb2 import CreateExperiment, SearchRuns, LogBatch
from mlflow.store.file_store import FileStore
from mlflow.utils.validation import MAX_BATCH_LOG_REQUEST_SIZE


@pytest.fixture()
def mock_get_request_message():
    with mock.patch('mlflow.server.handlers._get_request_message') as m:
        yield m


@pytest.fixture()
def mock_get_request_json():
    with mock.patch('mlflow.server.handlers._get_request_json') as m:
        yield m


@pytest.fixture()
def mock_store():
    with mock.patch('mlflow.server.handlers._get_store') as m:
        mock_store = mock.MagicMock()
        m.return_value = mock_store
        yield mock_store


def test_get_endpoints():
    endpoints = get_endpoints()
    create_experiment_endpoint = [e for e in endpoints if e[1] == _create_experiment]
    assert len(create_experiment_endpoint) == 2


def test_can_parse_json():
    request = mock.MagicMock()
    request.method = "POST"
    request.get_json = mock.MagicMock()
    request.get_json.return_value = {"name": "hello"}
    msg = _get_request_message(CreateExperiment(), flask_request=request)
    assert msg.name == "hello"


def test_can_parse_post_json_with_unknown_fields():
    request = mock.MagicMock()
    request.method = "POST"
    request.get_json = mock.MagicMock()
    request.get_json.return_value = {"name": "hello", "WHAT IS THIS FIELD EVEN": "DOING"}
    msg = _get_request_message(CreateExperiment(), flask_request=request)
    assert msg.name == "hello"


def test_can_parse_get_json_with_unknown_fields():
    request = mock.MagicMock()
    request.method = "GET"
    request.query_string = b"name=hello&superDuperUnknown=field"
    msg = _get_request_message(CreateExperiment(), flask_request=request)
    assert msg.name == "hello"


# Previous versions of the client sent a doubly string encoded JSON blob,
# so this test ensures continued compliance with such clients.
def test_can_parse_json_string():
    request = mock.MagicMock()
    request.method = "POST"
    request.get_json = mock.MagicMock()
    request.get_json.return_value = '{"name": "hello2"}'
    msg = _get_request_message(CreateExperiment(), flask_request=request)
    assert msg.name == "hello2"


def test_search_runs_default_view_type(mock_get_request_message, mock_store):
    """
    Search Runs default view type is filled in as ViewType.ACTIVE_ONLY
    """
    mock_get_request_message.return_value = SearchRuns(experiment_ids=[0], anded_expressions=[])
    _search_runs()
    args, _ = mock_store.search_runs.call_args
    assert args[2] == ViewType.ACTIVE_ONLY


def _assert_logged_entities(run_id, metric_entities, param_entities, tag_entities):
    client = mlflow.tracking.MlflowClient()
    store = mlflow.tracking.utils._get_store()
    run = client.get_run(run_id)
    # Assert logged metrics
    all_logged_metrics = sum([store.get_metric_history(run_id, key)
                              for key in run.data.metrics], [])
    assert len(all_logged_metrics) == len(metric_entities)
    logged_metrics = [(m.key, m.value, m.timestamp) for m in all_logged_metrics]
    assert set(logged_metrics) == set([(m.key, m.value, m.timestamp) for m in metric_entities])
    logged_tags = set([(tag_key, tag_value) for tag_key, tag_value in run.data.tags.items()])
    assert set([(tag.key, tag.value) for tag in tag_entities]) <= logged_tags
    assert len(run.data.params) == len(param_entities)
    logged_params = [(param_key, param_val) for param_key, param_val in run.data.params.items()]
    assert set(logged_params) == set([(param.key, param.value) for param in param_entities])


def test_log_batch_handler_success(mock_get_request_message, mock_get_request_json, tmpdir):
    # Test success cases for the LogBatch API
    def _test_log_batch_helper_success(
            metric_entities, param_entities, tag_entities,
            expected_metrics=None, expected_params=None, expected_tags=None):
        """
        Simulates a LogBatch API request using the provided metrics/params/tags, asserting that it
        succeeds & that the backing store contains either the set of expected metrics/params/tags
        (if provided) or, by default, the metrics/params/tags used in the API request.
        """
        with mlflow.start_run() as active_run:
            run_id = active_run.info.run_uuid
            mock_get_request_message.return_value = LogBatch(
                run_id=run_id,
                metrics=[m.to_proto() for m in metric_entities],
                params=[p.to_proto() for p in param_entities],
                tags=[t.to_proto() for t in tag_entities])
            response = _log_batch()
            assert response.status_code == 200
            json_response = json.loads(response.get_data())
            assert json_response == {}
            _assert_logged_entities(
                run_id, expected_metrics or metric_entities, expected_params or param_entities,
                expected_tags or tag_entities)

    store = FileStore(tmpdir.strpath)
    mock_get_request_json.return_value = "{}"  # Mock request JSON so it passes length validation
    server_patch = mock.patch('mlflow.server.handlers._get_store', return_value=store)
    client_patch = mock.patch('mlflow.tracking.utils._get_store', return_value=store)
    with server_patch, client_patch:
        mlflow.set_experiment("log-batch-experiment")
        # Log an empty payload
        _test_log_batch_helper_success([], [], [])
        # Log multiple metrics/params/tags
        _test_log_batch_helper_success(
            metric_entities=[Metric(key="m-key", value=3.2 * i, timestamp=i) for i in range(3)],
            param_entities=[Param(key="p-key-%s" % i, value="p-val-%s" % i) for i in range(4)],
            tag_entities=[RunTag(key="t-key-%s" % i, value="t-val-%s" % i) for i in range(5)])
        # Log metrics with the same key
        _test_log_batch_helper_success(
            metric_entities=[Metric(key="m-key", value=3.2 * i, timestamp=3) for i in range(3)],
            param_entities=[], tag_entities=[])
        # Log tags with the same key, verify the last one gets written
        same_key_tags = [RunTag(key="t-key", value="t-val-%s" % i) for i in range(5)]
        _test_log_batch_helper_success(
            metric_entities=[], param_entities=[], tag_entities=same_key_tags,
            expected_tags=[same_key_tags[-1]])


def test_log_batch_api_req(mock_get_request_json):
    mock_get_request_json.return_value = "a" * (MAX_BATCH_LOG_REQUEST_SIZE + 1)
    response = _log_batch()
    assert response.status_code == 500
    json_response = json.loads(response.get_data())
    assert json_response["error_code"] == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert ("Batched logging API requests must be at most %s bytes" % MAX_BATCH_LOG_REQUEST_SIZE
            in json_response["message"])


def test_catch_mlflow_exception():
    @catch_mlflow_exception
    def test_handler():
        raise MlflowException('test error', error_code=INTERNAL_ERROR)

    # pylint: disable=assignment-from-no-return
    response = test_handler()
    json_response = json.loads(response.get_data())
    assert response.status_code == 500
    assert json_response['error_code'] == ErrorCode.Name(INTERNAL_ERROR)
    assert json_response['message'] == 'test error'
