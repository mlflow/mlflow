import json

import mock
import pytest

from mlflow.entities import ViewType, Metric, RunTag, Param
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, ErrorCode
from mlflow.server.handlers import get_endpoints, _create_experiment, _get_request_message, \
    _search_runs, _log_batch, catch_mlflow_exception
from mlflow.protos.service_pb2 import CreateExperiment, SearchRuns, LogBatch


@pytest.fixture()
def mock_get_request_message():
    with mock.patch('mlflow.server.handlers._get_request_message') as m:
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


def test_log_batch_handler(mock_get_request_message, mock_store):
    metrics = [Metric(key="my-metric-key", value=3.2, timestamp=1)]
    params = [Param(key="my-param-key", value="my-param-val")]
    tags = [RunTag(key="my-tag-key", value="my-tag-val")]
    mock_get_request_message.return_value = LogBatch(
        run_id="abc",
        metrics=[m.to_proto() for m in metrics],
        params=[p.to_proto() for p in params],
        tags=[t.to_proto() for t in tags])
    response = _log_batch()
    assert response.status_code == 200
    json_response = json.loads(response.get_data())
    assert json_response == {}
    _, kwargs = mock_store.log_batch.call_args
    assert kwargs["run_id"] == "abc"
    assert [dict(m) for m in kwargs["metrics"]] == [dict(m) for m in metrics]
    assert [dict(p) for p in kwargs["params"]] == [dict(p) for p in params]
    assert [dict(t) for t in kwargs["tags"]] == [dict(t) for t in tags]



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
