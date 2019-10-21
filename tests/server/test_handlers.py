import json
import uuid

import mock
import pytest

import os
import mlflow
from mlflow.entities import ViewType
from mlflow.entities.model_registry import RegisteredModel, RegisteredModelDetailed, \
    ModelVersionDetailed, ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.server.handlers import get_endpoints, _create_experiment, _get_request_message, \
    _search_runs, _log_batch, catch_mlflow_exception, _create_registered_model, \
    _update_registered_model, _delete_registered_model, _get_registered_model_details, \
    _list_registered_models, _get_latest_versions, _create_model_version, _update_model_version, \
    _delete_model_version, _get_model_version_download_uri, _get_model_version_stages, \
    _search_model_versions, _get_model_version_details
from mlflow.server import BACKEND_STORE_URI_ENV_VAR
from mlflow.store.entities.paged_list import PagedList
from mlflow.protos.service_pb2 import CreateExperiment, SearchRuns
from mlflow.protos.model_registry_pb2 import CreateRegisteredModel, UpdateRegisteredModel, \
    DeleteRegisteredModel, ListRegisteredModels, GetRegisteredModelDetails, GetLatestVersions, \
    CreateModelVersion, UpdateModelVersion, DeleteModelVersion, GetModelVersionDetails, \
    GetModelVersionDownloadUri, SearchModelVersions, GetModelVersionStages
from mlflow.utils.proto_json_utils import message_to_json
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
def mock_tracking_store():
    with mock.patch('mlflow.server.handlers._get_tracking_store') as m:
        mock_store = mock.MagicMock()
        m.return_value = mock_store
        yield mock_store


@pytest.fixture()
def mock_model_registry_store():
    with mock.patch('mlflow.server.handlers._get_model_registry_store') as m:
        mock_store = mock.MagicMock()
        m.return_value = mock_store
        yield mock_store


def test_get_endpoints():
    endpoints = get_endpoints()
    create_experiment_endpoint = [e for e in endpoints if e[1] == _create_experiment]
    assert len(create_experiment_endpoint) == 4


def test_all_model_registry_endpoints_available():
    endpoints = {handler: method for (path, handler, method) in get_endpoints()}
    print(endpoints)

    # Test that each of the handler is enabled as an endpoint with appropriate method.
    expected_endpoints = {
        "POST": [
            _create_registered_model,
            _get_registered_model_details,
            _get_latest_versions,
            _create_model_version,
            _get_model_version_details,
            _get_model_version_stages,
            _get_model_version_download_uri,
        ],
        "PATCH": [
            _update_registered_model,
            _update_model_version,
        ],
        "DELETE": [
            _delete_registered_model,
            _delete_registered_model,
        ],
        "GET": [
            _list_registered_models,
            _search_model_versions,
        ]
    }
    # TODO: efficient mechanism to test endpoint path
    for method, handlers in expected_endpoints.items():
        for handler in handlers:
            assert handler in endpoints
            assert endpoints[handler] == [method]


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


def test_search_runs_default_view_type(mock_get_request_message, mock_tracking_store):
    """
    Search Runs default view type is filled in as ViewType.ACTIVE_ONLY
    """
    mock_get_request_message.return_value = SearchRuns(experiment_ids=["0"])
    mock_tracking_store.search_runs.return_value = PagedList([], None)
    _search_runs()
    args, _ = mock_tracking_store.search_runs.call_args
    assert args[2] == ViewType.ACTIVE_ONLY


def test_log_batch_api_req(mock_get_request_json):
    mock_get_request_json.return_value = "a" * (MAX_BATCH_LOG_REQUEST_SIZE + 1)
    response = _log_batch()
    assert response.status_code == 400
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


@pytest.mark.large
def test_mlflow_server_with_installed_plugin(tmpdir):
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""
    from mlflow_test_plugin import PluginFileStore

    env = {
        BACKEND_STORE_URI_ENV_VAR: "file-plugin:%s" % tmpdir.strpath,
    }
    with mock.patch.dict(os.environ, env):
        mlflow.server.handlers._tracking_store = None
        try:
            plugin_file_store = mlflow.server.handlers._get_tracking_store()
        finally:
            mlflow.server.handlers._tracking_store = None
        assert isinstance(plugin_file_store, PluginFileStore)
        assert plugin_file_store.is_plugin


def jsonify(obj):
    def _jsonify(obj):
        return json.loads(message_to_json(obj.to_proto()))
    if isinstance(obj, list):
        return [_jsonify(o) for o in obj]
    else:
        return _jsonify(obj)


# Tests for Model Registry handlers
def test_create_registered_model(mock_get_request_message, mock_model_registry_store):
    mock_get_request_message.return_value = CreateRegisteredModel(name="model_1")
    rm = RegisteredModel("model_1")
    mock_model_registry_store.create_registered_model.return_value = rm
    resp = _create_registered_model()
    args, _ = mock_model_registry_store.create_registered_model.call_args
    assert args == ("model_1", )
    assert json.loads(resp.get_data()) == {"registered_model": jsonify(rm)}


def test_get_registered_model_details(mock_get_request_message, mock_model_registry_store):
    rm = RegisteredModel("model1")
    mock_get_request_message.return_value = GetRegisteredModelDetails(
        registered_model=rm.to_proto())
    rmd = RegisteredModelDetailed(name="model_1", creation_timestamp=111,
                                  last_updated_timestamp=222, description="Test model",
                                  latest_versions=[])
    mock_model_registry_store.get_registered_model_details.return_value = rmd
    resp = _get_registered_model_details()
    args, _ = mock_model_registry_store.get_registered_model_details.call_args
    assert args == (rm, )
    assert json.loads(resp.get_data()) == {"registered_model_detailed": jsonify(rmd)}


def test_update_registered_model(mock_get_request_message, mock_model_registry_store):
    rm1 = RegisteredModel("model_1")
    mock_get_request_message.return_value = UpdateRegisteredModel(registered_model=rm1.to_proto(),
                                                                  name="model_2",
                                                                  description="Test model")
    rm2 = RegisteredModel("model_2")
    mock_model_registry_store.update_registered_model.return_value = rm2
    resp = _update_registered_model()
    args, _ = mock_model_registry_store.update_registered_model.call_args
    assert args == (rm1, u"model_2", u"Test model")
    assert json.loads(resp.get_data()) == {"registered_model": jsonify(rm2)}


def test_delete_registered_model(mock_get_request_message, mock_model_registry_store):
    rm = RegisteredModel("model_1")
    mock_get_request_message.return_value = DeleteRegisteredModel(registered_model=rm.to_proto())
    _delete_registered_model()
    args, _ = mock_model_registry_store.delete_registered_model.call_args
    assert args == (rm, )


def test_list_registered_models(mock_get_request_message, mock_model_registry_store):
    mock_get_request_message.return_value = ListRegisteredModels()
    rmds = [
        RegisteredModelDetailed(name="model_1", creation_timestamp=111,
                                last_updated_timestamp=222, description="Test model",
                                latest_versions=[]),
        RegisteredModelDetailed(name="model_2", creation_timestamp=111,
                                last_updated_timestamp=333, description="Another model",
                                latest_versions=[]),
    ]
    mock_model_registry_store.list_registered_models.return_value = rmds
    resp = _list_registered_models()
    args, _ = mock_model_registry_store.list_registered_models.call_args
    assert args == ()
    assert json.loads(resp.get_data()) == {"registered_models_detailed": jsonify(rmds)}


def test_get_latest_versions(mock_get_request_message, mock_model_registry_store):
    rm = RegisteredModel("model1")
    mock_get_request_message.return_value = GetLatestVersions(registered_model=rm.to_proto())
    mvds = [
        ModelVersionDetailed(registered_model=rm, version=5, creation_timestamp=1,
                             last_updated_timestamp=12, description="v 5", user_id="u1",
                             current_stage="Production", source="A/B", run_id=uuid.uuid4().hex,
                             status="READY", status_message=None),
        ModelVersionDetailed(registered_model=rm, version=1, creation_timestamp=1,
                             last_updated_timestamp=1200, description="v 1", user_id="u1",
                             current_stage="Archived", source="A/B2", run_id=uuid.uuid4().hex,
                             status="READY", status_message=None),
        ModelVersionDetailed(registered_model=rm, version=12, creation_timestamp=100,
                             last_updated_timestamp=None, description="v 12", user_id="u2",
                             current_stage="Staging", source="A/B3", run_id=uuid.uuid4().hex,
                             status="READY", status_message=None),
    ]
    mock_model_registry_store.get_latest_versions.return_value = mvds
    resp = _get_latest_versions()
    args, _ = mock_model_registry_store.get_latest_versions.call_args
    assert args == (rm, )
    assert json.loads(resp.get_data()) == {"model_versions_detailed": jsonify(mvds)}


def test_create_model_version(mock_get_request_message, mock_model_registry_store):
    run_id = uuid.uuid4().hex
    mock_get_request_message.return_value = CreateModelVersion(name="model_1", source="A/B",
                                                               run_id=run_id)
    mv = ModelVersion(registered_model=RegisteredModel(name="model_1"), version=12)
    mock_model_registry_store.create_model_version.return_value = mv
    resp = _create_model_version()
    args, _ = mock_model_registry_store.create_model_version.call_args
    assert args == ("model_1", "A/B", run_id)
    assert json.loads(resp.get_data()) == {"model_version": jsonify(mv)}


def test_get_model_version_details(mock_get_request_message, mock_model_registry_store):
    rm = RegisteredModel("model1")
    mv = ModelVersion(registered_model=rm, version=32)
    mock_get_request_message.return_value = GetModelVersionDetails(model_version=mv.to_proto())
    mvd = ModelVersionDetailed(registered_model=rm, version=5, creation_timestamp=1,
                               last_updated_timestamp=12, description="v 5", user_id="u1",
                               current_stage="Production", source="A/B", run_id=uuid.uuid4().hex,
                               status="READY", status_message=None)
    mock_model_registry_store.get_model_version_details.return_value = mvd
    resp = _get_model_version_details()
    args, _ = mock_model_registry_store.get_model_version_details.call_args
    assert args == (mv, )
    assert json.loads(resp.get_data()) == {"model_version_detailed": jsonify(mvd)}


def test_update_model_version(mock_get_request_message, mock_model_registry_store):
    rm = RegisteredModel("model1")
    mv = ModelVersion(registered_model=rm, version=32)
    mock_get_request_message.return_value = UpdateModelVersion(model_version=mv.to_proto(),
                                                               stage="Production",
                                                               description="Great model!")
    _update_model_version()
    args, _ = mock_model_registry_store.update_model_version.call_args
    assert args == (mv, "Production", "Great model!")


def test_delete_model_version(mock_get_request_message, mock_model_registry_store):
    rm = RegisteredModel("model1")
    mv = ModelVersion(registered_model=rm, version=32)
    mock_get_request_message.return_value = DeleteModelVersion(model_version=mv.to_proto())
    _delete_model_version()
    args, _ = mock_model_registry_store.delete_model_version.call_args
    assert args == (mv, )


def test_get_model_version_download_uri(mock_get_request_message, mock_model_registry_store):
    rm = RegisteredModel("model1")
    mv = ModelVersion(registered_model=rm, version=32)
    mock_get_request_message.return_value = GetModelVersionDownloadUri(model_version=mv.to_proto())
    mock_model_registry_store.get_model_version_download_uri.return_value = "some/download/path"
    resp = _get_model_version_download_uri()
    args, _ = mock_model_registry_store.get_model_version_download_uri.call_args
    assert args == (mv, )
    assert json.loads(resp.get_data()) == {"artifact_uri": "some/download/path"}


def test_model_version_stages(mock_get_request_message, mock_model_registry_store):
    rm = RegisteredModel("model1")
    mv = ModelVersion(registered_model=rm, version=32)
    mock_get_request_message.return_value = GetModelVersionStages(model_version=mv.to_proto())
    stages = ["Stage1", "Production", "0", "5% traffic", "None"]
    mock_model_registry_store.get_model_version_stages.return_value = stages
    resp = _get_model_version_stages()
    args, _ = mock_model_registry_store.get_model_version_stages.call_args
    assert args == (mv, )
    assert json.loads(resp.get_data()) == {"stages": stages}


def test_search_model_versions(mock_get_request_message, mock_model_registry_store):
    mock_get_request_message.return_value = SearchModelVersions(filter="source_path = 'A/B/CD'")
    mvds = [
        ModelVersionDetailed(RegisteredModel(name="model_1"), version=5, creation_timestamp=100,
                             last_updated_timestamp=1200, description="v 5", user_id="u1",
                             current_stage="Production", source="A/B/CD", run_id=uuid.uuid4().hex,
                             status="READY", status_message=None),
        ModelVersionDetailed(RegisteredModel(name="model_1"), version=12, creation_timestamp=110,
                             last_updated_timestamp=2000, description="v 12", user_id="u2",
                             current_stage="Production", source="A/B/CD", run_id=uuid.uuid4().hex,
                             status="READY", status_message=None),
        ModelVersionDetailed(RegisteredModel(name="ads_model"), version=8, creation_timestamp=200,
                             last_updated_timestamp=2000, description="v 8", user_id="u1",
                             current_stage="Staging", source="A/B/CD", run_id=uuid.uuid4().hex,
                             status="READY", status_message=None),
        ModelVersionDetailed(RegisteredModel(name="fraud_detection_model"), version=345,
                             creation_timestamp=1000, last_updated_timestamp=1001,
                             description="newest version",  user_id="u12", current_stage="None",
                             source="A/B/CD",  run_id=uuid.uuid4().hex, status="READY",
                             status_message=None),
    ]
    mock_model_registry_store.search_model_versions.return_value = mvds
    resp = _search_model_versions()
    args, _ = mock_model_registry_store.search_model_versions.call_args
    assert args == ("source_path = 'A/B/CD'", )
    assert json.loads(resp.get_data()) == {"model_versions_detailed": jsonify(mvds)}
