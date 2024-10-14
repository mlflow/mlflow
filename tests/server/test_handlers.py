import json
import uuid
from unittest import mock

import pytest

import mlflow
from mlflow.entities import ViewType
from mlflow.entities.model_registry import (
    ModelVersion,
    ModelVersionTag,
    RegisteredModel,
    RegisteredModelTag,
)
from mlflow.entities.trace_info import TraceInfo
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.protos.model_registry_pb2 import (
    CreateModelVersion,
    CreateRegisteredModel,
    DeleteModelVersion,
    DeleteModelVersionTag,
    DeleteRegisteredModel,
    DeleteRegisteredModelAlias,
    DeleteRegisteredModelTag,
    GetLatestVersions,
    GetModelVersion,
    GetModelVersionByAlias,
    GetModelVersionDownloadUri,
    GetRegisteredModel,
    RenameRegisteredModel,
    SearchModelVersions,
    SearchRegisteredModels,
    SetModelVersionTag,
    SetRegisteredModelAlias,
    SetRegisteredModelTag,
    TransitionModelVersionStage,
    UpdateModelVersion,
    UpdateRegisteredModel,
)
from mlflow.protos.service_pb2 import CreateExperiment, SearchRuns
from mlflow.server import (
    ARTIFACTS_DESTINATION_ENV_VAR,
    BACKEND_STORE_URI_ENV_VAR,
    SERVE_ARTIFACTS_ENV_VAR,
    app,
)
from mlflow.server.handlers import (
    _convert_path_parameter_to_flask_format,
    _create_experiment,
    _create_model_version,
    _create_registered_model,
    _delete_artifact_mlflow_artifacts,
    _delete_model_version,
    _delete_model_version_tag,
    _delete_registered_model,
    _delete_registered_model_alias,
    _delete_registered_model_tag,
    _get_latest_versions,
    _get_model_version,
    _get_model_version_by_alias,
    _get_model_version_download_uri,
    _get_registered_model,
    _get_request_message,
    _get_trace_artifact_repo,
    _log_batch,
    _rename_registered_model,
    _search_model_versions,
    _search_registered_models,
    _search_runs,
    _set_model_version_tag,
    _set_registered_model_alias,
    _set_registered_model_tag,
    _transition_stage,
    _update_model_version,
    _update_registered_model,
    _validate_source,
    catch_mlflow_exception,
    get_endpoints,
)
from mlflow.store.artifact.azure_blob_artifact_repo import AzureBlobArtifactRepository
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry import (
    SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD,
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
)
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.validation import MAX_BATCH_LOG_REQUEST_SIZE


@pytest.fixture
def mock_get_request_message():
    with mock.patch("mlflow.server.handlers._get_request_message") as m:
        yield m


@pytest.fixture
def mock_get_request_json():
    with mock.patch("mlflow.server.handlers._get_request_json") as m:
        yield m


@pytest.fixture
def mock_tracking_store():
    with mock.patch("mlflow.server.handlers._get_tracking_store") as m:
        mock_store = mock.MagicMock()
        m.return_value = mock_store
        yield mock_store


@pytest.fixture
def mock_model_registry_store():
    with mock.patch("mlflow.server.handlers._get_model_registry_store") as m:
        mock_store = mock.MagicMock()
        m.return_value = mock_store
        yield mock_store


@pytest.fixture
def enable_serve_artifacts(monkeypatch):
    monkeypatch.setenv(SERVE_ARTIFACTS_ENV_VAR, "true")


def test_health():
    with app.test_client() as c:
        response = c.get("/health")
        assert response.status_code == 200
        assert response.get_data().decode() == "OK"


def test_version():
    with app.test_client() as c:
        response = c.get("/version")
        assert response.status_code == 200
        assert response.get_data().decode() == mlflow.__version__


def test_get_endpoints():
    endpoints = get_endpoints()
    create_experiment_endpoint = [e for e in endpoints if e[1] == _create_experiment]
    assert len(create_experiment_endpoint) == 2


def test_convert_path_parameter_to_flask_format():
    converted = _convert_path_parameter_to_flask_format("/mlflow/trace")
    assert "/mlflow/trace" == converted

    converted = _convert_path_parameter_to_flask_format("/mlflow/trace/{request_id}")
    assert "/mlflow/trace/<request_id>" == converted

    converted = _convert_path_parameter_to_flask_format("/mlflow/{foo}/{bar}/{baz}")
    assert "/mlflow/<foo>/<bar>/<baz>" == converted


def test_all_model_registry_endpoints_available():
    endpoints = {handler: method for (path, handler, method) in get_endpoints()}

    # Test that each of the handler is enabled as an endpoint with appropriate method.
    expected_endpoints = {
        "POST": [
            _create_registered_model,
            _create_model_version,
            _rename_registered_model,
            _transition_stage,
        ],
        "PATCH": [_update_registered_model, _update_model_version],
        "DELETE": [_delete_registered_model, _delete_registered_model],
        "GET": [
            _search_model_versions,
            _get_latest_versions,
            _get_registered_model,
            _get_model_version,
            _get_model_version_download_uri,
        ],
    }
    # TODO: efficient mechanism to test endpoint path
    for method, handlers in expected_endpoints.items():
        for handler in handlers:
            assert handler in endpoints
            assert endpoints[handler] == [method]


def test_can_parse_json():
    request = mock.MagicMock()
    request.method = "POST"
    request.content_type = "application/json"
    request.get_json = mock.MagicMock()
    request.get_json.return_value = {"name": "hello"}
    msg = _get_request_message(CreateExperiment(), flask_request=request)
    assert msg.name == "hello"


def test_can_parse_post_json_with_unknown_fields():
    request = mock.MagicMock()
    request.method = "POST"
    request.content_type = "application/json"
    request.get_json = mock.MagicMock()
    request.get_json.return_value = {"name": "hello", "WHAT IS THIS FIELD EVEN": "DOING"}
    msg = _get_request_message(CreateExperiment(), flask_request=request)
    assert msg.name == "hello"


def test_can_parse_post_json_with_content_type_params():
    request = mock.MagicMock()
    request.method = "POST"
    request.content_type = "application/json; charset=utf-8"
    request.get_json = mock.MagicMock()
    request.get_json.return_value = {"name": "hello"}
    msg = _get_request_message(CreateExperiment(), flask_request=request)
    assert msg.name == "hello"


def test_can_parse_get_json_with_unknown_fields():
    request = mock.MagicMock()
    request.method = "GET"
    request.args = {"name": "hello", "superDuperUnknown": "field"}
    msg = _get_request_message(CreateExperiment(), flask_request=request)
    assert msg.name == "hello"


# Previous versions of the client sent a doubly string encoded JSON blob,
# so this test ensures continued compliance with such clients.
def test_can_parse_json_string():
    request = mock.MagicMock()
    request.method = "POST"
    request.content_type = "application/json"
    request.get_json = mock.MagicMock()
    request.get_json.return_value = '{"name": "hello2"}'
    msg = _get_request_message(CreateExperiment(), flask_request=request)
    assert msg.name == "hello2"


def test_can_block_post_request_with_invalid_content_type():
    request = mock.MagicMock()
    request.method = "POST"
    request.content_type = "text/plain"
    request.get_json = mock.MagicMock()
    request.get_json.return_value = {"name": "hello"}
    with pytest.raises(MlflowException, match=r"Bad Request. Content-Type"):
        _get_request_message(CreateExperiment(), flask_request=request)


def test_can_block_post_request_with_missing_content_type():
    request = mock.MagicMock()
    request.method = "POST"
    request.content_type = None
    request.get_json = mock.MagicMock()
    request.get_json.return_value = {"name": "hello"}
    with pytest.raises(MlflowException, match=r"Bad Request. Content-Type"):
        _get_request_message(CreateExperiment(), flask_request=request)


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
    assert (
        f"Batched logging API requests must be at most {MAX_BATCH_LOG_REQUEST_SIZE} bytes"
        in json_response["message"]
    )


def test_catch_mlflow_exception():
    @catch_mlflow_exception
    def test_handler():
        raise MlflowException("test error", error_code=INTERNAL_ERROR)

    response = test_handler()
    json_response = json.loads(response.get_data())
    assert response.status_code == 500
    assert json_response["error_code"] == ErrorCode.Name(INTERNAL_ERROR)
    assert json_response["message"] == "test error"


def test_mlflow_server_with_installed_plugin(tmp_path, monkeypatch):
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""
    from mlflow_test_plugin.file_store import PluginFileStore

    monkeypatch.setenv(BACKEND_STORE_URI_ENV_VAR, f"file-plugin:{tmp_path}")
    monkeypatch.setattr(mlflow.server.handlers, "_tracking_store", None)
    plugin_file_store = mlflow.server.handlers._get_tracking_store()
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
    tags = [
        RegisteredModelTag(key="key", value="value"),
        RegisteredModelTag(key="anotherKey", value="some other value"),
    ]
    mock_get_request_message.return_value = CreateRegisteredModel(
        name="model_1", tags=[tag.to_proto() for tag in tags]
    )
    rm = RegisteredModel("model_1", tags=tags)
    mock_model_registry_store.create_registered_model.return_value = rm
    resp = _create_registered_model()
    _, args = mock_model_registry_store.create_registered_model.call_args
    assert args["name"] == "model_1"
    assert {tag.key: tag.value for tag in args["tags"]} == {tag.key: tag.value for tag in tags}
    assert json.loads(resp.get_data()) == {"registered_model": jsonify(rm)}


def test_get_registered_model(mock_get_request_message, mock_model_registry_store):
    name = "model1"
    mock_get_request_message.return_value = GetRegisteredModel(name=name)
    rmd = RegisteredModel(
        name=name,
        creation_timestamp=111,
        last_updated_timestamp=222,
        description="Test model",
        latest_versions=[],
    )
    mock_model_registry_store.get_registered_model.return_value = rmd
    resp = _get_registered_model()
    _, args = mock_model_registry_store.get_registered_model.call_args
    assert args == {"name": name}
    assert json.loads(resp.get_data()) == {"registered_model": jsonify(rmd)}


def test_update_registered_model(mock_get_request_message, mock_model_registry_store):
    name = "model_1"
    description = "Test model"
    mock_get_request_message.return_value = UpdateRegisteredModel(
        name=name, description=description
    )
    rm2 = RegisteredModel(name, description=description)
    mock_model_registry_store.update_registered_model.return_value = rm2
    resp = _update_registered_model()
    _, args = mock_model_registry_store.update_registered_model.call_args
    assert args == {"name": name, "description": "Test model"}
    assert json.loads(resp.get_data()) == {"registered_model": jsonify(rm2)}


def test_rename_registered_model(mock_get_request_message, mock_model_registry_store):
    name = "model_1"
    new_name = "model_2"
    mock_get_request_message.return_value = RenameRegisteredModel(name=name, new_name=new_name)
    rm2 = RegisteredModel(new_name)
    mock_model_registry_store.rename_registered_model.return_value = rm2
    resp = _rename_registered_model()
    _, args = mock_model_registry_store.rename_registered_model.call_args
    assert args == {"name": name, "new_name": new_name}
    assert json.loads(resp.get_data()) == {"registered_model": jsonify(rm2)}


def test_delete_registered_model(mock_get_request_message, mock_model_registry_store):
    name = "model_1"
    mock_get_request_message.return_value = DeleteRegisteredModel(name=name)
    _delete_registered_model()
    _, args = mock_model_registry_store.delete_registered_model.call_args
    assert args == {"name": name}


def test_search_registered_models(mock_get_request_message, mock_model_registry_store):
    rmds = [
        RegisteredModel(
            name="model_1",
            creation_timestamp=111,
            last_updated_timestamp=222,
            description="Test model",
            latest_versions=[],
        ),
        RegisteredModel(
            name="model_2",
            creation_timestamp=111,
            last_updated_timestamp=333,
            description="Another model",
            latest_versions=[],
        ),
    ]
    mock_get_request_message.return_value = SearchRegisteredModels()
    mock_model_registry_store.search_registered_models.return_value = PagedList(rmds, None)
    resp = _search_registered_models()
    _, args = mock_model_registry_store.search_registered_models.call_args
    assert args == {
        "filter_string": "",
        "max_results": SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
        "order_by": [],
        "page_token": "",
    }
    assert json.loads(resp.get_data()) == {"registered_models": jsonify(rmds)}

    mock_get_request_message.return_value = SearchRegisteredModels(filter="hello")
    mock_model_registry_store.search_registered_models.return_value = PagedList(rmds[:1], "tok")
    resp = _search_registered_models()
    _, args = mock_model_registry_store.search_registered_models.call_args
    assert args == {
        "filter_string": "hello",
        "max_results": SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
        "order_by": [],
        "page_token": "",
    }
    assert json.loads(resp.get_data()) == {
        "registered_models": jsonify(rmds[:1]),
        "next_page_token": "tok",
    }

    mock_get_request_message.return_value = SearchRegisteredModels(filter="hi", max_results=5)
    mock_model_registry_store.search_registered_models.return_value = PagedList([rmds[0]], "tik")
    resp = _search_registered_models()
    _, args = mock_model_registry_store.search_registered_models.call_args
    assert args == {"filter_string": "hi", "max_results": 5, "order_by": [], "page_token": ""}
    assert json.loads(resp.get_data()) == {
        "registered_models": jsonify([rmds[0]]),
        "next_page_token": "tik",
    }

    mock_get_request_message.return_value = SearchRegisteredModels(
        filter="hey", max_results=500, order_by=["a", "B desc"], page_token="prev"
    )
    mock_model_registry_store.search_registered_models.return_value = PagedList(rmds, "DONE")
    resp = _search_registered_models()
    _, args = mock_model_registry_store.search_registered_models.call_args
    assert args == {
        "filter_string": "hey",
        "max_results": 500,
        "order_by": ["a", "B desc"],
        "page_token": "prev",
    }
    assert json.loads(resp.get_data()) == {
        "registered_models": jsonify(rmds),
        "next_page_token": "DONE",
    }


def test_get_latest_versions(mock_get_request_message, mock_model_registry_store):
    name = "model1"
    mock_get_request_message.return_value = GetLatestVersions(name=name)
    mvds = [
        ModelVersion(
            name=name,
            version="5",
            creation_timestamp=1,
            last_updated_timestamp=12,
            description="v 5",
            user_id="u1",
            current_stage="Production",
            source="A/B",
            run_id=uuid.uuid4().hex,
            status="READY",
            status_message=None,
        ),
        ModelVersion(
            name=name,
            version="1",
            creation_timestamp=1,
            last_updated_timestamp=1200,
            description="v 1",
            user_id="u1",
            current_stage="Archived",
            source="A/B2",
            run_id=uuid.uuid4().hex,
            status="READY",
            status_message=None,
        ),
        ModelVersion(
            name=name,
            version="12",
            creation_timestamp=100,
            last_updated_timestamp=None,
            description="v 12",
            user_id="u2",
            current_stage="Staging",
            source="A/B3",
            run_id=uuid.uuid4().hex,
            status="READY",
            status_message=None,
        ),
    ]
    mock_model_registry_store.get_latest_versions.return_value = mvds
    resp = _get_latest_versions()
    _, args = mock_model_registry_store.get_latest_versions.call_args
    assert args == {"name": name, "stages": []}
    assert json.loads(resp.get_data()) == {"model_versions": jsonify(mvds)}

    for stages in [[], ["None"], ["Staging"], ["Staging", "Production"]]:
        mock_get_request_message.return_value = GetLatestVersions(name=name, stages=stages)
        _get_latest_versions()
        _, args = mock_model_registry_store.get_latest_versions.call_args
        assert args == {"name": name, "stages": stages}


def test_create_model_version(mock_get_request_message, mock_model_registry_store):
    run_id = uuid.uuid4().hex
    tags = [
        ModelVersionTag(key="key", value="value"),
        ModelVersionTag(key="anotherKey", value="some other value"),
    ]
    run_link = "localhost:5000/path/to/run"
    mock_get_request_message.return_value = CreateModelVersion(
        name="model_1",
        source=f"runs:/{run_id}",
        run_id=run_id,
        run_link=run_link,
        tags=[tag.to_proto() for tag in tags],
    )
    mv = ModelVersion(
        name="model_1", version="12", creation_timestamp=123, tags=tags, run_link=run_link
    )
    mock_model_registry_store.create_model_version.return_value = mv
    resp = _create_model_version()
    _, args = mock_model_registry_store.create_model_version.call_args
    assert args["name"] == "model_1"
    assert args["source"] == f"runs:/{run_id}"
    assert args["run_id"] == run_id
    assert {tag.key: tag.value for tag in args["tags"]} == {tag.key: tag.value for tag in tags}
    assert args["run_link"] == run_link
    assert json.loads(resp.get_data()) == {"model_version": jsonify(mv)}


def test_set_registered_model_tag(mock_get_request_message, mock_model_registry_store):
    name = "model1"
    tag = RegisteredModelTag(key="some weird key", value="some value")
    mock_get_request_message.return_value = SetRegisteredModelTag(
        name=name, key=tag.key, value=tag.value
    )
    _set_registered_model_tag()
    _, args = mock_model_registry_store.set_registered_model_tag.call_args
    assert args == {"name": name, "tag": tag}


def test_delete_registered_model_tag(mock_get_request_message, mock_model_registry_store):
    name = "model1"
    key = "some weird key"
    mock_get_request_message.return_value = DeleteRegisteredModelTag(name=name, key=key)
    _delete_registered_model_tag()
    _, args = mock_model_registry_store.delete_registered_model_tag.call_args
    assert args == {"name": name, "key": key}


def test_get_model_version_details(mock_get_request_message, mock_model_registry_store):
    mock_get_request_message.return_value = GetModelVersion(name="model1", version="32")
    mvd = ModelVersion(
        name="model1",
        version="5",
        creation_timestamp=1,
        last_updated_timestamp=12,
        description="v 5",
        user_id="u1",
        current_stage="Production",
        source="A/B",
        run_id=uuid.uuid4().hex,
        status="READY",
        status_message=None,
    )
    mock_model_registry_store.get_model_version.return_value = mvd
    resp = _get_model_version()
    _, args = mock_model_registry_store.get_model_version.call_args
    assert args == {"name": "model1", "version": "32"}
    assert json.loads(resp.get_data()) == {"model_version": jsonify(mvd)}


def test_update_model_version(mock_get_request_message, mock_model_registry_store):
    name = "model1"
    version = "32"
    description = "Great model!"
    mock_get_request_message.return_value = UpdateModelVersion(
        name=name, version=version, description=description
    )

    mv = ModelVersion(name=name, version=version, creation_timestamp=123, description=description)
    mock_model_registry_store.update_model_version.return_value = mv
    _update_model_version()
    _, args = mock_model_registry_store.update_model_version.call_args
    assert args == {"name": name, "version": version, "description": description}


def test_transition_model_version_stage(mock_get_request_message, mock_model_registry_store):
    name = "model1"
    version = "32"
    stage = "Production"
    mock_get_request_message.return_value = TransitionModelVersionStage(
        name=name, version=version, stage=stage
    )
    mv = ModelVersion(name=name, version=version, creation_timestamp=123, current_stage=stage)
    mock_model_registry_store.transition_model_version_stage.return_value = mv
    _transition_stage()
    _, args = mock_model_registry_store.transition_model_version_stage.call_args
    assert args == {
        "name": name,
        "version": version,
        "stage": stage,
        "archive_existing_versions": False,
    }


def test_delete_model_version(mock_get_request_message, mock_model_registry_store):
    name = "model1"
    version = "32"
    mock_get_request_message.return_value = DeleteModelVersion(name=name, version=version)
    _delete_model_version()
    _, args = mock_model_registry_store.delete_model_version.call_args
    assert args == {"name": name, "version": version}


def test_get_model_version_download_uri(mock_get_request_message, mock_model_registry_store):
    name = "model1"
    version = "32"
    mock_get_request_message.return_value = GetModelVersionDownloadUri(name=name, version=version)
    mock_model_registry_store.get_model_version_download_uri.return_value = "some/download/path"
    resp = _get_model_version_download_uri()
    _, args = mock_model_registry_store.get_model_version_download_uri.call_args
    assert args == {"name": name, "version": version}
    assert json.loads(resp.get_data()) == {"artifact_uri": "some/download/path"}


def test_search_model_versions(mock_get_request_message, mock_model_registry_store):
    mvds = [
        ModelVersion(
            name="model_1",
            version="5",
            creation_timestamp=100,
            last_updated_timestamp=3200,
            description="v 5",
            user_id="u1",
            current_stage="Production",
            source="A/B/CD",
            run_id=uuid.uuid4().hex,
            status="READY",
            status_message=None,
        ),
        ModelVersion(
            name="model_1",
            version="12",
            creation_timestamp=110,
            last_updated_timestamp=2000,
            description="v 12",
            user_id="u2",
            current_stage="Production",
            source="A/B/CD",
            run_id=uuid.uuid4().hex,
            status="READY",
            status_message=None,
        ),
        ModelVersion(
            name="ads_model",
            version="8",
            creation_timestamp=200,
            last_updated_timestamp=1000,
            description="v 8",
            user_id="u1",
            current_stage="Staging",
            source="A/B/CD",
            run_id=uuid.uuid4().hex,
            status="READY",
            status_message=None,
        ),
        ModelVersion(
            name="fraud_detection_model",
            version="345",
            creation_timestamp=1000,
            last_updated_timestamp=999,
            description="newest version",
            user_id="u12",
            current_stage="None",
            source="A/B/CD",
            run_id=uuid.uuid4().hex,
            status="READY",
            status_message=None,
        ),
    ]
    mock_get_request_message.return_value = SearchModelVersions(filter="source_path = 'A/B/CD'")
    mock_model_registry_store.search_model_versions.return_value = PagedList(mvds, None)
    resp = _search_model_versions()
    mock_model_registry_store.search_model_versions.assert_called_with(
        filter_string="source_path = 'A/B/CD'",
        max_results=SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD,
        order_by=[],
        page_token="",
    )
    assert json.loads(resp.get_data()) == {"model_versions": jsonify(mvds)}

    mock_get_request_message.return_value = SearchModelVersions(filter="name='model_1'")
    mock_model_registry_store.search_model_versions.return_value = PagedList(mvds[:1], "tok")
    resp = _search_model_versions()
    mock_model_registry_store.search_model_versions.assert_called_with(
        filter_string="name='model_1'",
        max_results=SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD,
        order_by=[],
        page_token="",
    )
    assert json.loads(resp.get_data()) == {
        "model_versions": jsonify(mvds[:1]),
        "next_page_token": "tok",
    }

    mock_get_request_message.return_value = SearchModelVersions(filter="version<=12", max_results=2)
    mock_model_registry_store.search_model_versions.return_value = PagedList(
        [mvds[0], mvds[2]], "next"
    )
    resp = _search_model_versions()
    mock_model_registry_store.search_model_versions.assert_called_with(
        filter_string="version<=12", max_results=2, order_by=[], page_token=""
    )
    assert json.loads(resp.get_data()) == {
        "model_versions": jsonify([mvds[0], mvds[2]]),
        "next_page_token": "next",
    }

    mock_get_request_message.return_value = SearchModelVersions(
        filter="version<=12", max_results=2, order_by=["version DESC"], page_token="prev"
    )
    mock_model_registry_store.search_model_versions.return_value = PagedList(mvds[1:3], "next")
    resp = _search_model_versions()
    mock_model_registry_store.search_model_versions.assert_called_with(
        filter_string="version<=12", max_results=2, order_by=["version DESC"], page_token="prev"
    )
    assert json.loads(resp.get_data()) == {
        "model_versions": jsonify(mvds[1:3]),
        "next_page_token": "next",
    }


def test_set_model_version_tag(mock_get_request_message, mock_model_registry_store):
    name = "model1"
    version = "1"
    tag = ModelVersionTag(key="some weird key", value="some value")
    mock_get_request_message.return_value = SetModelVersionTag(
        name=name, version=version, key=tag.key, value=tag.value
    )
    _set_model_version_tag()
    _, args = mock_model_registry_store.set_model_version_tag.call_args
    assert args == {"name": name, "version": version, "tag": tag}


def test_delete_model_version_tag(mock_get_request_message, mock_model_registry_store):
    name = "model1"
    version = "1"
    key = "some weird key"
    mock_get_request_message.return_value = DeleteModelVersionTag(
        name=name, version=version, key=key
    )
    _delete_model_version_tag()
    _, args = mock_model_registry_store.delete_model_version_tag.call_args
    assert args == {"name": name, "version": version, "key": key}


def test_set_registered_model_alias(mock_get_request_message, mock_model_registry_store):
    name = "model1"
    alias = "test_alias"
    version = "1"
    mock_get_request_message.return_value = SetRegisteredModelAlias(
        name=name, alias=alias, version=version
    )
    _set_registered_model_alias()
    _, args = mock_model_registry_store.set_registered_model_alias.call_args
    assert args == {"name": name, "alias": alias, "version": version}


def test_delete_registered_model_alias(mock_get_request_message, mock_model_registry_store):
    name = "model1"
    alias = "test_alias"
    mock_get_request_message.return_value = DeleteRegisteredModelAlias(name=name, alias=alias)
    _delete_registered_model_alias()
    _, args = mock_model_registry_store.delete_registered_model_alias.call_args
    assert args == {"name": name, "alias": alias}


def test_get_model_version_by_alias(mock_get_request_message, mock_model_registry_store):
    name = "model1"
    alias = "test_alias"
    mock_get_request_message.return_value = GetModelVersionByAlias(name=name, alias=alias)
    mvd = ModelVersion(
        name="model1",
        version="5",
        creation_timestamp=1,
        last_updated_timestamp=12,
        description="v 5",
        user_id="u1",
        current_stage="Production",
        source="A/B",
        run_id=uuid.uuid4().hex,
        status="READY",
        status_message=None,
        aliases=["test_alias"],
    )
    mock_model_registry_store.get_model_version_by_alias.return_value = mvd
    resp = _get_model_version_by_alias()
    _, args = mock_model_registry_store.get_model_version_by_alias.call_args
    assert args == {"name": name, "alias": alias}
    assert json.loads(resp.get_data()) == {"model_version": jsonify(mvd)}


@pytest.mark.parametrize(
    "path",
    [
        "/path",
        "path/../to/file",
        "/etc/passwd",
        "/etc/passwd%00.jpg",
        "/file://etc/passwd",
        "%2E%2E%2F%2E%2E%2Fpath",
    ],
)
def test_delete_artifact_mlflow_artifacts_throws_for_malicious_path(enable_serve_artifacts, path):
    response = _delete_artifact_mlflow_artifacts(path)
    assert response.status_code == 400
    json_response = json.loads(response.get_data())
    assert json_response["error_code"] == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert json_response["message"] == "Invalid path"


@pytest.mark.parametrize(
    "uri",
    [
        "http://host#/abc/etc/",
        "http://host/;..%2F..%2Fetc",
    ],
)
def test_local_file_read_write_by_pass_vulnerability(uri):
    request = mock.MagicMock()
    request.method = "POST"
    request.content_type = "application/json; charset=utf-8"
    request.get_json = mock.MagicMock()
    request.get_json.return_value = {
        "name": "hello",
        "artifact_location": uri,
    }
    msg = _get_request_message(CreateExperiment(), flask_request=request)
    with mock.patch("mlflow.server.handlers._get_request_message", return_value=msg):
        response = _create_experiment()
        json_response = json.loads(response.get_data())
        assert (
            json_response["message"] == "'artifact_location' URL can't include fragments or params."
        )

    # Test if source is a local filesystem path, `_validate_source` validates that the run
    # artifact_uri is also a local filesystem path.
    run_id = uuid.uuid4().hex
    with mock.patch("mlflow.server.handlers._get_tracking_store") as mock_get_tracking_store:
        mock_get_tracking_store().get_run(
            run_id
        ).info.artifact_uri = f"http://host/{run_id}/artifacts/abc"

        with pytest.raises(
            MlflowException,
            match=(
                "the run_id request parameter has to be specified and the local "
                "path has to be contained within the artifact directory of the "
                "run specified by the run_id"
            ),
        ):
            _validate_source("/local/path/xyz", run_id)


@pytest.mark.parametrize(
    ("location", "expected_class", "expected_uri"),
    [
        ("file:///0/traces/123", LocalArtifactRepository, "file:///0/traces/123"),
        ("s3://bucket/0/traces/123", S3ArtifactRepository, "s3://bucket/0/traces/123"),
        (
            "wasbs://container@account.blob.core.windows.net/bucket/1/traces/123",
            AzureBlobArtifactRepository,
            "wasbs://container@account.blob.core.windows.net/bucket/1/traces/123",
        ),
        # Proxy URI must be resolved to the actual storage URI
        (
            "https://127.0.0.1/api/2.0/mlflow-artifacts/artifacts/2/traces/123",
            S3ArtifactRepository,
            "s3://bucket/2/traces/123",
        ),
        ("mlflow-artifacts:/1/traces/123", S3ArtifactRepository, "s3://bucket/1/traces/123"),
    ],
)
def test_get_trace_artifact_repo(location, expected_class, expected_uri, monkeypatch):
    monkeypatch.setenv(SERVE_ARTIFACTS_ENV_VAR, "true")
    monkeypatch.setenv(ARTIFACTS_DESTINATION_ENV_VAR, "s3://bucket")
    trace_info = TraceInfo("123", "0", 0, 1, "OK", tags={MLFLOW_ARTIFACT_LOCATION: location})
    repo = _get_trace_artifact_repo(trace_info)
    assert isinstance(repo, expected_class)
    assert repo.artifact_uri == expected_uri
