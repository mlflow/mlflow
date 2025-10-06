import json
import uuid
from unittest import mock

import pytest

import mlflow
from mlflow.entities import ScorerVersion, TraceInfo, TraceState, ViewType
from mlflow.entities.model_registry import (
    ModelVersion,
    ModelVersionTag,
    RegisteredModel,
    RegisteredModelTag,
)
from mlflow.entities.model_registry.prompt_version import IS_PROMPT_TAG_KEY, PROMPT_TEXT_TAG_KEY
from mlflow.entities.trace_location import TraceLocation as EntityTraceLocation
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
from mlflow.protos.service_pb2 import (
    CalculateTraceFilterCorrelation,
    CreateExperiment,
    DeleteScorer,
    GetScorer,
    ListScorers,
    ListScorerVersions,
    RegisterScorer,
    SearchExperiments,
    SearchLoggedModels,
    SearchRuns,
    SearchTraces,
    SearchTracesV3,
    TraceLocation,
)
from mlflow.protos.webhooks_pb2 import ListWebhooks
from mlflow.server import (
    ARTIFACTS_DESTINATION_ENV_VAR,
    BACKEND_STORE_URI_ENV_VAR,
    SERVE_ARTIFACTS_ENV_VAR,
    app,
)
from mlflow.server.handlers import (
    ModelRegistryStoreRegistryWrapper,
    TrackingStoreRegistryWrapper,
    _calculate_trace_filter_correlation,
    _convert_path_parameter_to_flask_format,
    _create_dataset_handler,
    _create_experiment,
    _create_model_version,
    _create_registered_model,
    _delete_artifact_mlflow_artifacts,
    _delete_dataset_handler,
    _delete_dataset_tag_handler,
    _delete_model_version,
    _delete_model_version_tag,
    _delete_registered_model,
    _delete_registered_model_alias,
    _delete_registered_model_tag,
    _delete_scorer,
    _deprecated_search_traces_v2,
    _get_dataset_experiment_ids_handler,
    _get_dataset_handler,
    _get_dataset_records_handler,
    _get_latest_versions,
    _get_model_version,
    _get_model_version_by_alias,
    _get_model_version_download_uri,
    _get_registered_model,
    _get_request_message,
    _get_scorer,
    _get_trace_artifact_repo,
    _get_trace_info_v4,
    _list_scorer_versions,
    _list_scorers,
    _list_webhooks,
    _log_batch,
    _register_scorer,
    _rename_registered_model,
    _search_evaluation_datasets_handler,
    _search_experiments,
    _search_logged_models,
    _search_model_versions,
    _search_registered_models,
    _search_runs,
    _search_traces_v3,
    _set_dataset_tags_handler,
    _set_model_version_tag,
    _set_registered_model_alias,
    _set_registered_model_tag,
    _transition_stage,
    _update_model_version,
    _update_registered_model,
    _upsert_dataset_records_handler,
    _validate_source_run,
    catch_mlflow_exception,
    get_endpoints,
)
from mlflow.store._unity_catalog.registry.rest_store import UcModelRegistryStore
from mlflow.store.artifact.azure_blob_artifact_repo import AzureBlobArtifactRepository
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry import (
    SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD,
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
)
from mlflow.store.model_registry.rest_store import RestStore as ModelRegistryRestStore
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracing.analysis import TraceFilterCorrelationResult
from mlflow.tracing.constant import TRACE_ID_V4_PREFIX
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
        mock_store.list_webhooks_by_event.return_value = PagedList([], None)
        m.return_value = mock_store
        yield mock_store


@pytest.fixture
def enable_serve_artifacts(monkeypatch):
    monkeypatch.setenv(SERVE_ARTIFACTS_ENV_VAR, "true")


@pytest.fixture
def mock_evaluation_dataset():
    from mlflow.protos.datasets_pb2 import Dataset as ProtoDataset

    dataset = mock.MagicMock()
    dataset.dataset_id = "d-1234567890abcdef1234567890abcdef"
    dataset.name = "test_dataset"
    dataset.digest = "abc123"
    dataset.created_time = 1234567890
    dataset.last_update_time = 1234567890
    dataset.created_by = "test_user"
    dataset.last_updated_by = "test_user"
    dataset.tags = {"env": "test", "version": "1.0"}
    dataset.experiment_ids = ["0", "1"]
    dataset.records = []
    dataset.schema = json.dumps(
        {"inputs": {"question": "string"}, "expectations": {"accuracy": "float"}}
    )
    dataset.profile = json.dumps({"record_count": 0})

    proto_dataset = ProtoDataset()
    proto_dataset.dataset_id = dataset.dataset_id
    proto_dataset.name = dataset.name
    proto_dataset.digest = dataset.digest
    proto_dataset.created_time = dataset.created_time
    proto_dataset.last_update_time = dataset.last_update_time
    proto_dataset.created_by = dataset.created_by
    proto_dataset.last_updated_by = dataset.last_updated_by
    proto_dataset.schema = dataset.schema
    proto_dataset.profile = dataset.profile

    dataset.to_proto = mock.MagicMock(return_value=proto_dataset)

    return dataset


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
    _, kwargs = mock_tracking_store.search_runs.call_args
    assert kwargs["run_view_type"] == ViewType.ACTIVE_ONLY


def test_search_runs_empty_page_token(mock_get_request_message, mock_tracking_store):
    """
    Test that empty page_token from protobuf is converted to None before calling store
    """
    # Create proto without setting page_token
    search_runs_proto = SearchRuns()
    search_runs_proto.experiment_ids.append("0")
    search_runs_proto.max_results = 10
    # Verify protobuf returns empty string for unset field
    assert search_runs_proto.page_token == ""

    mock_get_request_message.return_value = search_runs_proto
    mock_tracking_store.search_runs.return_value = PagedList([], None)

    _search_runs()

    # Verify store was called with None, not empty string
    mock_tracking_store.search_runs.assert_called_once()
    call_kwargs = mock_tracking_store.search_runs.call_args.kwargs
    assert call_kwargs["page_token"] is None  # page_token should be None, not ""


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
        "page_token": None,
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
        "page_token": None,
    }
    assert json.loads(resp.get_data()) == {
        "registered_models": jsonify(rmds[:1]),
        "next_page_token": "tok",
    }

    mock_get_request_message.return_value = SearchRegisteredModels(filter="hi", max_results=5)
    mock_model_registry_store.search_registered_models.return_value = PagedList([rmds[0]], "tik")
    resp = _search_registered_models()
    _, args = mock_model_registry_store.search_registered_models.call_args
    assert args == {"filter_string": "hi", "max_results": 5, "order_by": [], "page_token": None}
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
        page_token=None,
    )
    assert json.loads(resp.get_data()) == {"model_versions": jsonify(mvds)}

    mock_get_request_message.return_value = SearchModelVersions(filter="name='model_1'")
    mock_model_registry_store.search_model_versions.return_value = PagedList(mvds[:1], "tok")
    resp = _search_model_versions()
    mock_model_registry_store.search_model_versions.assert_called_with(
        filter_string="name='model_1'",
        max_results=SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD,
        order_by=[],
        page_token=None,
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
        filter_string="version<=12", max_results=2, order_by=[], page_token=None
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
            _validate_source_run("/local/path/xyz", run_id)


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
    trace_info = TraceInfo(
        trace_id="123",
        trace_location=EntityTraceLocation.from_experiment_id("0"),
        request_time=0,
        execution_duration=1,
        state=TraceState.OK,
        tags={MLFLOW_ARTIFACT_LOCATION: location},
    )
    repo = _get_trace_artifact_repo(trace_info)
    assert isinstance(repo, expected_class)
    assert repo.artifact_uri == expected_uri


### Prompt Registry Tests ###
def test_create_prompt_as_registered_model(mock_get_request_message, mock_model_registry_store):
    tags = [RegisteredModelTag(key=IS_PROMPT_TAG_KEY, value="true")]
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


def test_create_prompt_as_model_version(mock_get_request_message, mock_model_registry_store):
    tags = [
        ModelVersionTag(key=IS_PROMPT_TAG_KEY, value="true"),
        ModelVersionTag(key=PROMPT_TEXT_TAG_KEY, value="some prompt text"),
    ]
    mock_get_request_message.return_value = CreateModelVersion(
        name="model_1",
        tags=[tag.to_proto() for tag in tags],
        source=None,
        run_id=None,
        run_link=None,
    )
    mv = ModelVersion(
        name="prompt_1", version="12", creation_timestamp=123, tags=tags, run_link=None
    )
    mock_model_registry_store.create_model_version.return_value = mv
    resp = _create_model_version()
    _, args = mock_model_registry_store.create_model_version.call_args
    assert args["name"] == "model_1"
    assert args["source"] == ""
    assert args["run_id"] == ""
    assert {tag.key: tag.value for tag in args["tags"]} == {tag.key: tag.value for tag in tags}
    assert args["run_link"] == ""
    assert json.loads(resp.get_data()) == {"model_version": jsonify(mv)}


def test_create_evaluation_dataset(mock_tracking_store, mock_evaluation_dataset):
    mock_tracking_store.create_dataset.return_value = mock_evaluation_dataset

    with app.test_request_context(
        method="POST",
        json={
            "name": "test_dataset",
            "experiment_ids": ["0", "1"],
            "tags": json.dumps({"env": "test"}),
        },
    ):
        _create_dataset_handler()

    mock_tracking_store.create_dataset.assert_called_once_with(
        name="test_dataset",
        experiment_ids=["0", "1"],
        tags={"env": "test"},
    )


def test_get_evaluation_dataset(mock_tracking_store, mock_evaluation_dataset):
    mock_tracking_store.get_dataset.return_value = mock_evaluation_dataset

    dataset_id = "d-1234567890abcdef1234567890abcdef"
    with app.test_request_context(method="GET"):
        _get_dataset_handler(dataset_id)

    mock_tracking_store.get_dataset.assert_called_once_with(dataset_id)


def test_delete_evaluation_dataset(mock_tracking_store):
    dataset_id = "d-1234567890abcdef1234567890abcdef"
    with app.test_request_context(method="DELETE"):
        _delete_dataset_handler(dataset_id)

    mock_tracking_store.delete_dataset.assert_called_once_with(dataset_id)


def test_search_datasets(mock_tracking_store):
    from mlflow.protos.datasets_pb2 import Dataset as ProtoDataset

    datasets = []
    for i in range(2):
        ds = mock.MagicMock()
        ds.name = f"dataset_{i}"
        proto = ProtoDataset()
        proto.dataset_id = f"d-{i:032d}"
        proto.name = ds.name
        ds.to_proto.return_value = proto
        datasets.append(ds)

    paged_list = PagedList(datasets, "next_token")
    mock_tracking_store.search_datasets.return_value = paged_list

    with app.test_request_context(
        method="POST",
        json={
            "experiment_ids": ["0", "1"],
            "filter_string": "name = 'dataset_1'",
            "max_results": 10,
            "order_by": ["name DESC"],
            "page_token": "token123",
        },
    ):
        _search_evaluation_datasets_handler()

    mock_tracking_store.search_datasets.assert_called_once_with(
        experiment_ids=["0", "1"],
        filter_string="name = 'dataset_1'",
        max_results=10,
        order_by=["name DESC"],
        page_token="token123",
    )


def test_set_dataset_tags(mock_tracking_store):
    dataset_id = "d-1234567890abcdef1234567890abcdef"
    with app.test_request_context(
        method="POST",
        json={
            "tags": json.dumps({"env": "production", "version": "2.0"}),
        },
    ):
        _set_dataset_tags_handler(dataset_id)

    mock_tracking_store.set_dataset_tags.assert_called_once_with(
        dataset_id=dataset_id,
        tags={"env": "production", "version": "2.0"},
    )


def test_delete_dataset_tag(mock_tracking_store):
    dataset_id = "d-1234567890abcdef1234567890abcdef"
    key = "deprecated_tag"
    with app.test_request_context(method="DELETE"):
        _delete_dataset_tag_handler(dataset_id, key)

    mock_tracking_store.delete_dataset_tag.assert_called_once_with(
        dataset_id=dataset_id,
        key=key,
    )


def test_upsert_dataset_records(mock_tracking_store):
    mock_tracking_store.upsert_dataset_records.return_value = {
        "inserted": 2,
        "updated": 0,
    }

    dataset_id = "d-1234567890abcdef1234567890abcdef"
    records = [
        {"inputs": {"q": "test1"}, "expectations": {"score": 0.9}},
        {"inputs": {"q": "test2"}, "expectations": {"score": 0.8}},
    ]

    with app.test_request_context(
        method="POST",
        json={
            "records": json.dumps(records),
        },
    ):
        resp = _upsert_dataset_records_handler(dataset_id)

    mock_tracking_store.upsert_dataset_records.assert_called_once_with(
        dataset_id=dataset_id,
        records=records,
    )

    response_data = json.loads(resp.get_data())
    assert response_data["inserted_count"] == 2
    assert response_data["updated_count"] == 0


def test_get_dataset_experiment_ids(mock_tracking_store):
    mock_tracking_store.get_dataset_experiment_ids.return_value = [
        "exp1",
        "exp2",
        "exp3",
    ]

    dataset_id = "d-1234567890abcdef1234567890abcdef"
    with app.test_request_context(method="GET"):
        resp = _get_dataset_experiment_ids_handler(dataset_id)

    mock_tracking_store.get_dataset_experiment_ids.assert_called_once_with(dataset_id=dataset_id)

    response_data = json.loads(resp.get_data())
    assert response_data["experiment_ids"] == ["exp1", "exp2", "exp3"]


def test_get_dataset_records(mock_tracking_store):
    records = []
    for i in range(3):
        record = mock.MagicMock()
        record.dataset_id = "d-1234567890abcdef1234567890abcdef"
        record.dataset_record_id = f"r-00{i}"
        record.inputs = {"question": f"test{i}"}
        record.expectations = {"score": 0.9 - i * 0.1}
        record.tags = {}
        record.created_time = 1234567890 + i
        record.last_update_time = 1234567890 + i
        record.to_dict.return_value = {
            "dataset_id": record.dataset_id,
            "dataset_record_id": record.dataset_record_id,
            "inputs": record.inputs,
            "expectations": record.expectations,
            "tags": record.tags,
            "created_time": record.created_time,
            "last_update_time": record.last_update_time,
        }
        records.append(record)

    mock_tracking_store._load_dataset_records.return_value = (records, None)

    dataset_id = "d-1234567890abcdef1234567890abcdef"
    with app.test_request_context(method="GET"):
        resp = _get_dataset_records_handler(dataset_id)

    mock_tracking_store._load_dataset_records.assert_called_with(
        dataset_id, max_results=1000, page_token=None
    )

    response_data = json.loads(resp.get_data())
    records_data = json.loads(response_data["records"])
    assert len(records_data) == 3
    assert records_data[0]["dataset_record_id"] == "r-000"

    mock_tracking_store._load_dataset_records.return_value = (records[:2], "token_page2")

    with app.test_request_context(
        method="GET",
        json={
            "max_results": 2,
            "page_token": None,
        },
    ):
        resp = _get_dataset_records_handler(dataset_id)

    mock_tracking_store._load_dataset_records.assert_called_with(
        dataset_id, max_results=2, page_token=None
    )

    response_data = json.loads(resp.get_data())
    records_data = json.loads(response_data["records"])
    assert len(records_data) == 2
    assert response_data["next_page_token"] == "token_page2"

    mock_tracking_store._load_dataset_records.return_value = (records[2:], None)

    with app.test_request_context(
        method="GET",
        json={
            "max_results": 2,
            "page_token": "token_page2",
        },
    ):
        resp = _get_dataset_records_handler(dataset_id)

    mock_tracking_store._load_dataset_records.assert_called_with(
        dataset_id, max_results=2, page_token="token_page2"
    )

    response_data = json.loads(resp.get_data())
    records_data = json.loads(response_data["records"])
    assert len(records_data) == 1
    assert "next_page_token" not in response_data or response_data["next_page_token"] == ""


def test_get_dataset_records_empty(mock_tracking_store):
    mock_tracking_store._load_dataset_records.return_value = ([], None)

    dataset_id = "d-1234567890abcdef1234567890abcdef"
    with app.test_request_context(method="GET"):
        resp = _get_dataset_records_handler(dataset_id)

    response_data = json.loads(resp.get_data())
    records_data = json.loads(response_data["records"])
    assert len(records_data) == 0
    assert "next_page_token" not in response_data or response_data["next_page_token"] == ""


def test_get_dataset_records_pagination(mock_tracking_store):
    dataset_id = "d-1234567890abcdef1234567890abcdef"
    all_records = []
    for i in range(50):
        record = mock.Mock()
        record.dataset_record_id = f"r-{i:03d}"
        record.inputs = {"q": f"Question {i}"}
        record.expectations = {"a": f"Answer {i}"}
        record.tags = {}
        record.source_type = "TRACE"
        record.source_id = f"trace-{i}"
        record.created_time = 1609459200 + i
        record.to_dict.return_value = {
            "dataset_record_id": f"r-{i:03d}",
            "inputs": {"q": f"Question {i}"},
            "expectations": {"a": f"Answer {i}"},
            "tags": {},
            "source_type": "TRACE",
            "source_id": f"trace-{i}",
            "created_time": 1609459200 + i,
        }
        all_records.append(record)
    mock_tracking_store._load_dataset_records.return_value = (all_records[:20], "token_20")

    with app.test_request_context(
        method="GET",
        json={"max_results": 20},
    ):
        resp = _get_dataset_records_handler(dataset_id)

    mock_tracking_store._load_dataset_records.assert_called_with(
        dataset_id, max_results=20, page_token=None
    )

    response_data = json.loads(resp.get_data())
    records_data = json.loads(response_data["records"])
    assert len(records_data) == 20
    assert response_data["next_page_token"] == "token_20"
    assert records_data[0]["dataset_record_id"] == "r-000"
    assert records_data[19]["dataset_record_id"] == "r-019"
    mock_tracking_store._load_dataset_records.return_value = (all_records[20:40], "token_40")

    with app.test_request_context(
        method="GET",
        json={"max_results": 20, "page_token": "token_20"},
    ):
        resp = _get_dataset_records_handler(dataset_id)

    mock_tracking_store._load_dataset_records.assert_called_with(
        dataset_id, max_results=20, page_token="token_20"
    )

    response_data = json.loads(resp.get_data())
    records_data = json.loads(response_data["records"])
    assert len(records_data) == 20
    assert response_data["next_page_token"] == "token_40"
    assert records_data[0]["dataset_record_id"] == "r-020"
    mock_tracking_store._load_dataset_records.return_value = (all_records[40:], None)

    with app.test_request_context(
        method="GET",
        json={"max_results": 20, "page_token": "token_40"},
    ):
        resp = _get_dataset_records_handler(dataset_id)

    response_data = json.loads(resp.get_data())
    records_data = json.loads(response_data["records"])
    assert len(records_data) == 10
    assert "next_page_token" not in response_data or response_data["next_page_token"] == ""
    assert records_data[0]["dataset_record_id"] == "r-040"
    assert records_data[9]["dataset_record_id"] == "r-049"


def test_register_scorer(mock_get_request_message, mock_tracking_store):
    """Test register_scorer handler."""
    experiment_id = "123"
    name = "accuracy_scorer"
    serialized_scorer = "serialized_scorer_data"

    mock_get_request_message.return_value = RegisterScorer(
        experiment_id=experiment_id, name=name, serialized_scorer=serialized_scorer
    )

    mock_tracking_store.register_scorer.return_value = 1

    resp = _register_scorer()

    # Verify the tracking store was called with correct arguments
    mock_tracking_store.register_scorer.assert_called_once_with(
        experiment_id, name, serialized_scorer
    )

    # Verify the response
    response_data = json.loads(resp.get_data())
    assert response_data == {"version": 1}


def test_list_scorers(mock_get_request_message, mock_tracking_store):
    """Test list_scorers handler."""
    experiment_id = "123"

    mock_get_request_message.return_value = ListScorers(experiment_id=experiment_id)

    # Create mock scorers
    scorers = [
        ScorerVersion(
            experiment_id=123,
            scorer_name="accuracy_scorer",
            scorer_version=1,
            serialized_scorer="serialized_accuracy_scorer",
            creation_time=12345,
        ),
        ScorerVersion(
            experiment_id=123,
            scorer_name="safety_scorer",
            scorer_version=2,
            serialized_scorer="serialized_safety_scorer",
            creation_time=12345,
        ),
    ]

    mock_tracking_store.list_scorers.return_value = scorers

    resp = _list_scorers()

    # Verify the tracking store was called with correct arguments
    mock_tracking_store.list_scorers.assert_called_once_with(experiment_id)

    # Verify the response
    response_data = json.loads(resp.get_data())
    assert len(response_data["scorers"]) == 2
    assert response_data["scorers"][0]["scorer_name"] == "accuracy_scorer"
    assert response_data["scorers"][0]["scorer_version"] == 1
    assert response_data["scorers"][0]["serialized_scorer"] == "serialized_accuracy_scorer"
    assert response_data["scorers"][1]["scorer_name"] == "safety_scorer"
    assert response_data["scorers"][1]["scorer_version"] == 2
    assert response_data["scorers"][1]["serialized_scorer"] == "serialized_safety_scorer"


def test_list_scorer_versions(mock_get_request_message, mock_tracking_store):
    """Test list_scorer_versions handler."""
    experiment_id = "123"
    name = "accuracy_scorer"

    mock_get_request_message.return_value = ListScorerVersions(
        experiment_id=experiment_id, name=name
    )

    # Create mock scorers with multiple versions
    scorers = [
        ScorerVersion(
            experiment_id=123,
            scorer_name="accuracy_scorer",
            scorer_version=1,
            serialized_scorer="serialized_accuracy_scorer_v1",
            creation_time=12345,
        ),
        ScorerVersion(
            experiment_id=123,
            scorer_name="accuracy_scorer",
            scorer_version=2,
            serialized_scorer="serialized_accuracy_scorer_v2",
            creation_time=12345,
        ),
    ]

    mock_tracking_store.list_scorer_versions.return_value = scorers

    resp = _list_scorer_versions()

    # Verify the tracking store was called with correct arguments
    mock_tracking_store.list_scorer_versions.assert_called_once_with(experiment_id, name)

    # Verify the response
    response_data = json.loads(resp.get_data())
    assert len(response_data["scorers"]) == 2
    assert response_data["scorers"][0]["scorer_version"] == 1
    assert response_data["scorers"][0]["serialized_scorer"] == "serialized_accuracy_scorer_v1"
    assert response_data["scorers"][1]["scorer_version"] == 2
    assert response_data["scorers"][1]["serialized_scorer"] == "serialized_accuracy_scorer_v2"


def test_get_scorer_with_version(mock_get_request_message, mock_tracking_store):
    """Test get_scorer handler with specific version."""
    experiment_id = "123"
    name = "accuracy_scorer"
    version = 2

    mock_get_request_message.return_value = GetScorer(
        experiment_id=experiment_id, name=name, version=version
    )

    # Mock the return value as a ScorerVersion entity
    mock_scorer_version = ScorerVersion(
        experiment_id=123,
        scorer_name="accuracy_scorer",
        scorer_version=2,
        serialized_scorer="serialized_accuracy_scorer_v2",
        creation_time=1640995200000,
    )
    mock_tracking_store.get_scorer.return_value = mock_scorer_version

    resp = _get_scorer()

    # Verify the tracking store was called with correct arguments (positional)
    mock_tracking_store.get_scorer.assert_called_once_with(experiment_id, name, version)

    # Verify the response
    response_data = json.loads(resp.get_data())
    assert response_data["scorer"]["experiment_id"] == 123
    assert response_data["scorer"]["scorer_name"] == "accuracy_scorer"
    assert response_data["scorer"]["scorer_version"] == 2
    assert response_data["scorer"]["serialized_scorer"] == "serialized_accuracy_scorer_v2"
    assert response_data["scorer"]["creation_time"] == 1640995200000


def test_get_scorer_without_version(mock_get_request_message, mock_tracking_store):
    """Test get_scorer handler without version (should return latest)."""
    experiment_id = "123"
    name = "accuracy_scorer"

    mock_get_request_message.return_value = GetScorer(experiment_id=experiment_id, name=name)

    # Mock the return value as a ScorerVersion entity
    mock_scorer_version = ScorerVersion(
        experiment_id=123,
        scorer_name="accuracy_scorer",
        scorer_version=3,
        serialized_scorer="serialized_accuracy_scorer_latest",
        creation_time=1640995200000,
    )
    mock_tracking_store.get_scorer.return_value = mock_scorer_version

    resp = _get_scorer()

    # Verify the tracking store was called with correct arguments (positional, version=None)
    mock_tracking_store.get_scorer.assert_called_once_with(experiment_id, name, None)

    # Verify the response
    response_data = json.loads(resp.get_data())
    assert response_data["scorer"]["experiment_id"] == 123
    assert response_data["scorer"]["scorer_name"] == "accuracy_scorer"
    assert response_data["scorer"]["scorer_version"] == 3
    assert response_data["scorer"]["serialized_scorer"] == "serialized_accuracy_scorer_latest"
    assert response_data["scorer"]["creation_time"] == 1640995200000


def test_delete_scorer_with_version(mock_get_request_message, mock_tracking_store):
    """Test delete_scorer handler with specific version."""
    experiment_id = "123"
    name = "accuracy_scorer"
    version = 2

    mock_get_request_message.return_value = DeleteScorer(
        experiment_id=experiment_id, name=name, version=version
    )

    resp = _delete_scorer()

    # Verify the tracking store was called with correct arguments (positional)
    mock_tracking_store.delete_scorer.assert_called_once_with(experiment_id, name, version)

    # Verify the response (should be empty for delete operations)
    response_data = json.loads(resp.get_data())
    assert response_data == {}


def test_delete_scorer_without_version(mock_get_request_message, mock_tracking_store):
    """Test delete_scorer handler without version (should delete all versions)."""
    experiment_id = "123"
    name = "accuracy_scorer"

    mock_get_request_message.return_value = DeleteScorer(experiment_id=experiment_id, name=name)

    resp = _delete_scorer()

    # Verify the tracking store was called with correct arguments (positional, version=None)
    mock_tracking_store.delete_scorer.assert_called_once_with(experiment_id, name, None)

    # Verify the response (should be empty for delete operations)
    response_data = json.loads(resp.get_data())
    assert response_data == {}


def test_calculate_trace_filter_correlation(mock_get_request_message, mock_tracking_store):
    experiment_ids = ["123", "456"]
    filter_string1 = "span.type = 'LLM'"
    filter_string2 = "feedback.quality > 0.8"
    base_filter = "request_time > 1000"

    mock_request = CalculateTraceFilterCorrelation(
        experiment_ids=experiment_ids,
        filter_string1=filter_string1,
        filter_string2=filter_string2,
        base_filter=base_filter,
    )
    mock_get_request_message.return_value = mock_request

    mock_result = TraceFilterCorrelationResult(
        npmi=0.456,
        npmi_smoothed=0.445,
        filter1_count=100,
        filter2_count=80,
        joint_count=50,
        total_count=200,
    )
    mock_tracking_store.calculate_trace_filter_correlation.return_value = mock_result

    resp = _calculate_trace_filter_correlation()

    mock_tracking_store.calculate_trace_filter_correlation.assert_called_once_with(
        experiment_ids=experiment_ids,
        filter_string1=filter_string1,
        filter_string2=filter_string2,
        base_filter=base_filter,
    )

    response_data = json.loads(resp.get_data())
    assert response_data["npmi"] == 0.456
    assert response_data["npmi_smoothed"] == 0.445
    assert response_data["filter1_count"] == 100
    assert response_data["filter2_count"] == 80
    assert response_data["joint_count"] == 50
    assert response_data["total_count"] == 200


def test_calculate_trace_filter_correlation_without_base_filter(
    mock_get_request_message, mock_tracking_store
):
    experiment_ids = ["123"]
    filter_string1 = "span.type = 'LLM'"
    filter_string2 = "feedback.quality > 0.8"

    mock_request = CalculateTraceFilterCorrelation(
        experiment_ids=experiment_ids,
        filter_string1=filter_string1,
        filter_string2=filter_string2,
    )
    mock_get_request_message.return_value = mock_request

    mock_result = TraceFilterCorrelationResult(
        npmi=0.789,
        npmi_smoothed=0.775,
        filter1_count=50,
        filter2_count=40,
        joint_count=30,
        total_count=100,
    )
    mock_tracking_store.calculate_trace_filter_correlation.return_value = mock_result

    resp = _calculate_trace_filter_correlation()

    mock_tracking_store.calculate_trace_filter_correlation.assert_called_once_with(
        experiment_ids=experiment_ids,
        filter_string1=filter_string1,
        filter_string2=filter_string2,
        base_filter=None,
    )

    response_data = json.loads(resp.get_data())
    assert response_data["npmi"] == 0.789
    assert response_data["npmi_smoothed"] == 0.775
    assert response_data["filter1_count"] == 50
    assert response_data["filter2_count"] == 40
    assert response_data["joint_count"] == 30
    assert response_data["total_count"] == 100


def test_calculate_trace_filter_correlation_with_nan_npmi(
    mock_get_request_message, mock_tracking_store
):
    experiment_ids = ["123"]
    filter_string1 = "span.type = 'LLM'"
    filter_string2 = "feedback.quality > 0.8"

    mock_request = CalculateTraceFilterCorrelation(
        experiment_ids=experiment_ids,
        filter_string1=filter_string1,
        filter_string2=filter_string2,
    )
    mock_get_request_message.return_value = mock_request

    mock_result = TraceFilterCorrelationResult(
        npmi=float("nan"),
        npmi_smoothed=None,
        filter1_count=0,
        filter2_count=0,
        joint_count=0,
        total_count=100,
    )
    mock_tracking_store.calculate_trace_filter_correlation.return_value = mock_result

    resp = _calculate_trace_filter_correlation()

    mock_tracking_store.calculate_trace_filter_correlation.assert_called_once_with(
        experiment_ids=experiment_ids,
        filter_string1=filter_string1,
        filter_string2=filter_string2,
        base_filter=None,
    )

    response_data = json.loads(resp.get_data())
    assert "npmi" not in response_data
    assert "npmi_smoothed" not in response_data
    assert response_data["filter1_count"] == 0
    assert response_data["filter2_count"] == 0
    assert response_data["joint_count"] == 0
    assert response_data["total_count"] == 100


def test_databricks_tracking_store_registration():
    """Test that Databricks tracking store is properly registered."""
    registry = TrackingStoreRegistryWrapper()

    # Test that the correct store type is returned for databricks scheme
    store = registry.get_store("databricks", artifact_uri=None)
    assert isinstance(store, RestStore)

    # Verify that the store was created with the right get_host_creds function
    # The RestStore should have a get_host_creds attribute that is a partial function
    assert hasattr(store, "get_host_creds")
    assert store.get_host_creds.func.__name__ == "get_databricks_host_creds"
    assert store.get_host_creds.args == ("databricks",)


def test_databricks_model_registry_store_registration():
    """Test that Databricks model registry stores are properly registered."""
    registry = ModelRegistryStoreRegistryWrapper()

    # Test that the correct store type is returned for databricks
    store = registry.get_store("databricks")
    assert isinstance(store, ModelRegistryRestStore)

    # Verify that the store was created with the right get_host_creds function
    assert hasattr(store, "get_host_creds")
    assert store.get_host_creds.func.__name__ == "get_databricks_host_creds"
    assert store.get_host_creds.args == ("databricks",)

    # Test that the correct store type is returned for databricks-uc
    uc_store = registry.get_store("databricks-uc")
    assert isinstance(uc_store, UcModelRegistryStore)

    # Verify that the UC store was created with the right get_host_creds function
    # Note: UcModelRegistryStore uses get_databricks_host_creds internally,
    # not get_databricks_uc_host_creds
    assert hasattr(uc_store, "get_host_creds")
    assert uc_store.get_host_creds.func.__name__ == "get_databricks_host_creds"
    assert uc_store.get_host_creds.args == ("databricks-uc",)

    # Also verify it has tracking_uri set
    assert hasattr(uc_store, "tracking_uri")
    # The tracking_uri will be set based on environment/test config
    # In test environment, it may be set to a test sqlite database
    assert uc_store.tracking_uri is not None


def test_search_experiments_empty_page_token(mock_get_request_message, mock_tracking_store):
    """Test that _search_experiments converts empty page_token to None."""
    # Create proto without setting page_token - it defaults to empty string
    search_experiments_proto = SearchExperiments()
    search_experiments_proto.max_results = 10

    # Verify that proto's default page_token is empty string
    assert search_experiments_proto.page_token == ""

    mock_get_request_message.return_value = search_experiments_proto
    mock_tracking_store.search_experiments.return_value = PagedList([], None)

    _search_experiments()

    # Verify that search_experiments was called with page_token=None (not empty string)
    mock_tracking_store.search_experiments.assert_called_once()
    call_kwargs = mock_tracking_store.search_experiments.call_args.kwargs
    assert call_kwargs.get("page_token") is None
    assert call_kwargs.get("max_results") == 10


def test_search_registered_models_empty_page_token(
    mock_get_request_message, mock_model_registry_store
):
    """Test that _search_registered_models converts empty page_token to None."""
    # Create proto without setting page_token - it defaults to empty string
    search_registered_models_proto = SearchRegisteredModels()
    search_registered_models_proto.max_results = 10

    # Verify that proto's default page_token is empty string
    assert search_registered_models_proto.page_token == ""

    mock_get_request_message.return_value = search_registered_models_proto
    mock_model_registry_store.search_registered_models.return_value = PagedList([], None)

    _search_registered_models()

    # Verify that search_registered_models was called with page_token=None (not empty string)
    mock_model_registry_store.search_registered_models.assert_called_once()
    call_kwargs = mock_model_registry_store.search_registered_models.call_args.kwargs
    assert call_kwargs.get("page_token") is None
    assert call_kwargs.get("max_results") == 10


def test_search_model_versions_empty_page_token(
    mock_get_request_message, mock_model_registry_store
):
    """Test that _search_model_versions converts empty page_token to None."""
    # Create proto without setting page_token - it defaults to empty string
    search_model_versions_proto = SearchModelVersions()
    search_model_versions_proto.max_results = 10

    # Verify that proto's default page_token is empty string
    assert search_model_versions_proto.page_token == ""

    mock_get_request_message.return_value = search_model_versions_proto
    mock_model_registry_store.search_model_versions.return_value = PagedList([], None)

    _search_model_versions()

    # Verify that search_model_versions was called with page_token=None (not empty string)
    mock_model_registry_store.search_model_versions.assert_called_once()
    call_kwargs = mock_model_registry_store.search_model_versions.call_args.kwargs
    assert call_kwargs.get("page_token") is None
    assert call_kwargs.get("max_results") == 10


def test_search_traces_v3_empty_page_token(mock_get_request_message, mock_tracking_store):
    """Test that _search_traces_v3 converts empty page_token to None."""
    # Create proto without setting page_token - it defaults to empty string
    # SearchTracesV3 requires locations field
    search_traces_proto = SearchTracesV3()
    location = TraceLocation()
    location.mlflow_experiment.experiment_id = "1"
    search_traces_proto.locations.append(location)
    search_traces_proto.max_results = 10

    # Verify that proto's default page_token is empty string
    assert search_traces_proto.page_token == ""

    mock_get_request_message.return_value = search_traces_proto
    mock_tracking_store.search_traces.return_value = ([], None)

    _search_traces_v3()

    # Verify that search_traces was called with page_token=None (not empty string)
    mock_tracking_store.search_traces.assert_called_once()
    call_kwargs = mock_tracking_store.search_traces.call_args.kwargs
    assert call_kwargs.get("page_token") is None
    assert call_kwargs.get("max_results") == 10


def test_deprecated_search_traces_v2_empty_page_token(
    mock_get_request_message, mock_tracking_store
):
    """Test that _deprecated_search_traces_v2 converts empty page_token to None."""
    # Create proto without setting page_token - it defaults to empty string
    search_traces_proto = SearchTraces()
    search_traces_proto.max_results = 10

    # Verify that proto's default page_token is empty string
    assert search_traces_proto.page_token == ""

    mock_get_request_message.return_value = search_traces_proto
    mock_tracking_store.search_traces.return_value = ([], None)

    _deprecated_search_traces_v2()

    # Verify that search_traces was called with page_token=None (not empty string)
    mock_tracking_store.search_traces.assert_called_once()
    call_kwargs = mock_tracking_store.search_traces.call_args.kwargs
    assert call_kwargs.get("page_token") is None
    assert call_kwargs.get("max_results") == 10


def test_get_trace_info_v4_handler(mock_tracking_store):
    trace_id = "test-trace-123"
    location = "catalog.schema"
    full_v4_trace_id = f"{TRACE_ID_V4_PREFIX}{location}/{trace_id}"

    mock_trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=EntityTraceLocation.from_uc_schema(
            catalog_name="catalog", schema_name="schema"
        ),
        request_time=1234567890,
        execution_duration=5000,
        state=TraceState.OK,
        trace_metadata={"test": "metadata"},
        tags={"test": "tag"},
    )

    mock_tracking_store.get_trace_info.return_value = mock_trace_info

    response = _get_trace_info_v4(location, trace_id)

    mock_tracking_store.get_trace_info.assert_called_once_with(full_v4_trace_id)
    assert response is not None


def test_search_logged_models_empty_page_token(mock_get_request_message, mock_tracking_store):
    """Test that _search_logged_models converts empty page_token to None."""
    # Create proto without setting page_token - it defaults to empty string
    search_logged_models_proto = SearchLoggedModels()
    search_logged_models_proto.max_results = 10

    # Verify that proto's default page_token is empty string
    assert search_logged_models_proto.page_token == ""

    mock_get_request_message.return_value = search_logged_models_proto
    mock_tracking_store.search_logged_models.return_value = PagedList([], None)

    _search_logged_models()

    # Verify that search_logged_models was called with page_token=None (not empty string)
    mock_tracking_store.search_logged_models.assert_called_once()
    call_kwargs = mock_tracking_store.search_logged_models.call_args.kwargs
    assert call_kwargs.get("page_token") is None
    assert call_kwargs.get("max_results") == 10


def test_list_webhooks_empty_page_token(mock_get_request_message, mock_model_registry_store):
    """Test that _list_webhooks converts empty page_token to None."""
    # Create proto without setting page_token - it defaults to empty string
    list_webhooks_proto = ListWebhooks()
    list_webhooks_proto.max_results = 10

    # Verify that proto's default page_token is empty string
    assert list_webhooks_proto.page_token == ""

    mock_get_request_message.return_value = list_webhooks_proto
    mock_model_registry_store.list_webhooks.return_value = PagedList([], None)

    _list_webhooks()

    # Verify that list_webhooks was called with page_token=None (not empty string)
    mock_model_registry_store.list_webhooks.assert_called_once()
    call_kwargs = mock_model_registry_store.list_webhooks.call_args.kwargs
    assert call_kwargs.get("page_token") is None
    assert call_kwargs.get("max_results") == 10
