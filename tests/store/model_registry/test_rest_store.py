import json
import pytest
import uuid
from unittest import mock

from mlflow.entities.model_registry import RegisteredModelTag, ModelVersionTag
from mlflow.protos.model_registry_pb2 import (
    CreateRegisteredModel,
    UpdateRegisteredModel,
    DeleteRegisteredModel,
    GetRegisteredModel,
    GetLatestVersions,
    CreateModelVersion,
    UpdateModelVersion,
    DeleteModelVersion,
    GetModelVersion,
    GetModelVersionDownloadUri,
    SearchModelVersions,
    RenameRegisteredModel,
    TransitionModelVersionStage,
    SearchRegisteredModels,
    SetRegisteredModelTag,
    SetModelVersionTag,
    DeleteRegisteredModelTag,
    DeleteModelVersionTag,
    SetRegisteredModelAlias,
    DeleteRegisteredModelAlias,
    GetModelVersionByAlias,
)
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import MlflowHostCreds
from tests.helper_functions import mock_http_request_200, mock_http_request_403_200


@pytest.fixture
def creds():
    return MlflowHostCreds("https://hello")


@pytest.fixture
def store(creds):
    return RestStore(lambda: creds)


def _args(host_creds, endpoint, method, json_body):
    res = {"host_creds": host_creds, "endpoint": "/api/2.0/mlflow/%s" % endpoint, "method": method}
    if method == "GET":
        res["params"] = json.loads(json_body)
    else:
        res["json"] = json.loads(json_body)
    return res


def _verify_requests(http_request, creds, endpoint, method, proto_message):
    json_body = message_to_json(proto_message)
    http_request.assert_any_call(**(_args(creds, endpoint, method, json_body)))


def _verify_all_requests(http_request, creds, endpoints, proto_message):
    json_body = message_to_json(proto_message)
    http_request.assert_has_calls(
        [mock.call(**(_args(creds, endpoint, method, json_body))) for endpoint, method in endpoints]
    )


def test_create_registered_model(store, creds):
    tags = [
        RegisteredModelTag(key="key", value="value"),
        RegisteredModelTag(key="anotherKey", value="some other value"),
    ]
    description = "best model ever"
    with mock_http_request_200() as mock_http:
        store.create_registered_model("model_1", tags, description)
    _verify_requests(
        mock_http,
        creds,
        "registered-models/create",
        "POST",
        CreateRegisteredModel(
            name="model_1", tags=[tag.to_proto() for tag in tags], description=description
        ),
    )


def test_update_registered_model_name(store, creds):
    name = "model_1"
    new_name = "model_2"
    with mock_http_request_200() as mock_http:
        store.rename_registered_model(name=name, new_name=new_name)
    _verify_requests(
        mock_http,
        creds,
        "registered-models/rename",
        "POST",
        RenameRegisteredModel(name=name, new_name=new_name),
    )


def test_update_registered_model_description(store, creds):
    name = "model_1"
    description = "test model"
    with mock_http_request_200() as mock_http:
        store.update_registered_model(name=name, description=description)
    _verify_requests(
        mock_http,
        creds,
        "registered-models/update",
        "PATCH",
        UpdateRegisteredModel(name=name, description=description),
    )


def test_delete_registered_model(store, creds):
    name = "model_1"
    with mock_http_request_200() as mock_http:
        store.delete_registered_model(name=name)
    _verify_requests(
        mock_http, creds, "registered-models/delete", "DELETE", DeleteRegisteredModel(name=name)
    )


def test_search_registered_models(store, creds):
    with mock_http_request_200() as mock_http:
        store.search_registered_models()
    _verify_requests(mock_http, creds, "registered-models/search", "GET", SearchRegisteredModels())


@pytest.mark.parametrize("filter_string", [None, "model = 'yo'"])
@pytest.mark.parametrize("max_results", [None, 400])
@pytest.mark.parametrize("page_token", [None, "blah"])
@pytest.mark.parametrize("order_by", [None, ["x", "Y"]])
def test_search_registered_models_params(
    store, creds, filter_string, max_results, page_token, order_by
):
    params = {
        "filter_string": filter_string,
        "max_results": max_results,
        "page_token": page_token,
        "order_by": order_by,
    }
    params = {k: v for k, v in params.items() if v is not None}
    with mock_http_request_200() as mock_http:
        store.search_registered_models(**params)
    if "filter_string" in params:
        params["filter"] = params.pop("filter_string")
    _verify_requests(
        mock_http,
        creds,
        "registered-models/search",
        "GET",
        SearchRegisteredModels(**params),
    )


def test_get_registered_model(store, creds):
    name = "model_1"
    with mock_http_request_200() as mock_http:
        store.get_registered_model(name=name)
    _verify_requests(
        mock_http, creds, "registered-models/get", "GET", GetRegisteredModel(name=name)
    )


def test_get_latest_versions(store, creds):
    name = "model_1"
    with mock_http_request_403_200() as mock_http:
        store.get_latest_versions(name=name)
    endpoint = "registered-models/get-latest-versions"
    endpoints = [(endpoint, "POST"), (endpoint, "GET")]
    _verify_all_requests(mock_http, creds, endpoints, GetLatestVersions(name=name))


def test_get_latest_versions_with_stages(store, creds):
    name = "model_1"
    with mock_http_request_403_200() as mock_http:
        store.get_latest_versions(name=name, stages=["blaah"])
    endpoint = "registered-models/get-latest-versions"
    endpoints = [(endpoint, "POST"), (endpoint, "GET")]
    _verify_all_requests(
        mock_http, creds, endpoints, GetLatestVersions(name=name, stages=["blaah"])
    )


def test_set_registered_model_tag(store, creds):
    name = "model_1"
    tag = RegisteredModelTag(key="key", value="value")
    with mock_http_request_200() as mock_http:
        store.set_registered_model_tag(name=name, tag=tag)
    _verify_requests(
        mock_http,
        creds,
        "registered-models/set-tag",
        "POST",
        SetRegisteredModelTag(name=name, key=tag.key, value=tag.value),
    )


def test_delete_registered_model_tag(store, creds):
    name = "model_1"
    with mock_http_request_200() as mock_http:
        store.delete_registered_model_tag(name=name, key="key")
    _verify_requests(
        mock_http,
        creds,
        "registered-models/delete-tag",
        "DELETE",
        DeleteRegisteredModelTag(name=name, key="key"),
    )


def test_create_model_version(store, creds):
    with mock_http_request_200() as mock_http:
        store.create_model_version("model_1", "path/to/source")
    _verify_requests(
        mock_http,
        creds,
        "model-versions/create",
        "POST",
        CreateModelVersion(name="model_1", source="path/to/source"),
    )
    # test optional fields
    run_id = uuid.uuid4().hex
    tags = [
        ModelVersionTag(key="key", value="value"),
        ModelVersionTag(key="anotherKey", value="some other value"),
    ]
    run_link = "localhost:5000/path/to/run"
    description = "version description"
    with mock_http_request_200() as mock_http:
        store.create_model_version(
            "model_1",
            "path/to/source",
            run_id,
            tags,
            run_link=run_link,
            description=description,
        )
    _verify_requests(
        mock_http,
        creds,
        "model-versions/create",
        "POST",
        CreateModelVersion(
            name="model_1",
            source="path/to/source",
            run_id=run_id,
            run_link=run_link,
            tags=[tag.to_proto() for tag in tags],
            description=description,
        ),
    )


def test_transition_model_version_stage(store, creds):
    name = "model_1"
    version = "5"
    with mock_http_request_200() as mock_http:
        store.transition_model_version_stage(
            name=name, version=version, stage="prod", archive_existing_versions=True
        )
    _verify_requests(
        mock_http,
        creds,
        "model-versions/transition-stage",
        "POST",
        TransitionModelVersionStage(
            name=name, version=version, stage="prod", archive_existing_versions=True
        ),
    )


def test_update_model_version_description(store, creds):
    name = "model_1"
    version = "5"
    description = "test model version"
    with mock_http_request_200() as mock_http:
        store.update_model_version(name=name, version=version, description=description)
    _verify_requests(
        mock_http,
        creds,
        "model-versions/update",
        "PATCH",
        UpdateModelVersion(name=name, version=version, description="test model version"),
    )


def test_delete_model_version(store, creds):
    name = "model_1"
    version = "12"
    with mock_http_request_200() as mock_http:
        store.delete_model_version(name=name, version=version)
    _verify_requests(
        mock_http,
        creds,
        "model-versions/delete",
        "DELETE",
        DeleteModelVersion(name=name, version=version),
    )


def test_get_model_version_details(store, creds):
    name = "model_11"
    version = "8"
    with mock_http_request_200() as mock_http:
        store.get_model_version(name=name, version=version)
    _verify_requests(
        mock_http, creds, "model-versions/get", "GET", GetModelVersion(name=name, version=version)
    )


def test_get_model_version_download_uri(store, creds):
    name = "model_11"
    version = "8"
    with mock_http_request_200() as mock_http:
        store.get_model_version_download_uri(name=name, version=version)
    _verify_requests(
        mock_http,
        creds,
        "model-versions/get-download-uri",
        "GET",
        GetModelVersionDownloadUri(name=name, version=version),
    )


def test_search_model_versions(store, creds):
    with mock_http_request_200() as mock_http:
        store.search_model_versions()
    _verify_requests(mock_http, creds, "model-versions/search", "GET", SearchModelVersions())


@pytest.mark.parametrize("filter_string", [None, "name = 'model_12'"])
@pytest.mark.parametrize("max_results", [None, 400])
@pytest.mark.parametrize("page_token", [None, "blah"])
@pytest.mark.parametrize("order_by", ["version DESC", "creation_time DESC"])
def test_search_model_versions_params(
    store, creds, filter_string, max_results, page_token, order_by
):
    params = {
        "filter_string": filter_string,
        "max_results": max_results,
        "page_token": page_token,
        "order_by": order_by,
    }
    params = {k: v for k, v in params.items() if v is not None}
    with mock_http_request_200() as mock_http:
        store.search_model_versions(**params)
    if "filter_string" in params:
        params["filter"] = params.pop("filter_string")
    _verify_requests(
        mock_http,
        creds,
        "model-versions/search",
        "GET",
        SearchModelVersions(**params),
    )


def test_set_model_version_tag(store, creds):
    name = "model_1"
    tag = ModelVersionTag(key="key", value="value")
    with mock_http_request_200() as mock_http:
        store.set_model_version_tag(name=name, version="1", tag=tag)
    _verify_requests(
        mock_http,
        creds,
        "model-versions/set-tag",
        "POST",
        SetModelVersionTag(name=name, version="1", key=tag.key, value=tag.value),
    )


def test_delete_model_version_tag(store, creds):
    name = "model_1"
    with mock_http_request_200() as mock_http:
        store.delete_model_version_tag(name=name, version="1", key="key")
    _verify_requests(
        mock_http,
        creds,
        "model-versions/delete-tag",
        "DELETE",
        DeleteModelVersionTag(name=name, version="1", key="key"),
    )


def test_set_registered_model_alias(store, creds):
    name = "model_1"
    with mock_http_request_200() as mock_http:
        store.set_registered_model_alias(name=name, alias="test_alias", version="1")
    _verify_requests(
        mock_http,
        creds,
        "registered-models/alias",
        "POST",
        SetRegisteredModelAlias(name=name, alias="test_alias", version="1"),
    )


def test_delete_registered_model_alias(store, creds):
    name = "model_1"
    with mock_http_request_200() as mock_http:
        store.delete_registered_model_alias(name=name, alias="test_alias")
    _verify_requests(
        mock_http,
        creds,
        "registered-models/alias",
        "DELETE",
        DeleteRegisteredModelAlias(name=name, alias="test_alias"),
    )


def test_get_model_version_by_alias(store, creds):
    name = "model_1"
    with mock_http_request_200() as mock_http:
        store.get_model_version_by_alias(name=name, alias="test_alias")
    _verify_requests(
        mock_http,
        creds,
        "registered-models/alias",
        "GET",
        GetModelVersionByAlias(name=name, alias="test_alias"),
    )
