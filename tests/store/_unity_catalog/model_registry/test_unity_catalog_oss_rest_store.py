import json
from unittest import mock

import pytest

from mlflow.protos.unity_catalog_oss_messages_pb2 import (
    AwsCredentials,
    CreateRegisteredModel,
    DeleteModelVersion,
    DeleteRegisteredModel,
    FinalizeModelVersion,
    GetModelVersion,
    GetRegisteredModel,
    ListModelVersions,
    ListRegisteredModels,
    ModelVersionInfo,
    TemporaryCredentials,
    UpdateModelVersion,
    UpdateRegisteredModel,
)
from mlflow.store._unity_catalog.registry.uc_oss_rest_store import UnityCatalogOssStore
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.store.artifact.optimized_s3_artifact_repo import OptimizedS3ArtifactRepository
from mlflow.utils.proto_json_utils import message_to_json

from tests.helper_functions import mock_http_200
from tests.store._unity_catalog.conftest import _REGISTRY_HOST_CREDS


@pytest.fixture
def store(mock_databricks_uc_oss_host_creds):
    with mock.patch("mlflow.utils.oss_registry_utils.get_oss_host_creds"):
        yield UnityCatalogOssStore(store_uri="uc")


@pytest.fixture
def creds():
    with mock.patch(
        "mlflow.store._unity_catalog.registry.uc_oss_rest_store.get_oss_host_creds",
        return_value=_REGISTRY_HOST_CREDS,
    ):
        yield


def _args(endpoint, method, json_body, host_creds, extra_headers):
    res = {
        "host_creds": host_creds,
        "endpoint": f"/api/2.1/unity-catalog/{endpoint}",
        "method": method,
        "extra_headers": extra_headers,
    }
    if extra_headers is None:
        del res["extra_headers"]
    if method == "GET":
        res["params"] = json.loads(json_body)
    else:
        res["json"] = json.loads(json_body)
    return res


def _verify_requests(
    http_request,
    endpoint,
    method,
    proto_message,
    host_creds=_REGISTRY_HOST_CREDS,
    extra_headers=None,
):
    json_body = message_to_json(proto_message)
    call_args = _args(endpoint, method, json_body, host_creds, extra_headers)
    http_request.assert_any_call(**call_args)


@mock_http_200
def test_create_registered_model(mock_http, store):
    description = "best model ever"
    store.create_registered_model(name="catalog_1.schema_1.model_1", description=description)
    _verify_requests(
        mock_http,
        "models",
        "POST",
        CreateRegisteredModel(
            name="model_1",
            catalog_name="catalog_1",
            schema_name="schema_1",
            comment=description,
        ),
    )


@mock_http_200
def test_update_registered_model(mock_http, store, creds):
    description = "best model ever"
    store.update_registered_model(name="catalog_1.schema_1.model_1", description=description)
    _verify_requests(
        mock_http,
        "models/catalog_1.schema_1.model_1",
        "PATCH",
        UpdateRegisteredModel(
            full_name="catalog_1.schema_1.model_1",
            comment=description,
        ),
    )


@mock_http_200
def test_get_registered_model(mock_http, store, creds):
    model_name = "catalog_1.schema_1.model_1"
    store.get_registered_model(name=model_name)
    _verify_requests(
        mock_http,
        "models/catalog_1.schema_1.model_1",
        "GET",
        GetRegisteredModel(full_name="catalog_1.schema_1.model_1"),
    )


@mock_http_200
def test_delete_registered_model(mock_http, store, creds):
    model_name = "catalog_1.schema_1.model_1"
    store.delete_registered_model(name=model_name)
    _verify_requests(
        mock_http,
        "models/catalog_1.schema_1.model_1",
        "DELETE",
        DeleteRegisteredModel(
            full_name="catalog_1.schema_1.model_1",
        ),
    )


@mock_http_200
def test_create_model_version(mock_http, store, creds):
    # Mock the context manager for _local_model_dir
    mock_local_model_dir = mock.Mock()
    mock_local_model_dir.__enter__ = mock.Mock(return_value="/mock/local/model/dir")
    mock_local_model_dir.__exit__ = mock.Mock(return_value=None)

    with mock.patch.object(store, "_local_model_dir", return_value=mock_local_model_dir):
        with mock.patch.object(
            store, "_get_artifact_repo", return_value=mock.Mock()
        ) as mock_artifact_repo:
            mock_artifact_repo.log_artifacts.return_value = None

            model_name = "catalog_1.schema_1.model_1"
            store.create_model_version(
                name=model_name, source="source", run_id="run_id", description="description"
            )
            _verify_requests(
                mock_http,
                "models/catalog_1.schema_1.model_1/versions/0/finalize",
                "PATCH",
                FinalizeModelVersion(full_name=model_name, version=0),
            )


@mock_http_200
def test_get_model_version(mock_http, store, creds):
    model_name = "catalog_1.schema_1.model_1"
    version = 0
    store.get_model_version(name=model_name, version=version)
    _verify_requests(
        mock_http,
        "models/catalog_1.schema_1.model_1/versions/0",
        "GET",
        GetModelVersion(full_name=model_name, version=version),
    )


@mock_http_200
def test_update_model_version(mock_http, store, creds):
    model_name = "catalog_1.schema_1.model_1"
    version = 0
    store.update_model_version(name=model_name, version=version, description="new description")
    _verify_requests(
        mock_http,
        "models/catalog_1.schema_1.model_1/versions/0",
        "PATCH",
        UpdateModelVersion(
            full_name=model_name,
            version=0,
            comment="new description",
        ),
    )


@mock_http_200
def test_delete_model_version(mock_http, store, creds):
    model_name = "catalog_1.schema_1.model_1"
    version = 0
    store.delete_model_version(name=model_name, version=version)
    _verify_requests(
        mock_http,
        "models/catalog_1.schema_1.model_1/versions/0",
        "DELETE",
        DeleteModelVersion(full_name=model_name, version=version),
    )


@mock_http_200
def test_search_registered_models(mock_http, store, creds):
    max_results = 10
    page_token = "page_token"
    store.search_registered_models(
        max_results=max_results,
        page_token=page_token,
    )
    _verify_requests(
        mock_http,
        "models",
        "GET",
        ListRegisteredModels(
            max_results=max_results,
            page_token=page_token,
        ),
    )


@mock_http_200
def test_search_model_versions(mock_http, store, creds):
    filter_string = "name = 'catalog_1.schema_1.model_1'"
    max_results = 10
    page_token = "page_token"

    store.search_model_versions(
        filter_string=filter_string,
        max_results=max_results,
        page_token=page_token,
    )

    _verify_requests(
        mock_http,
        "models/catalog_1.schema_1.model_1/versions",
        "GET",
        ListModelVersions(
            full_name="catalog_1.schema_1.model_1", page_token=page_token, max_results=max_results
        ),
    )


def test_get_artifact_repo_file_uri(store, creds):
    model_version_response = ModelVersionInfo(
        model_name="model_1",
        catalog_name="catalog_1",
        schema_name="schema_1",
        version=0,
        source="models:/catalog_1.schema_1.model_1/0",
        storage_location="file:/mock/local/model/dir",
    )
    result = store._get_artifact_repo(model_version_response)
    assert isinstance(result, LocalArtifactRepository)


def test_get_artifact_repo_s3(store, creds):
    model_version_response = ModelVersionInfo(
        model_name="model_1",
        catalog_name="catalog_1",
        schema_name="schema_1",
        version=0,
        source="models:/catalog_1.schema_1.model_1/0",
        storage_location="s3://my_bucket/my/file.txt",
    )
    temporary_creds = TemporaryCredentials(
        aws_temp_credentials=AwsCredentials(
            access_key_id="fake_key_id",
            secret_access_key="fake_secret_access_key",
            session_token="fake_session_token",
        )
    )
    with mock.patch.object(
        store, "_get_temporary_model_version_write_credentials_oss", return_value=temporary_creds
    ):
        result = store._get_artifact_repo(model_version_response)
        assert isinstance(result, OptimizedS3ArtifactRepository)
