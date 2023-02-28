from itertools import combinations

import json
import pytest
from unittest import mock
from unittest.mock import ANY
import os

from google.cloud.storage import Client
from requests import Response

from mlflow.entities.model_registry import RegisteredModelTag, ModelVersionTag
from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import GetRun
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    CreateRegisteredModelRequest,
    UpdateRegisteredModelRequest,
    DeleteRegisteredModelRequest,
    FinalizeModelVersionRequest,
    FinalizeModelVersionResponse,
    GetRegisteredModelRequest,
    SearchRegisteredModelsRequest,
    CreateModelVersionRequest,
    CreateModelVersionResponse,
    GetModelVersionRequest,
    UpdateModelVersionRequest,
    DeleteModelVersionRequest,
    SearchModelVersionsRequest,
    GetModelVersionDownloadUriRequest,
    GenerateTemporaryModelVersionCredentialsRequest,
    GenerateTemporaryModelVersionCredentialsResponse,
    ModelVersion as ProtoModelVersion,
    MODEL_VERSION_READ_WRITE,
    TemporaryCredentials,
    AwsCredentials,
    AzureUserDelegationSAS,
    GcpOauthToken,
)
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.store.artifact.azure_data_lake_artifact_repo import AzureDataLakeArtifactRepository
from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository
from mlflow.store._unity_catalog.registry.rest_store import (
    UcModelRegistryStore,
    _DATABRICKS_ORG_ID_HEADER,
)
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import MlflowHostCreds
from tests.helper_functions import mock_http_200


def host_creds():
    return MlflowHostCreds("https://hello")


@pytest.fixture
def store():
    return UcModelRegistryStore(host_creds, host_creds)


def _args(endpoint, method, json_body):
    res = {
        "host_creds": host_creds(),
        "endpoint": f"/api/2.0/mlflow/unity-catalog/{endpoint}",
        "method": method,
    }
    if method == "GET":
        res["params"] = json.loads(json_body)
    else:
        res["json"] = json.loads(json_body)
    return res


def _verify_requests(http_request, endpoint, method, proto_message):
    json_body = message_to_json(proto_message)
    http_request.assert_any_call(**(_args(endpoint, method, json_body)))


def _expected_unsupported_method_error_message(method):
    return f"Method '{method}' is unsupported for models in the Unity Catalog"


def _expected_unsupported_arg_error_message(arg):
    return f"Argument '{arg}' is unsupported for models in the Unity Catalog"


def _verify_all_requests(http_request, endpoints, proto_message):
    json_body = message_to_json(proto_message)
    http_request.assert_has_calls(
        [mock.call(**(_args(endpoint, method, json_body))) for endpoint, method in endpoints]
    )


@mock_http_200
def test_create_registered_model(mock_http, store):
    description = "best model ever"
    store.create_registered_model(name="model_1", description=description)
    _verify_requests(
        mock_http,
        "registered-models/create",
        "POST",
        CreateRegisteredModelRequest(name="model_1", description=description),
    )


def test_create_registered_model_with_tags_unsupported(store):
    tags = [
        RegisteredModelTag(key="key", value="value"),
        RegisteredModelTag(key="anotherKey", value="some other value"),
    ]
    description = "best model ever"
    with pytest.raises(MlflowException, match=_expected_unsupported_arg_error_message("tags")):
        store.create_registered_model(name="model_1", tags=tags, description=description)


@mock_http_200
def test_update_registered_model_name(mock_http, store):
    name = "model_1"
    new_name = "model_2"
    with pytest.raises(
        MlflowException, match=_expected_unsupported_method_error_message("rename_registered_model")
    ):
        store.rename_registered_model(name=name, new_name=new_name)


@mock_http_200
def test_update_registered_model_description(mock_http, store):
    name = "model_1"
    description = "test model"
    store.update_registered_model(name=name, description=description)
    _verify_requests(
        mock_http,
        "registered-models/update",
        "PATCH",
        UpdateRegisteredModelRequest(name=name, description=description),
    )


@mock_http_200
def test_delete_registered_model(mock_http, store):
    name = "model_1"
    store.delete_registered_model(name=name)
    _verify_requests(
        mock_http, "registered-models/delete", "DELETE", DeleteRegisteredModelRequest(name=name)
    )


@mock_http_200
def test_search_registered_model(mock_http, store):
    store.search_registered_models()
    _verify_requests(mock_http, "registered-models/search", "GET", SearchRegisteredModelsRequest())
    params_list = [
        {"max_results": 400},
        {"page_token": "blah"},
    ]
    # test all combination of params
    for sz in range(3):
        for combination in combinations(params_list, sz):
            params = {k: v for d in combination for k, v in d.items()}
            store.search_registered_models(**params)
            _verify_requests(
                mock_http,
                "registered-models/search",
                "GET",
                SearchRegisteredModelsRequest(**params),
            )


def test_search_registered_models_invalid_args(store):
    params_list = [
        {"filter_string": "model = 'yo'"},
        {"order_by": ["x", "Y"]},
    ]
    # test all combination of invalid params
    for sz in range(1, 3):
        for combination in combinations(params_list, sz):
            params = {k: v for d in combination for k, v in d.items()}
            with pytest.raises(
                MlflowException, match="unsupported for models in the Unity Catalog"
            ):
                store.search_registered_models(**params)


@mock_http_200
def test_get_registered_model(mock_http, store):
    name = "model_1"
    store.get_registered_model(name=name)
    _verify_requests(
        mock_http, "registered-models/get", "GET", GetRegisteredModelRequest(name=name)
    )


def test_get_latest_versions_unsupported(store):
    name = "model_1"
    expected_err_msg = _expected_unsupported_method_error_message("get_latest_versions")
    with pytest.raises(MlflowException, match=expected_err_msg):
        store.get_latest_versions(name=name)
    with pytest.raises(MlflowException, match=expected_err_msg):
        store.get_latest_versions(name=name, stages=["Production"])


def test_set_registered_model_tag_unsupported(store):
    name = "model_1"
    tag = RegisteredModelTag(key="key", value="value")
    expected_err_msg = _expected_unsupported_method_error_message("set_registered_model_tag")
    with pytest.raises(MlflowException, match=expected_err_msg):
        store.set_registered_model_tag(name=name, tag=tag)


def test_delete_registered_model_tag_unsupported(store):
    name = "model_1"
    expected_err_msg = _expected_unsupported_method_error_message("delete_registered_model_tag")
    with pytest.raises(MlflowException, match=expected_err_msg):
        store.delete_registered_model_tag(name=name, key="key")


def test_download_source_doesnt_leak_files(store, tmp_path):
    parentd = tmp_path.joinpath("data")
    parentd.mkdir()
    parentd.joinpath("a.txt").write_text("A")
    with store._download_source(parentd) as local_dir:
        with open(os.path.join(local_dir, "a.txt"), "r") as handle:
            assert handle.read() == "A"
    assert not os.path.exists(local_dir)


def test_get_workspace_id_returns_none_if_no_request_header(store):
    with mock.patch("mlflow.utils.rest_utils.http_request") as request_mock:
        mock_response = mock.MagicMock(autospec=Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.text = str(dict())
        request_mock.return_value = mock_response
        assert store._get_workspace_id(run_id="some_run_id") is None


def _get_workspace_id_for_run(run_id=None):
    return str(123) if run_id is not None else None


def get_request_mock(
    name, version, source, storage_location, temp_credentials, description=None, run_id=None
):
    def request_mock(
        host_creds,
        endpoint,
        method,
        max_retries=None,
        backoff_factor=None,
        retry_codes=None,
        timeout=None,
        **kwargs,
    ):
        run_workspace_id = _get_workspace_id_for_run(run_id)
        model_version_temp_credentials_response = GenerateTemporaryModelVersionCredentialsResponse(
            credentials=temp_credentials
        )
        req_info_to_response = {
            (
                "/api/2.0/mlflow/unity-catalog/model-versions/create",
                "POST",
                message_to_json(
                    CreateModelVersionRequest(
                        name=name,
                        source=source,
                        description=description,
                        run_id=run_id,
                        run_tracking_server_id=run_workspace_id,
                    )
                ),
            ): CreateModelVersionResponse(
                model_version=ProtoModelVersion(
                    name=name, version=version, storage_location=storage_location
                )
            ),
            (
                "/api/2.0/mlflow/unity-catalog/model-versions/generate-temporary-credentials",
                "POST",
                message_to_json(
                    GenerateTemporaryModelVersionCredentialsRequest(
                        name=name, version=version, operation=MODEL_VERSION_READ_WRITE
                    )
                ),
            ): model_version_temp_credentials_response,
            (
                "/api/2.0/mlflow/unity-catalog/model-versions/finalize",
                "POST",
                message_to_json(FinalizeModelVersionRequest(name=name, version=version)),
            ): FinalizeModelVersionResponse(),
        }
        if run_id is not None:
            req_info_to_response[
                ("/api/2.0/mlflow/runs/get", "GET", message_to_json(GetRun(run_id=run_id)))
            ] = GetRun.Response()

        if method == "POST":
            json_dict = kwargs["json"]
        else:
            json_dict = kwargs["params"]
        response_message = req_info_to_response[(endpoint, method, json.dumps(json_dict, indent=2))]
        mock_resp = mock.MagicMock(autospec=Response)
        mock_resp.status_code = 200
        mock_resp.text = message_to_json(response_message)
        mock_resp.headers = {_DATABRICKS_ORG_ID_HEADER: run_workspace_id}
        return mock_resp

    return request_mock


def _assert_create_model_version_endpoints_called(
    request_mock, name, source, version, run_id=None, description=None
):
    """
    Asserts that endpoints related to the model version creation flow were called on the provided
    `request_mock`
    """
    for endpoint, proto_message in [
        (
            "model-versions/create",
            CreateModelVersionRequest(
                name=name,
                source=source,
                run_id=run_id,
                description=description,
                run_tracking_server_id=_get_workspace_id_for_run(run_id),
            ),
        ),
        (
            "model-versions/generate-temporary-credentials",
            GenerateTemporaryModelVersionCredentialsRequest(
                name=name, version=version, operation=MODEL_VERSION_READ_WRITE
            ),
        ),
        (
            "model-versions/finalize",
            FinalizeModelVersionRequest(name=name, version=version),
        ),
    ]:
        _verify_requests(
            http_request=request_mock,
            endpoint=endpoint,
            method="POST",
            proto_message=proto_message,
        )


def test_create_model_version_aws(store, tmp_path):
    access_key_id = "fake-key"
    secret_access_key = "secret-key"
    session_token = "session-token"
    aws_temp_creds = TemporaryCredentials(
        aws_temp_credentials=AwsCredentials(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
        )
    )
    storage_location = "s3://blah"
    source = str(tmp_path)
    model_name = "model_1"
    version = "1"
    mock_artifact_repo = mock.MagicMock(autospec=S3ArtifactRepository)
    with mock.patch(
        "mlflow.utils.rest_utils.http_request",
        side_effect=get_request_mock(
            name=model_name,
            version=version,
            temp_credentials=aws_temp_creds,
            storage_location=storage_location,
            source=source,
        ),
    ) as request_mock, mock.patch(
        "mlflow.store.artifact.s3_artifact_repo.S3ArtifactRepository",
        return_value=mock_artifact_repo,
    ) as s3_artifact_repo_class_mock:
        store.create_model_version(name=model_name, source=source)
        # Verify that s3 artifact repo mock was called with expected args
        s3_artifact_repo_class_mock.assert_called_once_with(
            artifact_uri=storage_location,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
        )
        mock_artifact_repo.log_artifacts.assert_called_once_with(local_dir=source, artifact_path="")
        _assert_create_model_version_endpoints_called(
            request_mock=request_mock, name=model_name, source=source, version=version
        )


def test_create_model_version_azure(store, tmp_path):
    storage_location = "abfss://filesystem@account.dfs.core.windows.net"
    fake_sas_token = "fake_session_token"
    temporary_creds = TemporaryCredentials(
        azure_user_delegation_sas=AzureUserDelegationSAS(sas_token=fake_sas_token)
    )
    source = str(tmp_path)
    model_name = "model_1"
    version = "1"
    with mock.patch(
        "mlflow.utils.rest_utils.http_request",
        side_effect=get_request_mock(
            name=model_name,
            version=version,
            temp_credentials=temporary_creds,
            storage_location=storage_location,
            source=source,
        ),
    ) as request_mock, mock.patch(
        "mlflow.store.artifact.azure_data_lake_artifact_repo.AzureDataLakeArtifactRepository"
    ) as adls_artifact_repo_class_mock:
        mock_adls_repo = mock.MagicMock(autospec=AzureDataLakeArtifactRepository)
        adls_artifact_repo_class_mock.return_value = mock_adls_repo
        store.create_model_version(name=model_name, source=source)
        adls_artifact_repo_class_mock.assert_called_once_with(
            artifact_uri=storage_location, credential=ANY
        )
        adls_repo_args = adls_artifact_repo_class_mock.call_args_list[0]
        credential = adls_repo_args[1]["credential"]
        assert credential.signature == fake_sas_token
        mock_adls_repo.log_artifacts.assert_called_once_with(local_dir=source, artifact_path="")
        _assert_create_model_version_endpoints_called(
            request_mock=request_mock, name=model_name, source=source, version=version
        )


@pytest.mark.parametrize(
    "create_args",
    [
        ("name", "source"),
        ("name", "source", "description", "run_id"),
    ],
)
def test_create_model_version_gcp(store, tmp_path, create_args):
    storage_location = "gs://test_bucket/some/path"
    fake_oauth_token = "fake_session_token"
    temporary_creds = TemporaryCredentials(
        gcp_oauth_token=GcpOauthToken(oauth_token=fake_oauth_token)
    )
    source = str(tmp_path)
    model_name = "model_1"
    all_create_args = {
        "name": model_name,
        "source": source,
        "description": "my_description",
        "run_id": "some_run_id",
    }
    create_kwargs = {key: value for key, value in all_create_args.items() if key in create_args}
    mock_gcs_repo = mock.MagicMock(autospec=GCSArtifactRepository)
    with mock.patch("mlflow.utils.rest_utils.http_request") as request_mock, mock.patch(
        "google.cloud.storage.Client", return_value=mock.MagicMock(autospec=Client)
    ) as gcs_client_class_mock, mock.patch(
        "mlflow.store.artifact.gcs_artifact_repo.GCSArtifactRepository", return_value=mock_gcs_repo
    ) as gcs_artifact_repo_class_mock:
        version = "1"
        request_mock.side_effect = get_request_mock(
            **create_kwargs,
            version=version,
            temp_credentials=temporary_creds,
            storage_location=storage_location,
        )
        store.create_model_version(**create_kwargs)
        # Verify that gcs artifact repo mock was called with expected args
        gcs_artifact_repo_class_mock.assert_called_once_with(
            artifact_uri=storage_location, client=ANY
        )
        mock_gcs_repo.log_artifacts.assert_called_once_with(local_dir=source, artifact_path="")
        gcs_client_args = gcs_client_class_mock.call_args_list[0]
        credentials = gcs_client_args[1]["credentials"]
        assert credentials.token == fake_oauth_token
        _assert_create_model_version_endpoints_called(
            request_mock=request_mock, version=version, **create_kwargs
        )


def test_create_model_version_unsupported_fields(store):
    with pytest.raises(MlflowException, match=_expected_unsupported_arg_error_message("run_link")):
        store.create_model_version(name="mymodel", source="mysource", run_link="https://google.com")
    with pytest.raises(MlflowException, match=_expected_unsupported_arg_error_message("tags")):
        store.create_model_version(
            name="mymodel", source="mysource", tags=[ModelVersionTag("a", "b")]
        )


def test_transition_model_version_stage_unsupported(store):
    name = "model_1"
    version = "5"
    with pytest.raises(
        MlflowException,
        match=_expected_unsupported_method_error_message("transition_model_version_stage"),
    ):
        store.transition_model_version_stage(
            name=name, version=version, stage="prod", archive_existing_versions=True
        )


@mock_http_200
def test_update_model_version_description(mock_http, store):
    name = "model_1"
    version = "5"
    description = "test model version"
    store.update_model_version(name=name, version=version, description=description)
    _verify_requests(
        mock_http,
        "model-versions/update",
        "PATCH",
        UpdateModelVersionRequest(name=name, version=version, description="test model version"),
    )


@mock_http_200
def test_delete_model_version(mock_http, store):
    name = "model_1"
    version = "12"
    store.delete_model_version(name=name, version=version)
    _verify_requests(
        mock_http,
        "model-versions/delete",
        "DELETE",
        DeleteModelVersionRequest(name=name, version=version),
    )


@mock_http_200
def test_get_model_version_details(mock_http, store):
    name = "model_11"
    version = "8"
    store.get_model_version(name=name, version=version)
    _verify_requests(
        mock_http, "model-versions/get", "GET", GetModelVersionRequest(name=name, version=version)
    )


@mock_http_200
def test_get_model_version_download_uri(mock_http, store):
    name = "model_11"
    version = "8"
    store.get_model_version_download_uri(name=name, version=version)
    _verify_requests(
        mock_http,
        "model-versions/get-download-uri",
        "GET",
        GetModelVersionDownloadUriRequest(name=name, version=version),
    )


@mock_http_200
def test_search_model_versions(mock_http, store):
    store.search_model_versions(filter_string="name='model_12'")
    _verify_requests(
        mock_http,
        "model-versions/search",
        "GET",
        SearchModelVersionsRequest(filter="name='model_12'"),
    )


def test_set_model_version_tag_unsupported(store):
    name = "model_1"
    tag = ModelVersionTag(key="key", value="value")
    with pytest.raises(
        MlflowException, match=_expected_unsupported_method_error_message("set_model_version_tag")
    ):
        store.set_model_version_tag(name=name, version="1", tag=tag)


def test_delete_model_version_tag_unsupported(store):
    name = "model_1"
    with pytest.raises(
        MlflowException,
        match=_expected_unsupported_method_error_message("delete_model_version_tag"),
    ):
        store.delete_model_version_tag(name=name, version="1", key="key")
