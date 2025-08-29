import base64
import json
import os
import shutil
from itertools import combinations
from unittest import mock
from unittest.mock import ANY

import pandas as pd
import pytest
import yaml
from botocore.client import BaseClient
from google.cloud.storage import Client
from requests import Response

from mlflow.data.dataset import Dataset
from mlflow.data.delta_dataset_source import DeltaDatasetSource
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.entities.logged_model import LoggedModel
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.entities.model_registry.prompt import Prompt
from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.entities.run import Run
from mlflow.entities.run_data import RunData
from mlflow.entities.run_info import RunInfo
from mlflow.entities.run_inputs import RunInputs
from mlflow.entities.run_tag import RunTag
from mlflow.exceptions import MlflowException, RestException
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature, Schema
from mlflow.prompt.constants import (
    LINKED_PROMPTS_TAG_KEY,
    PROMPT_TYPE_CHAT,
    PROMPT_TYPE_TAG_KEY,
    PROMPT_TYPE_TEXT,
    RESPONSE_FORMAT_TAG_KEY,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    MODEL_VERSION_OPERATION_READ_WRITE,
    AwsCredentials,
    AzureUserDelegationSAS,
    CreateModelVersionRequest,
    CreateModelVersionResponse,
    CreateRegisteredModelRequest,
    DeleteModelVersionRequest,
    DeleteModelVersionTagRequest,
    DeleteRegisteredModelAliasRequest,
    DeleteRegisteredModelRequest,
    DeleteRegisteredModelTagRequest,
    EncryptionDetails,
    Entity,
    FinalizeModelVersionRequest,
    FinalizeModelVersionResponse,
    GcpOauthToken,
    GenerateTemporaryModelVersionCredentialsRequest,
    GenerateTemporaryModelVersionCredentialsResponse,
    GetModelVersionByAliasRequest,
    GetModelVersionDownloadUriRequest,
    GetModelVersionRequest,
    GetRegisteredModelRequest,
    Job,
    LineageHeaderInfo,
    ModelVersion,
    Notebook,
    SearchModelVersionsRequest,
    SearchRegisteredModelsRequest,
    SetModelVersionTagRequest,
    SetRegisteredModelAliasRequest,
    SetRegisteredModelTagRequest,
    SseEncryptionAlgorithm,
    SseEncryptionDetails,
    StorageMode,
    TemporaryCredentials,
    UpdateModelVersionRequest,
    UpdateRegisteredModelRequest,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import ModelVersion as ProtoModelVersion
from mlflow.protos.service_pb2 import GetRun
from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
    LinkPromptsToTracesRequest,
    LinkPromptVersionsToModelsRequest,
    LinkPromptVersionsToRunsRequest,
)
from mlflow.store._unity_catalog.registry.rest_store import (
    _DATABRICKS_LINEAGE_ID_HEADER,
    _DATABRICKS_ORG_ID_HEADER,
    UcModelRegistryStore,
)
from mlflow.store.artifact.azure_data_lake_artifact_repo import AzureDataLakeArtifactRepository
from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository
from mlflow.store.artifact.optimized_s3_artifact_repo import OptimizedS3ArtifactRepository
from mlflow.store.artifact.presigned_url_artifact_repo import PresignedUrlArtifactRepository
from mlflow.types.schema import ColSpec, DataType
from mlflow.utils._unity_catalog_utils import (
    _ACTIVE_CATALOG_QUERY,
    _ACTIVE_SCHEMA_QUERY,
    get_artifact_repo_from_storage_info,
    uc_model_version_tag_from_mlflow_tags,
    uc_registered_model_tag_from_mlflow_tags,
)
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATABRICKS_JOB_ID,
    MLFLOW_DATABRICKS_JOB_RUN_ID,
    MLFLOW_DATABRICKS_NOTEBOOK_ID,
)
from mlflow.utils.proto_json_utils import message_to_json

from tests.helper_functions import mock_http_200
from tests.resources.data.dataset_source import SampleDatasetSource
from tests.store._unity_catalog.conftest import (
    _REGISTRY_HOST_CREDS,
    _TRACKING_HOST_CREDS,
)


@pytest.fixture
def store(mock_databricks_uc_host_creds):
    with mock.patch("mlflow.utils.databricks_utils.get_databricks_host_creds"):
        yield UcModelRegistryStore(store_uri="databricks-uc", tracking_uri="databricks")


@pytest.fixture
def spark_session(request):
    with mock.patch(
        "mlflow.store._unity_catalog.registry.rest_store._get_active_spark_session"
    ) as spark_session_getter:
        spark = mock.MagicMock()
        spark_session_getter.return_value = spark

        # Define a custom side effect function for spark sql queries
        def sql_side_effect(query):
            if query == _ACTIVE_CATALOG_QUERY:
                catalog_response_mock = mock.MagicMock()
                catalog_response_mock.collect.return_value = [{"catalog": request.param}]
                return catalog_response_mock
            elif query == _ACTIVE_SCHEMA_QUERY:
                schema_response_mock = mock.MagicMock()
                schema_response_mock.collect.return_value = [{"schema": "default"}]
                return schema_response_mock
            else:
                raise ValueError(f"Unexpected query: {query}")

        spark.sql.side_effect = sql_side_effect
        yield spark


def _args(endpoint, method, json_body, host_creds, extra_headers):
    res = {
        "host_creds": host_creds,
        "endpoint": f"/api/2.0/mlflow/unity-catalog/{endpoint}",
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
    http_request.assert_any_call(**(call_args))


def _expected_unsupported_method_error_message(method):
    return f"Method '{method}' is unsupported for models in the Unity Catalog"


def _expected_unsupported_arg_error_message(arg):
    return f"Argument '{arg}' is unsupported for models in the Unity Catalog"


@mock_http_200
def test_create_registered_model(mock_http, store):
    description = "best model ever"
    tags = [
        RegisteredModelTag(key="key", value="value"),
        RegisteredModelTag(key="anotherKey", value="some other value"),
    ]
    store.create_registered_model(name="model_1", description=description, tags=tags)
    _verify_requests(
        mock_http,
        "registered-models/create",
        "POST",
        CreateRegisteredModelRequest(
            name="model_1",
            description=description,
            tags=uc_registered_model_tag_from_mlflow_tags(tags),
        ),
    )


def test_create_registered_model_three_level_name_hint(store):
    """Test that creating a registered model with invalid name provides legacy registry hint."""
    # Mock the _call_endpoint method to raise a RestException with
    # "specify all three levels" message
    original_error_message = "Model name must specify all three levels"
    rest_exception = RestException(
        {"error_code": "INVALID_PARAMETER_VALUE", "message": original_error_message}
    )

    with mock.patch.object(store, "_call_endpoint", side_effect=rest_exception):
        with pytest.raises(MlflowException, match=original_error_message) as exc_info:
            store.create_registered_model(name="invalid_model")

    # Verify the exception message includes the original error and the legacy registry hint
    expected_hint = (
        "If you are trying to use the legacy Workspace Model Registry, instead of the"
        " recommended Unity Catalog Model Registry, set the Model Registry URI to"
        " 'databricks' (legacy) instead of 'databricks-uc' (recommended)."
    )
    assert original_error_message in str(exc_info.value)
    assert expected_hint in str(exc_info.value)


def test_create_registered_model_three_level_name_hint_with_period(store):
    """Test the hint works correctly when original error message ends with a period."""
    original_error_message = "Model name must specify all three levels."
    rest_exception = RestException(
        {"error_code": "INVALID_PARAMETER_VALUE", "message": original_error_message}
    )

    with mock.patch.object(store, "_call_endpoint", side_effect=rest_exception):
        with pytest.raises(MlflowException, match=original_error_message) as exc_info:
            store.create_registered_model(name="invalid_model")

    # Verify the period is removed before adding the hint
    expected_hint = (
        "If you are trying to use the legacy Workspace Model Registry, instead of the"
        " recommended Unity Catalog Model Registry, set the Model Registry URI to"
        " 'databricks' (legacy) instead of 'databricks-uc' (recommended)."
    )
    error_message = str(exc_info.value)
    assert "Model name must specify all three levels" in error_message
    assert expected_hint in error_message
    # Should not have double periods
    assert ". ." not in error_message


def test_create_registered_model_metastore_does_not_exist_hint(store):
    """
    Test that creating a registered model when metastore doesn't exist
    provides legacy registry hint.
    """
    # Mock the _call_endpoint method to raise a RestException with
    # "METASTORE_DOES_NOT_EXIST" message
    original_error_message = "METASTORE_DOES_NOT_EXIST: Metastore not found"
    rest_exception = RestException(
        {"error_code": "METASTORE_DOES_NOT_EXIST", "message": original_error_message}
    )

    with mock.patch.object(store, "_call_endpoint", side_effect=rest_exception):
        with pytest.raises(MlflowException, match=original_error_message) as exc_info:
            store.create_registered_model(name="test_model")

    # Verify the exception message includes the original error and the legacy registry hint
    expected_hint = (
        "If you are trying to use the Model Registry in a Databricks workspace that"
        " does not have Unity Catalog enabled, either enable Unity Catalog in the"
        " workspace (recommended) or set the Model Registry URI to 'databricks' to"
        " use the legacy Workspace Model Registry."
    )
    error_message = str(exc_info.value)
    assert original_error_message in error_message
    assert expected_hint in error_message


def test_create_registered_model_other_rest_exceptions_not_modified(store):
    """
    Test that RestExceptions unrelated to bad UC model names are not modified
    and are re-raised as-is.
    """
    original_error_message = "Some other error"
    rest_exception = RestException(
        {"error_code": "INTERNAL_ERROR", "message": original_error_message}
    )

    with mock.patch.object(store, "_call_endpoint", side_effect=rest_exception):
        with pytest.raises(RestException, match=original_error_message) as exc_info:
            store.create_registered_model(name="some_model")

    # Verify the original RestException is re-raised without modification
    assert str(exc_info.value) == "INTERNAL_ERROR: Some other error"


@pytest.fixture
def local_model_dir(tmp_path):
    fake_signature = ModelSignature(
        inputs=Schema([ColSpec(DataType.string)]), outputs=Schema([ColSpec(DataType.double)])
    )
    fake_mlmodel_contents = {
        "artifact_path": "some-artifact-path",
        "run_id": "abc123",
        "signature": fake_signature.to_dict(),
    }
    with open(tmp_path.joinpath(MLMODEL_FILE_NAME), "w") as handle:
        yaml.dump(fake_mlmodel_contents, handle)
    return tmp_path


@pytest.fixture
def langchain_local_model_dir(tmp_path):
    fake_signature = ModelSignature(
        inputs=Schema([ColSpec(DataType.string)]), outputs=Schema([ColSpec(DataType.string)])
    )
    fake_mlmodel_contents = {
        "artifact_path": "some-artifact-path",
        "run_id": "abc123",
        "signature": fake_signature.to_dict(),
        "flavors": {
            "langchain": {
                "databricks_dependency": {
                    "databricks_vector_search_index_name": ["index1", "index2"],
                    "databricks_embeddings_endpoint_name": ["embedding_endpoint"],
                    "databricks_llm_endpoint_name": ["llm_endpoint"],
                    "databricks_chat_endpoint_name": ["chat_endpoint"],
                }
            }
        },
    }
    with open(tmp_path.joinpath(MLMODEL_FILE_NAME), "w") as handle:
        yaml.dump(fake_mlmodel_contents, handle)
    return tmp_path


@pytest.fixture(params=[True, False])  # True tests with resources and False tests with auth policy
def langchain_local_model_dir_with_resources(request, tmp_path):
    fake_signature = ModelSignature(
        inputs=Schema([ColSpec(DataType.string)]), outputs=Schema([ColSpec(DataType.string)])
    )
    if request.param:
        fake_mlmodel_contents = {
            "artifact_path": "some-artifact-path",
            "run_id": "abc123",
            "signature": fake_signature.to_dict(),
            "resources": {
                "databricks": {
                    "serving_endpoint": [
                        {"name": "embedding_endpoint"},
                        {"name": "llm_endpoint"},
                        {"name": "chat_endpoint"},
                    ],
                    "vector_search_index": [{"name": "index1"}, {"name": "index2"}],
                    "function": [
                        {"name": "test.schema.test_function"},
                        {"name": "test.schema.test_function_2"},
                    ],
                    "uc_connection": [{"name": "test_connection"}],
                    "table": [
                        {"name": "test.schema.test_table"},
                        {"name": "test.schema.test_table_2"},
                    ],
                }
            },
        }
    else:
        fake_mlmodel_contents = {
            "artifact_path": "some-artifact-path",
            "run_id": "abc123",
            "signature": fake_signature.to_dict(),
            "auth_policy": {
                "system_auth_policy": {
                    "resources": {
                        "databricks": {
                            "serving_endpoint": [
                                {"name": "embedding_endpoint"},
                                {"name": "llm_endpoint"},
                                {"name": "chat_endpoint"},
                            ],
                            "vector_search_index": [{"name": "index1"}, {"name": "index2"}],
                            "function": [
                                {"name": "test.schema.test_function"},
                                {"name": "test.schema.test_function_2"},
                            ],
                            "uc_connection": [{"name": "test_connection"}],
                            "table": [
                                {"name": "test.schema.test_table"},
                                {"name": "test.schema.test_table_2"},
                            ],
                        }
                    },
                },
                "user_auth_policy": {
                    "api_scopes": [
                        "serving.serving-endpoints,vectorsearch.vector-search-endpoints,vectorsearch.vector-search-indexes"
                    ]
                },
            },
        }
    with open(tmp_path.joinpath(MLMODEL_FILE_NAME), "w") as handle:
        yaml.dump(fake_mlmodel_contents, handle)

    model_version_dependencies = [
        {"type": "DATABRICKS_VECTOR_INDEX", "name": "index1"},
        {"type": "DATABRICKS_VECTOR_INDEX", "name": "index2"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "embedding_endpoint"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "llm_endpoint"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "chat_endpoint"},
        {"type": "DATABRICKS_UC_FUNCTION", "name": "test.schema.test_function"},
        {"type": "DATABRICKS_UC_FUNCTION", "name": "test.schema.test_function_2"},
        {"type": "DATABRICKS_UC_CONNECTION", "name": "test_connection"},
        {"type": "DATABRICKS_TABLE", "name": "test.schema.test_table"},
        {"type": "DATABRICKS_TABLE", "name": "test.schema.test_table_2"},
    ]

    return (tmp_path, model_version_dependencies)


@pytest.fixture
def langchain_local_model_dir_with_invoker_resources(tmp_path):
    fake_signature = ModelSignature(
        inputs=Schema([ColSpec(DataType.string)]), outputs=Schema([ColSpec(DataType.string)])
    )
    fake_mlmodel_contents = {
        "artifact_path": "some-artifact-path",
        "run_id": "abc123",
        "signature": fake_signature.to_dict(),
        "resources": {
            "databricks": {
                "serving_endpoint": [
                    {"name": "embedding_endpoint"},
                    {"name": "llm_endpoint"},
                    {"name": "chat_endpoint"},
                ],
                "vector_search_index": [
                    {"name": "index1", "on_behalf_of_user": False},
                    {"name": "index2"},
                ],
                "function": [
                    {"name": "test.schema.test_function", "on_behalf_of_user": True},
                    {"name": "test.schema.test_function_2"},
                ],
                "uc_connection": [{"name": "test_connection", "on_behalf_of_user": False}],
                "table": [
                    {"name": "test.schema.test_table", "on_behalf_of_user": True},
                    {"name": "test.schema.test_table_2"},
                ],
            }
        },
    }
    with open(tmp_path.joinpath(MLMODEL_FILE_NAME), "w") as handle:
        yaml.dump(fake_mlmodel_contents, handle)

    model_version_dependencies = [
        {"type": "DATABRICKS_VECTOR_INDEX", "name": "index1"},
        {"type": "DATABRICKS_VECTOR_INDEX", "name": "index2"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "embedding_endpoint"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "llm_endpoint"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "chat_endpoint"},
        {"type": "DATABRICKS_UC_FUNCTION", "name": "test.schema.test_function_2"},
        {"type": "DATABRICKS_UC_CONNECTION", "name": "test_connection"},
        {"type": "DATABRICKS_TABLE", "name": "test.schema.test_table_2"},
    ]

    return (tmp_path, model_version_dependencies)


@pytest.fixture
def langchain_local_model_dir_no_dependencies(tmp_path):
    fake_signature = ModelSignature(
        inputs=Schema([ColSpec(DataType.string)]), outputs=Schema([ColSpec(DataType.string)])
    )
    fake_mlmodel_contents = {
        "artifact_path": "some-artifact-path",
        "run_id": "abc123",
        "signature": fake_signature.to_dict(),
        "flavors": {"langchain": {}},
    }
    with open(tmp_path.joinpath(MLMODEL_FILE_NAME), "w") as handle:
        yaml.dump(fake_mlmodel_contents, handle)
    return tmp_path


def test_create_model_version_with_langchain_dependencies(store, langchain_local_model_dir):
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
    source = str(langchain_local_model_dir)
    model_name = "model_1"
    version = "1"
    tags = [
        ModelVersionTag(key="key", value="value"),
        ModelVersionTag(key="anotherKey", value="some other value"),
    ]
    model_version_dependencies = [
        {"type": "DATABRICKS_VECTOR_INDEX", "name": "index1"},
        {"type": "DATABRICKS_VECTOR_INDEX", "name": "index2"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "embedding_endpoint"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "llm_endpoint"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "chat_endpoint"},
    ]

    mock_artifact_repo = mock.MagicMock(autospec=OptimizedS3ArtifactRepository)
    with (
        mock.patch(
            "mlflow.utils.rest_utils.http_request",
            side_effect=get_request_mock(
                name=model_name,
                version=version,
                temp_credentials=aws_temp_creds,
                storage_location=storage_location,
                source=source,
                tags=tags,
                model_version_dependencies=model_version_dependencies,
            ),
        ) as request_mock,
        mock.patch(
            "mlflow.store.artifact.optimized_s3_artifact_repo.OptimizedS3ArtifactRepository",
            return_value=mock_artifact_repo,
        ) as optimized_s3_artifact_repo_class_mock,
        mock.patch.dict("sys.modules", {"boto3": {}}),
    ):
        store.create_model_version(name=model_name, source=source, tags=tags)
        # Verify that s3 artifact repo mock was called with expected args
        optimized_s3_artifact_repo_class_mock.assert_called_once_with(
            artifact_uri=storage_location,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            credential_refresh_def=ANY,
            s3_upload_extra_args={},
        )
        mock_artifact_repo.log_artifacts.assert_called_once_with(local_dir=ANY, artifact_path="")
        _assert_create_model_version_endpoints_called(
            request_mock=request_mock,
            name=model_name,
            source=source,
            version=version,
            tags=tags,
            model_version_dependencies=model_version_dependencies,
        )


def test_create_model_version_with_resources(store, langchain_local_model_dir_with_resources):
    source, model_version_dependencies = langchain_local_model_dir_with_resources
    source = str(source)
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
    model_name = "model_1"
    version = "1"
    tags = [
        ModelVersionTag(key="key", value="value"),
        ModelVersionTag(key="anotherKey", value="some other value"),
    ]

    mock_artifact_repo = mock.MagicMock(autospec=OptimizedS3ArtifactRepository)
    with (
        mock.patch(
            "mlflow.utils.rest_utils.http_request",
            side_effect=get_request_mock(
                name=model_name,
                version=version,
                temp_credentials=aws_temp_creds,
                storage_location=storage_location,
                source=source,
                tags=tags,
                model_version_dependencies=model_version_dependencies,
            ),
        ) as request_mock,
        mock.patch(
            "mlflow.store.artifact.optimized_s3_artifact_repo.OptimizedS3ArtifactRepository",
            return_value=mock_artifact_repo,
        ) as optimized_s3_artifact_repo_class_mock,
        mock.patch.dict("sys.modules", {"boto3": {}}),
    ):
        store.create_model_version(name=model_name, source=source, tags=tags)
        # Verify that s3 artifact repo mock was called with expected args
        optimized_s3_artifact_repo_class_mock.assert_called_once_with(
            artifact_uri=storage_location,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            credential_refresh_def=ANY,
            s3_upload_extra_args={},
        )
        mock_artifact_repo.log_artifacts.assert_called_once_with(local_dir=ANY, artifact_path="")
        _assert_create_model_version_endpoints_called(
            request_mock=request_mock,
            name=model_name,
            source=source,
            version=version,
            tags=tags,
            model_version_dependencies=model_version_dependencies,
        )


def test_create_model_version_with_invoker_resources(
    store, langchain_local_model_dir_with_invoker_resources
):
    source, model_version_dependencies = langchain_local_model_dir_with_invoker_resources
    source = str(source)
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
    model_name = "model_1"
    version = "1"
    tags = [
        ModelVersionTag(key="key", value="value"),
        ModelVersionTag(key="anotherKey", value="some other value"),
    ]

    mock_artifact_repo = mock.MagicMock(autospec=OptimizedS3ArtifactRepository)
    with (
        mock.patch(
            "mlflow.utils.rest_utils.http_request",
            side_effect=get_request_mock(
                name=model_name,
                version=version,
                temp_credentials=aws_temp_creds,
                storage_location=storage_location,
                source=source,
                tags=tags,
                model_version_dependencies=model_version_dependencies,
            ),
        ) as request_mock,
        mock.patch(
            "mlflow.store.artifact.optimized_s3_artifact_repo.OptimizedS3ArtifactRepository",
            return_value=mock_artifact_repo,
        ) as optimized_s3_artifact_repo_class_mock,
        mock.patch.dict("sys.modules", {"boto3": {}}),
    ):
        store.create_model_version(name=model_name, source=source, tags=tags)
        # Verify that s3 artifact repo mock was called with expected args
        optimized_s3_artifact_repo_class_mock.assert_called_once_with(
            artifact_uri=storage_location,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            credential_refresh_def=ANY,
            s3_upload_extra_args={},
        )
        mock_artifact_repo.log_artifacts.assert_called_once_with(local_dir=ANY, artifact_path="")
        _assert_create_model_version_endpoints_called(
            request_mock=request_mock,
            name=model_name,
            source=source,
            version=version,
            tags=tags,
            model_version_dependencies=model_version_dependencies,
        )


def test_create_model_version_with_langchain_no_dependencies(
    store, langchain_local_model_dir_no_dependencies
):
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
    source = str(langchain_local_model_dir_no_dependencies)
    model_name = "model_1"
    version = "1"
    tags = [
        ModelVersionTag(key="key", value="value"),
        ModelVersionTag(key="anotherKey", value="some other value"),
    ]
    mock_artifact_repo = mock.MagicMock(autospec=OptimizedS3ArtifactRepository)
    with (
        mock.patch(
            "mlflow.utils.rest_utils.http_request",
            side_effect=get_request_mock(
                name=model_name,
                version=version,
                temp_credentials=aws_temp_creds,
                storage_location=storage_location,
                source=source,
                tags=tags,
                model_version_dependencies=None,
            ),
        ) as request_mock,
        mock.patch(
            "mlflow.store.artifact.optimized_s3_artifact_repo.OptimizedS3ArtifactRepository",
            return_value=mock_artifact_repo,
        ) as optimized_s3_artifact_repo_class_mock,
        mock.patch.dict("sys.modules", {"boto3": {}}),
    ):
        store.create_model_version(name=model_name, source=source, tags=tags)
        # Verify that s3 artifact repo mock was called with expected args
        optimized_s3_artifact_repo_class_mock.assert_called_once_with(
            artifact_uri=storage_location,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            credential_refresh_def=ANY,
            s3_upload_extra_args={},
        )
        mock_artifact_repo.log_artifacts.assert_called_once_with(local_dir=ANY, artifact_path="")
        _assert_create_model_version_endpoints_called(
            request_mock=request_mock,
            name=model_name,
            source=source,
            version=version,
            tags=tags,
            model_version_dependencies=None,
        )


def test_create_model_version_nonexistent_directory(store, tmp_path):
    fake_directory = str(tmp_path.joinpath("myfakepath"))
    with pytest.raises(
        MlflowException,
        match="Unable to download model artifacts from source artifact location",
    ):
        store.create_model_version(name="mymodel", source=fake_directory)


def test_create_model_version_missing_python_deps(store, local_model_dir):
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
    source = str(local_model_dir)
    model_name = "model_1"
    version = "1"
    with (
        mock.patch(
            "mlflow.utils.rest_utils.http_request",
            side_effect=get_request_mock(
                name=model_name,
                version=version,
                temp_credentials=aws_temp_creds,
                storage_location=storage_location,
                source=source,
            ),
        ),
        mock.patch.dict("sys.modules", {"boto3": None}),
        pytest.raises(
            MlflowException,
            match="Unable to import necessary dependencies to access model version files",
        ),
    ):
        store.create_model_version(name=model_name, source=str(local_model_dir))


_TEST_SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec(DataType.double)]), outputs=Schema([ColSpec(DataType.double)])
)


@pytest.fixture
def feature_store_local_model_dir(tmp_path):
    fake_mlmodel_contents = {
        "artifact_path": "some-artifact-path",
        "run_id": "abc123",
        "signature": _TEST_SIGNATURE.to_dict(),
        "flavors": {"python_function": {"loader_module": "databricks.feature_store.mlflow_model"}},
    }
    with open(tmp_path.joinpath(MLMODEL_FILE_NAME), "w") as handle:
        yaml.dump(fake_mlmodel_contents, handle)
    return tmp_path


def test_create_model_version_fails_fs_packaged_model(store, feature_store_local_model_dir):
    with pytest.raises(
        MlflowException,
        match="This model was packaged by Databricks Feature Store and can only be registered on "
        "a Databricks cluster.",
    ):
        store.create_model_version(name="model_1", source=str(feature_store_local_model_dir))


def test_create_model_version_missing_mlmodel(store, tmp_path):
    with pytest.raises(
        MlflowException,
        match="Unable to load model metadata. Ensure the source path of the model "
        "being registered points to a valid MLflow model directory ",
    ):
        store.create_model_version(name="mymodel", source=str(tmp_path))


def test_create_model_version_missing_signature(store, tmp_path):
    tmp_path.joinpath(MLMODEL_FILE_NAME).write_text(json.dumps({"a": "b"}))
    with pytest.raises(
        MlflowException,
        match="Model passed for registration did not contain any signature metadata",
    ):
        store.create_model_version(name="mymodel", source=str(tmp_path))


def test_create_model_version_missing_output_signature(store, tmp_path):
    fake_signature = ModelSignature(inputs=Schema([ColSpec(DataType.integer)]))
    fake_mlmodel_contents = {"signature": fake_signature.to_dict()}
    with open(tmp_path.joinpath(MLMODEL_FILE_NAME), "w") as handle:
        yaml.dump(fake_mlmodel_contents, handle)
    with pytest.raises(
        MlflowException,
        match="Model passed for registration contained a signature that includes only inputs",
    ):
        store.create_model_version(name="mymodel", source=str(tmp_path))


def test_create_model_version_with_optional_signature_validation_bypass_enabled(store, tmp_path):
    # Create a model directory without proper signature
    tmp_path.joinpath(MLMODEL_FILE_NAME).write_text(json.dumps({"a": "b"}))

    # Mock only the essential methods needed to test signature validation bypass
    with (
        mock.patch.object(store, "_validate_model_signature") as mock_validate_signature,
        mock.patch.object(store, "_local_model_dir") as mock_local_model_dir,
        mock.patch.object(store, "_call_endpoint") as mock_call_endpoint,
        mock.patch.object(store, "_get_artifact_repo") as mock_get_artifact_repo,
        mock.patch.object(store, "_finalize_model_version") as mock_finalize,
    ):
        # Setup minimal mocks
        mock_local_model_dir.return_value.__enter__.return_value = tmp_path
        mock_local_model_dir.return_value.__exit__.return_value = None

        # Mock the model version response
        mock_model_version = mock.Mock()
        mock_response = mock.Mock()
        mock_response.model_version = mock_model_version
        mock_call_endpoint.return_value = mock_response

        # Mock artifact repo
        mock_artifact_repo = mock.Mock()
        mock_get_artifact_repo.return_value = mock_artifact_repo

        # Mock finalization
        mock_finalized_mv = mock.Mock()
        mock_finalized_mv.status = 1
        mock_finalized_mv.name = "test_model"
        mock_finalized_mv.version = "1"
        mock_finalized_mv.aliases = []
        mock_finalized_mv.tags = []
        mock_finalized_mv.model_params = []
        mock_finalized_mv.model_metrics = []
        mock_deployment_job_state = mock.Mock()
        mock_deployment_job_state.job_id = "job123"
        mock_deployment_job_state.run_id = "run123"
        mock_deployment_job_state.job_state = 1
        mock_deployment_job_state.run_state = 1
        mock_deployment_job_state.current_task_name = "task1"
        mock_finalized_mv.deployment_job_state = mock_deployment_job_state
        mock_finalize.return_value = mock_finalized_mv

        # Call the method with bypass_signature_validation=True
        store._create_model_version_with_optional_signature_validation(
            name="test_model", source=str(tmp_path), bypass_signature_validation=True
        )

        # Verify that signature validation was bypassed
        mock_validate_signature.assert_not_called()


def test_create_model_version_with_optional_signature_validation_bypass_disabled(store, tmp_path):
    # Create a model directory without proper signature
    tmp_path.joinpath(MLMODEL_FILE_NAME).write_text(json.dumps({"a": "b"}))

    # Mock only the essential methods needed to test signature validation
    with (
        mock.patch.object(store, "_validate_model_signature") as mock_validate_signature,
        mock.patch.object(store, "_local_model_dir") as mock_local_model_dir,
        mock.patch.object(store, "_call_endpoint") as mock_call_endpoint,
        mock.patch.object(store, "_get_artifact_repo") as mock_get_artifact_repo,
        mock.patch.object(store, "_finalize_model_version") as mock_finalize,
    ):
        # Setup minimal mocks
        mock_local_model_dir.return_value.__enter__.return_value = tmp_path
        mock_local_model_dir.return_value.__exit__.return_value = None

        # Mock the model version response
        mock_model_version = mock.Mock()
        mock_response = mock.Mock()
        mock_response.model_version = mock_model_version
        mock_call_endpoint.return_value = mock_response

        # Mock artifact repo
        mock_artifact_repo = mock.Mock()
        mock_get_artifact_repo.return_value = mock_artifact_repo

        # Mock finalization
        mock_finalized_mv = mock.Mock()
        mock_finalized_mv.status = 1
        mock_finalized_mv.name = "test_model"
        mock_finalized_mv.version = "1"
        mock_finalized_mv.aliases = []
        mock_finalized_mv.tags = []
        mock_finalized_mv.model_params = []
        mock_finalized_mv.model_metrics = []
        mock_deployment_job_state = mock.Mock()
        mock_deployment_job_state.job_id = "job123"
        mock_deployment_job_state.run_id = "run123"
        mock_deployment_job_state.job_state = 1
        mock_deployment_job_state.run_state = 1
        mock_deployment_job_state.current_task_name = "task1"
        mock_finalized_mv.deployment_job_state = mock_deployment_job_state
        mock_finalize.return_value = mock_finalized_mv

        # Call the method with bypass_signature_validation=False
        store._create_model_version_with_optional_signature_validation(
            name="test_model", source=str(tmp_path), bypass_signature_validation=False
        )

        # Verify that signature validation was performed
        mock_validate_signature.assert_called_once_with(tmp_path)


@pytest.mark.parametrize(
    ("encryption_details", "extra_args"),
    [
        (
            SseEncryptionDetails(
                algorithm=SseEncryptionAlgorithm.AWS_SSE_S3,
            ),
            {
                "ServerSideEncryption": "AES256",
            },
        ),
        (
            SseEncryptionDetails(
                algorithm=SseEncryptionAlgorithm.AWS_SSE_KMS,
                aws_kms_key_arn="some:arn:test:key/key_id",
            ),
            {
                "ServerSideEncryption": "aws:kms",
                "SSEKMSKeyId": "key_id",
            },
        ),
    ],
)
def test_create_model_version_with_sse_kms_client(
    store, langchain_local_model_dir, encryption_details, extra_args
):
    access_key_id = "fake-key"
    secret_access_key = "secret-key"
    session_token = "session-token"
    aws_temp_creds = TemporaryCredentials(
        aws_temp_credentials=AwsCredentials(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
        ),
        encryption_details=EncryptionDetails(sse_encryption_details=encryption_details),
    )
    storage_location = "s3://blah"
    source = str(langchain_local_model_dir)
    model_name = "model_1"
    version = "1"
    tags = [
        ModelVersionTag(key="key", value="value"),
        ModelVersionTag(key="anotherKey", value="some other value"),
    ]
    model_version_dependencies = [
        {"type": "DATABRICKS_VECTOR_INDEX", "name": "index1"},
        {"type": "DATABRICKS_VECTOR_INDEX", "name": "index2"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "embedding_endpoint"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "llm_endpoint"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "chat_endpoint"},
    ]

    optimized_s3_repo_package = "mlflow.store.artifact.optimized_s3_artifact_repo"
    mock_s3_client = mock.MagicMock(autospec=BaseClient)
    with (
        mock.patch(
            "mlflow.utils.rest_utils.http_request",
            side_effect=get_request_mock(
                name=model_name,
                version=version,
                temp_credentials=aws_temp_creds,
                storage_location=storage_location,
                source=source,
                tags=tags,
                model_version_dependencies=model_version_dependencies,
            ),
        ),
        mock.patch(
            f"{optimized_s3_repo_package}.OptimizedS3ArtifactRepository._get_s3_client",
            return_value=mock_s3_client,
        ),
        mock.patch(
            f"{optimized_s3_repo_package}.OptimizedS3ArtifactRepository._get_region_name",
            return_value="us-east-1",
        ),
        mock.patch.dict("sys.modules", {"boto3": {}}),
    ):
        store.create_model_version(name=model_name, source=source, tags=tags)

        mock_s3_client.upload_file.assert_called_once_with(
            Filename=ANY, Bucket=ANY, Key=ANY, ExtraArgs=extra_args
        )


@pytest.mark.parametrize(
    ("encryption_details", "extra_args"),
    [
        (
            SseEncryptionDetails(
                algorithm=SseEncryptionAlgorithm.AWS_SSE_S3,
            ),
            {
                "ServerSideEncryption": "AES256",
            },
        ),
        (
            SseEncryptionDetails(
                algorithm=SseEncryptionAlgorithm.AWS_SSE_KMS,
                aws_kms_key_arn="some:arn:test:key/key_id",
            ),
            {
                "ServerSideEncryption": "aws:kms",
                "SSEKMSKeyId": "key_id",
            },
        ),
    ],
)
def test_create_model_version_with_sse_kms_store(
    store, langchain_local_model_dir, encryption_details, extra_args
):
    access_key_id = "fake-key"
    secret_access_key = "secret-key"
    session_token = "session-token"
    aws_temp_creds = TemporaryCredentials(
        aws_temp_credentials=AwsCredentials(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
        ),
        encryption_details=EncryptionDetails(sse_encryption_details=encryption_details),
    )
    storage_location = "s3://blah"
    source = str(langchain_local_model_dir)
    model_name = "model_1"
    version = "1"
    tags = [
        ModelVersionTag(key="key", value="value"),
        ModelVersionTag(key="anotherKey", value="some other value"),
    ]
    model_version_dependencies = [
        {"type": "DATABRICKS_VECTOR_INDEX", "name": "index1"},
        {"type": "DATABRICKS_VECTOR_INDEX", "name": "index2"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "embedding_endpoint"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "llm_endpoint"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "chat_endpoint"},
    ]

    mock_artifact_repo = mock.MagicMock(autospec=OptimizedS3ArtifactRepository)
    with (
        mock.patch(
            "mlflow.utils.rest_utils.http_request",
            side_effect=get_request_mock(
                name=model_name,
                version=version,
                temp_credentials=aws_temp_creds,
                storage_location=storage_location,
                source=source,
                tags=tags,
                model_version_dependencies=model_version_dependencies,
            ),
        ),
        mock.patch(
            "mlflow.store.artifact.optimized_s3_artifact_repo.OptimizedS3ArtifactRepository",
            return_value=mock_artifact_repo,
        ) as optimized_s3_artifact_repo_class_mock,
        mock.patch.dict("sys.modules", {"boto3": {}}),
    ):
        store.create_model_version(name=model_name, source=source, tags=tags)
        # Verify that s3 artifact repo mock was called with expected args
        optimized_s3_artifact_repo_class_mock.assert_called_once_with(
            artifact_uri=storage_location,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            credential_refresh_def=ANY,
            s3_upload_extra_args=extra_args,
        )


@pytest.mark.parametrize(
    ("flavor_config", "should_persist_api_called"),
    [
        # persist_pretrained_model should NOT be called for non-transformer models
        (
            {
                "python_function": {},
                "scikit-learn": {},
            },
            False,
        ),
        # persist_pretrained_model should NOT be called if model weights are saved locally
        (
            {
                "transformers": {
                    "model_binary": "model",
                    "source_model_name": "SOME_REPO",
                }
            },
            False,
        ),
        # persist_pretrained_model should be called if model weights are not saved locally
        (
            {
                "transformers": {
                    "source_model_name": "SOME_REPO",
                    "source_model_revision": "SOME_COMMIT_HASH",
                }
            },
            True,
        ),
    ],
)
def test_download_model_weights_if_not_saved(
    flavor_config, should_persist_api_called, store, tmp_path
):
    fake_mlmodel_contents = {
        "artifact_path": "some-artifact-path",
        "run_id": "abc123",
        "flavors": flavor_config,
        "signature": _TEST_SIGNATURE.to_dict(),
    }
    with tmp_path.joinpath(MLMODEL_FILE_NAME).open("w") as handle:
        yaml.dump(fake_mlmodel_contents, handle)

    if model_binary_path := flavor_config.get("transformers", {}).get("model_binary"):
        tmp_path.joinpath(model_binary_path).mkdir()

    with mock.patch("mlflow.transformers") as transformers_mock:
        store._download_model_weights_if_not_saved(str(tmp_path))

        if should_persist_api_called:
            transformers_mock.persist_pretrained_model.assert_called_once_with(str(tmp_path))
        else:
            transformers_mock.persist_pretrained_model.assert_not_called()


@mock_http_200
def test_update_registered_model_name(mock_http, store):
    name = "model_1"
    new_name = "model_2"
    store.rename_registered_model(name=name, new_name=new_name)
    _verify_requests(
        mock_http,
        "registered-models/update",
        "PATCH",
        UpdateRegisteredModelRequest(name=name, new_name=new_name),
    )


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
    with pytest.raises(
        MlflowException,
        match=f"{_expected_unsupported_method_error_message('get_latest_versions')}. "
        "To load the latest version of a model in Unity Catalog, you can set "
        "an alias on the model version and load it by alias",
    ):
        store.get_latest_versions(name=name)
    with pytest.raises(
        MlflowException,
        match=f"{_expected_unsupported_method_error_message('get_latest_versions')}. "
        "Detected attempt to load latest model version in stages",
    ):
        store.get_latest_versions(name=name, stages=["Production"])


@mock_http_200
def test_set_registered_model_tag(mock_http, store):
    name = "model_1"
    tag = RegisteredModelTag(key="key", value="value")
    store.set_registered_model_tag(name=name, tag=tag)
    _verify_requests(
        mock_http,
        "registered-models/set-tag",
        "POST",
        SetRegisteredModelTagRequest(name=name, key=tag.key, value=tag.value),
    )


@mock_http_200
def test_delete_registered_model_tag(mock_http, store):
    name = "model_1"
    store.delete_registered_model_tag(name=name, key="key")
    _verify_requests(
        mock_http,
        "registered-models/delete-tag",
        "DELETE",
        DeleteRegisteredModelTagRequest(name=name, key="key"),
    )


def test_get_notebook_id_returns_none_if_empty_run(store):
    assert store._get_notebook_id(None) is None


def test_get_notebook_id_returns_expected_id(store):
    test_tag = RunTag(key=MLFLOW_DATABRICKS_NOTEBOOK_ID, value="123")
    test_run_data = RunData(tags=[test_tag])
    test_run_info = RunInfo(
        "run_uuid",
        "experiment_id",
        "user_id",
        "status",
        "start_time",
        "end_time",
        "lifecycle_stage",
    )
    test_run = Run(run_data=test_run_data, run_info=test_run_info)
    assert store._get_notebook_id(test_run) == "123"


def test_get_job_id_returns_none_if_empty_run(store):
    assert store._get_job_id(None) is None


def test_get_job_id_returns_expected_id(store):
    test_tag = RunTag(key=MLFLOW_DATABRICKS_JOB_ID, value="123")
    test_run_data = RunData(tags=[test_tag])
    test_run_info = RunInfo(
        "run_uuid",
        "experiment_id",
        "user_id",
        "status",
        "start_time",
        "end_time",
        "lifecycle_stage",
    )
    test_run = Run(run_data=test_run_data, run_info=test_run_info)
    assert store._get_job_id(test_run) == "123"


def test_get_job_run_id_returns_none_if_empty_run(store):
    assert store._get_job_run_id(None) is None


def test_get_job_run_id_returns_expected_id(store):
    test_tag = RunTag(key=MLFLOW_DATABRICKS_JOB_RUN_ID, value="123")
    test_run_data = RunData(tags=[test_tag])
    test_run_info = RunInfo(
        "run_uuid",
        "experiment_id",
        "user_id",
        "status",
        "start_time",
        "end_time",
        "lifecycle_stage",
    )
    test_run = Run(run_data=test_run_data, run_info=test_run_info)
    assert store._get_job_run_id(test_run) == "123"


def test_get_workspace_id_returns_none_if_empty_headers(store):
    assert store._get_workspace_id(None) is None
    bad_headers = {}
    assert store._get_workspace_id(bad_headers) is None


def test_get_workspace_id_returns_expected_id(store):
    good_headers = {_DATABRICKS_ORG_ID_HEADER: "123"}
    assert store._get_workspace_id(good_headers) == "123"


@pytest.mark.parametrize(
    ("status_code", "response_text"),
    [
        (403, "{}"),
        (500, "<html><div>Not real json</div></html>"),
    ],
)
def test_get_run_and_headers_returns_none_if_request_fails(store, status_code, response_text):
    mock_response = mock.MagicMock(autospec=Response)
    mock_response.status_code = status_code
    mock_response.headers = {_DATABRICKS_ORG_ID_HEADER: 123}
    mock_response.text = response_text
    with mock.patch(
        "mlflow.store._unity_catalog.registry.rest_store.http_request", return_value=mock_response
    ):
        assert store._get_run_and_headers(run_id="some_run_id") == (None, None)


def test_get_run_and_headers_returns_none_if_tracking_uri_not_databricks(
    mock_databricks_uc_host_creds, tmp_path
):
    with mock.patch("mlflow.utils.databricks_utils.get_databricks_host_creds"):
        store = UcModelRegistryStore(store_uri="databricks-uc", tracking_uri=str(tmp_path))
        mock_response = mock.MagicMock(autospec=Response)
        mock_response.status_code = 200
        mock_response.headers = {_DATABRICKS_ORG_ID_HEADER: 123}
        mock_response.text = "{}"
        with mock.patch(
            "mlflow.store._unity_catalog.registry.rest_store.http_request",
            return_value=mock_response,
        ):
            assert store._get_run_and_headers(run_id="some_run_id") == (None, None)


def _get_workspace_id_for_run(run_id=None):
    return "123" if run_id is not None else None


def get_request_mock(
    name,
    version,
    source,
    storage_location,
    temp_credentials,
    description=None,
    run_id=None,
    tags=None,
    model_version_dependencies=None,
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
        uc_tags = uc_model_version_tag_from_mlflow_tags(tags) if tags is not None else []
        req_info_to_response = {
            (
                _REGISTRY_HOST_CREDS.host,
                "/api/2.0/mlflow/unity-catalog/model-versions/create",
                "POST",
                message_to_json(
                    CreateModelVersionRequest(
                        name=name,
                        source=source,
                        description=description,
                        run_id=run_id,
                        run_tracking_server_id=run_workspace_id,
                        tags=uc_tags,
                        feature_deps="",
                        model_version_dependencies=model_version_dependencies,
                    )
                ),
            ): CreateModelVersionResponse(
                model_version=ProtoModelVersion(
                    name=name, version=version, storage_location=storage_location, tags=uc_tags
                )
            ),
            (
                _REGISTRY_HOST_CREDS.host,
                "/api/2.0/mlflow/unity-catalog/model-versions/generate-temporary-credentials",
                "POST",
                message_to_json(
                    GenerateTemporaryModelVersionCredentialsRequest(
                        name=name, version=version, operation=MODEL_VERSION_OPERATION_READ_WRITE
                    )
                ),
            ): model_version_temp_credentials_response,
            (
                _REGISTRY_HOST_CREDS.host,
                "/api/2.0/mlflow/unity-catalog/model-versions/finalize",
                "POST",
                message_to_json(FinalizeModelVersionRequest(name=name, version=version)),
            ): FinalizeModelVersionResponse(),
        }
        if run_id is not None:
            req_info_to_response[
                (
                    _TRACKING_HOST_CREDS.host,
                    "/api/2.0/mlflow/runs/get",
                    "GET",
                    message_to_json(GetRun(run_id=run_id)),
                )
            ] = GetRun.Response()

        json_dict = kwargs["json"] if method == "POST" else kwargs["params"]
        response_message = req_info_to_response[
            (host_creds.host, endpoint, method, json.dumps(json_dict, indent=2))
        ]
        mock_resp = mock.MagicMock(autospec=Response)
        mock_resp.status_code = 200
        mock_resp.text = message_to_json(response_message)
        mock_resp.headers = {_DATABRICKS_ORG_ID_HEADER: run_workspace_id}
        return mock_resp

    return request_mock


def _assert_create_model_version_endpoints_called(
    request_mock,
    name,
    source,
    version,
    run_id=None,
    description=None,
    extra_headers=None,
    tags=None,
    model_version_dependencies=None,
):
    """
    Asserts that endpoints related to the model version creation flow were called on the provided
    `request_mock`
    """
    uc_tags = uc_model_version_tag_from_mlflow_tags(tags) if tags is not None else []
    for endpoint, proto_message in [
        (
            "model-versions/create",
            CreateModelVersionRequest(
                name=name,
                source=source,
                run_id=run_id,
                description=description,
                run_tracking_server_id=_get_workspace_id_for_run(run_id),
                tags=uc_tags,
                feature_deps="",
                model_version_dependencies=model_version_dependencies,
            ),
        ),
        (
            "model-versions/generate-temporary-credentials",
            GenerateTemporaryModelVersionCredentialsRequest(
                name=name, version=version, operation=MODEL_VERSION_OPERATION_READ_WRITE
            ),
        ),
        (
            "model-versions/finalize",
            FinalizeModelVersionRequest(name=name, version=version),
        ),
    ]:
        if endpoint == "model-versions/create" and extra_headers is not None:
            _verify_requests(
                http_request=request_mock,
                endpoint=endpoint,
                method="POST",
                proto_message=proto_message,
                extra_headers=extra_headers,
            )
        else:
            _verify_requests(
                http_request=request_mock,
                endpoint=endpoint,
                method="POST",
                proto_message=proto_message,
            )


def test_create_model_version_aws(store, local_model_dir):
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
    source = str(local_model_dir)
    model_name = "model_1"
    version = "1"
    tags = [
        ModelVersionTag(key="key", value="value"),
        ModelVersionTag(key="anotherKey", value="some other value"),
    ]
    mock_artifact_repo = mock.MagicMock(autospec=OptimizedS3ArtifactRepository)
    with (
        mock.patch(
            "mlflow.utils.rest_utils.http_request",
            side_effect=get_request_mock(
                name=model_name,
                version=version,
                temp_credentials=aws_temp_creds,
                storage_location=storage_location,
                source=source,
                tags=tags,
            ),
        ) as request_mock,
        mock.patch(
            "mlflow.store.artifact.optimized_s3_artifact_repo.OptimizedS3ArtifactRepository",
            return_value=mock_artifact_repo,
        ) as optimized_s3_artifact_repo_class_mock,
        mock.patch.dict("sys.modules", {"boto3": {}}),
    ):
        store.create_model_version(name=model_name, source=source, tags=tags)
        # Verify that s3 artifact repo mock was called with expected args
        optimized_s3_artifact_repo_class_mock.assert_called_once_with(
            artifact_uri=storage_location,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            credential_refresh_def=ANY,
            s3_upload_extra_args={},
        )
        mock_artifact_repo.log_artifacts.assert_called_once_with(local_dir=ANY, artifact_path="")
        _assert_create_model_version_endpoints_called(
            request_mock=request_mock, name=model_name, source=source, version=version, tags=tags
        )


def test_create_model_version_local_model_path(store, local_model_dir):
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
    source = "s3://model/version/source"
    model_name = "model_1"
    version = "1"
    tags = [
        ModelVersionTag(key="key", value="value"),
        ModelVersionTag(key="anotherKey", value="some other value"),
    ]
    mock_artifact_repo = mock.MagicMock(autospec=OptimizedS3ArtifactRepository)
    with (
        mock.patch(
            "mlflow.utils.rest_utils.http_request",
            side_effect=get_request_mock(
                name=model_name,
                version=version,
                temp_credentials=aws_temp_creds,
                storage_location=storage_location,
                source=source,
                tags=tags,
            ),
        ) as request_mock,
        mock.patch("mlflow.artifacts.download_artifacts") as mock_download_artifacts,
        mock.patch(
            "mlflow.store.artifact.optimized_s3_artifact_repo.OptimizedS3ArtifactRepository",
            return_value=mock_artifact_repo,
        ),
    ):
        store.create_model_version(
            name=model_name, source=source, tags=tags, local_model_path=local_model_dir
        )
        # Assert that we don't attempt to download model version files, and that we instead log
        # artifacts directly to the destination s3 location from the passed-in local_model_path
        mock_download_artifacts.assert_not_called()
        mock_artifact_repo.log_artifacts.assert_called_once_with(
            local_dir=local_model_dir, artifact_path=""
        )
        _assert_create_model_version_endpoints_called(
            request_mock=request_mock, name=model_name, source=source, version=version, tags=tags
        )


def test_create_model_version_doesnt_redownload_model_from_local_dir(store, local_model_dir):
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
    model_name = "model_1"
    version = "1"
    mock_artifact_repo = mock.MagicMock(autospec=OptimizedS3ArtifactRepository)
    model_dir = str(local_model_dir)
    with (
        mock.patch(
            "mlflow.utils.rest_utils.http_request",
            side_effect=get_request_mock(
                name=model_name,
                version=version,
                temp_credentials=aws_temp_creds,
                storage_location=storage_location,
                source=model_dir,
            ),
        ),
        mock.patch(
            "mlflow.store.artifact.optimized_s3_artifact_repo.OptimizedS3ArtifactRepository",
            return_value=mock_artifact_repo,
        ),
    ):
        # Assert that we create the model version from the local model dir directly,
        # rather than downloading it to a tmpdir + creating from there
        store.create_model_version(name=model_name, source=model_dir)
        mock_artifact_repo.log_artifacts.assert_called_once_with(
            local_dir=model_dir, artifact_path=""
        )


def test_create_model_version_remote_source(store, local_model_dir, tmp_path):
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
    source = "s3://model/version/source"
    model_name = "model_1"
    version = "1"
    mock_artifact_repo = mock.MagicMock(autospec=OptimizedS3ArtifactRepository)
    local_tmpdir = str(tmp_path.joinpath("local_tmpdir"))
    shutil.copytree(local_model_dir, local_tmpdir)
    with (
        mock.patch(
            "mlflow.utils.rest_utils.http_request",
            side_effect=get_request_mock(
                name=model_name,
                version=version,
                temp_credentials=aws_temp_creds,
                storage_location=storage_location,
                source=source,
            ),
        ) as request_mock,
        mock.patch(
            "mlflow.artifacts.download_artifacts",
            return_value=local_tmpdir,
        ) as mock_download_artifacts,
        mock.patch(
            "mlflow.store.artifact.optimized_s3_artifact_repo.OptimizedS3ArtifactRepository",
            return_value=mock_artifact_repo,
        ),
    ):
        store.create_model_version(name=model_name, source=source)
        # Assert that we attempt to download model version files and attempt to log
        # artifacts from the download destination directory
        mock_download_artifacts.assert_called_once_with(
            artifact_uri=source, tracking_uri="databricks"
        )
        mock_artifact_repo.log_artifacts.assert_called_once_with(
            local_dir=local_tmpdir, artifact_path=""
        )
        _assert_create_model_version_endpoints_called(
            request_mock=request_mock,
            name=model_name,
            source=source,
            version=version,
        )
        assert not os.path.exists(local_tmpdir)


def test_create_model_version_azure(store, local_model_dir):
    storage_location = "abfss://filesystem@account.dfs.core.windows.net"
    fake_sas_token = "fake_session_token"
    temporary_creds = TemporaryCredentials(
        azure_user_delegation_sas=AzureUserDelegationSAS(sas_token=fake_sas_token)
    )
    source = str(local_model_dir)
    model_name = "model_1"
    version = "1"
    tags = [
        ModelVersionTag(key="key", value="value"),
        ModelVersionTag(key="anotherKey", value="some other value"),
    ]
    mock_adls_repo = mock.MagicMock(autospec=AzureDataLakeArtifactRepository)
    with (
        mock.patch(
            "mlflow.utils.rest_utils.http_request",
            side_effect=get_request_mock(
                name=model_name,
                version=version,
                temp_credentials=temporary_creds,
                storage_location=storage_location,
                source=source,
                tags=tags,
            ),
        ) as request_mock,
        mock.patch(
            "mlflow.store.artifact.azure_data_lake_artifact_repo.AzureDataLakeArtifactRepository",
            return_value=mock_adls_repo,
        ) as adls_artifact_repo_class_mock,
    ):
        store.create_model_version(name=model_name, source=source, tags=tags)
        adls_artifact_repo_class_mock.assert_called_once_with(
            artifact_uri=storage_location,
            credential=ANY,
            credential_refresh_def=ANY,
        )
        adls_repo_args = adls_artifact_repo_class_mock.call_args_list[0]
        credential = adls_repo_args[1]["credential"]
        assert credential.signature == fake_sas_token
        mock_adls_repo.log_artifacts.assert_called_once_with(local_dir=ANY, artifact_path="")
        _assert_create_model_version_endpoints_called(
            request_mock=request_mock, name=model_name, source=source, version=version, tags=tags
        )


def test_create_model_version_unknown_storage_creds(store, local_model_dir):
    storage_location = "abfss://filesystem@account.dfs.core.windows.net"
    fake_sas_token = "fake_session_token"
    temporary_creds = TemporaryCredentials(
        azure_user_delegation_sas=AzureUserDelegationSAS(sas_token=fake_sas_token)
    )
    unknown_credential_type = "some_new_credential_type"
    source = str(local_model_dir)
    model_name = "model_1"
    version = "1"
    with (
        mock.patch(
            "mlflow.utils.rest_utils.http_request",
            side_effect=get_request_mock(
                name=model_name,
                version=version,
                temp_credentials=temporary_creds,
                storage_location=storage_location,
                source=source,
            ),
        ),
        mock.patch.object(TemporaryCredentials, "WhichOneof", return_value=unknown_credential_type),
        pytest.raises(
            MlflowException,
            match=f"Got unexpected credential type {unknown_credential_type} when "
            "attempting to access model version files",
        ),
    ):
        store.create_model_version(name=model_name, source=source)


@pytest.mark.parametrize(
    "create_args",
    [
        ("name", "source"),
        ("name", "source", "description", "run_id"),
    ],
)
def test_create_model_version_gcp(store, local_model_dir, create_args):
    storage_location = "gs://test_bucket/some/path"
    fake_oauth_token = "fake_session_token"
    temporary_creds = TemporaryCredentials(
        gcp_oauth_token=GcpOauthToken(oauth_token=fake_oauth_token)
    )
    source = str(local_model_dir)
    model_name = "model_1"
    all_create_args = {
        "name": model_name,
        "source": source,
        "description": "my_description",
        "run_id": "some_run_id",
        "tags": [
            ModelVersionTag(key="key", value="value"),
            ModelVersionTag(key="anotherKey", value="some other value"),
        ],
    }
    create_kwargs = {key: value for key, value in all_create_args.items() if key in create_args}
    mock_gcs_repo = mock.MagicMock(autospec=GCSArtifactRepository)
    version = "1"
    mock_request_fn = get_request_mock(
        **create_kwargs,
        version=version,
        temp_credentials=temporary_creds,
        storage_location=storage_location,
    )
    get_run_and_headers_retval = None, None
    if "run_id" in create_kwargs:
        test_notebook_tag = RunTag(key=MLFLOW_DATABRICKS_NOTEBOOK_ID, value="321")
        test_job_tag = RunTag(key=MLFLOW_DATABRICKS_JOB_ID, value="456")
        test_job_run_tag = RunTag(key=MLFLOW_DATABRICKS_JOB_RUN_ID, value="789")
        test_run_data = RunData(tags=[test_notebook_tag, test_job_tag, test_job_run_tag])
        test_run_info = RunInfo(
            "run_uuid",
            "experiment_id",
            "user_id",
            "status",
            "start_time",
            "end_time",
            "lifecycle_stage",
        )
        test_run = Run(run_data=test_run_data, run_info=test_run_info)
        get_run_and_headers_retval = ({_DATABRICKS_ORG_ID_HEADER: "123"}, test_run)
    with (
        mock.patch(
            "mlflow.store._unity_catalog.registry.rest_store.http_request",
            side_effect=mock_request_fn,
        ),
        mock.patch(
            "mlflow.store._unity_catalog.registry.rest_store.UcModelRegistryStore._get_run_and_headers",
            # Set the headers and Run retvals when the run_id is set
            return_value=get_run_and_headers_retval,
        ),
        mock.patch(
            "mlflow.utils.rest_utils.http_request",
            side_effect=mock_request_fn,
        ) as request_mock,
        mock.patch(
            "google.cloud.storage.Client", return_value=mock.MagicMock(autospec=Client)
        ) as gcs_client_class_mock,
        mock.patch(
            "mlflow.store.artifact.gcs_artifact_repo.GCSArtifactRepository",
            return_value=mock_gcs_repo,
        ) as gcs_artifact_repo_class_mock,
    ):
        store.create_model_version(**create_kwargs)
        # Verify that gcs artifact repo mock was called with expected args
        gcs_artifact_repo_class_mock.assert_called_once_with(
            artifact_uri=storage_location, client=ANY, credential_refresh_def=ANY
        )
        mock_gcs_repo.log_artifacts.assert_called_once_with(local_dir=ANY, artifact_path="")
        gcs_client_args = gcs_client_class_mock.call_args_list[0]
        credentials = gcs_client_args[1]["credentials"]
        assert credentials.token == fake_oauth_token
        if "run_id" in create_kwargs:
            _, run = store._get_run_and_headers("some_run_id")
            notebook_id = store._get_notebook_id(run)
            job_id = store._get_job_id(run)
            job_run_id = store._get_job_run_id(run)
            notebook_entity = Notebook(id=str(notebook_id))
            job_entity = Job(id=str(job_id), job_run_id=str(job_run_id))
            notebook_entity = Entity(notebook=notebook_entity)
            job_entity = Entity(job=job_entity)
            lineage_header_info = LineageHeaderInfo(entities=[notebook_entity, job_entity])
            expected_lineage_json = message_to_json(lineage_header_info)
            expected_lineage_header = base64.b64encode(expected_lineage_json.encode())
            assert expected_lineage_header.isascii()
            create_kwargs["extra_headers"] = {
                _DATABRICKS_LINEAGE_ID_HEADER: expected_lineage_header,
            }
        _assert_create_model_version_endpoints_called(
            request_mock=request_mock, version=version, **create_kwargs
        )


def test_local_model_dir_preserves_uc_volumes_path(tmp_path):
    store = UcModelRegistryStore(store_uri="databricks-uc", tracking_uri="databricks-uc")
    with (
        mock.patch(
            "mlflow.artifacts.download_artifacts", return_value=str(tmp_path)
        ) as mock_download_artifacts,
        mock.patch(
            # Pretend that `tmp_path` is a UC Volumes path
            "mlflow.store._unity_catalog.registry.rest_store.is_fuse_or_uc_volumes_uri",
            return_value=True,
        ) as mock_is_fuse_or_uc_volumes_uri,
    ):
        with store._local_model_dir(source=f"dbfs:{tmp_path}", local_model_path=None):
            pass
        mock_download_artifacts.assert_called_once()
        mock_is_fuse_or_uc_volumes_uri.assert_called_once()
        assert tmp_path.exists()


@pytest.mark.parametrize(
    ("num_inputs", "expected_truncation_size"),
    [
        (1, 1),
        (10, 10),
        (11, 10),
    ],
)
def test_input_source_truncation(num_inputs, expected_truncation_size, store):
    test_notebook_tag = RunTag(key=MLFLOW_DATABRICKS_NOTEBOOK_ID, value="321")
    test_job_tag = RunTag(key=MLFLOW_DATABRICKS_JOB_ID, value="456")
    test_job_run_tag = RunTag(key=MLFLOW_DATABRICKS_JOB_RUN_ID, value="789")
    test_run_data = RunData(tags=[test_notebook_tag, test_job_tag, test_job_run_tag])
    test_run_info = RunInfo(
        "run_uuid",
        "experiment_id",
        "user_id",
        "status",
        "start_time",
        "end_time",
        "lifecycle_stage",
    )
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    df = pd.DataFrame([1, 2, 3], columns=["Numbers"])
    input_list = []
    for count in range(num_inputs):
        input_list.append(
            Dataset(
                source=DeltaDatasetSource(
                    delta_table_name=f"temp_delta_versioned_with_id_{count}",
                    delta_table_version=1,
                    delta_table_id=f"uc_id_{count}",
                )
            )
        )
        # Let's double up the sources and verify non-Delta Datasets are filtered out
        input_list.append(
            Dataset(
                source=PandasDataset(
                    df=df,
                    source=source,
                    name=f"testname_{count}",
                )
            )
        )
    assert len(input_list) == num_inputs * 2
    test_run_inputs = RunInputs(dataset_inputs=input_list)
    test_run = Run(run_data=test_run_data, run_info=test_run_info, run_inputs=test_run_inputs)
    filtered_inputs = store._get_lineage_input_sources(test_run)
    assert len(filtered_inputs) == expected_truncation_size


def test_create_model_version_unsupported_fields(store):
    with pytest.raises(MlflowException, match=_expected_unsupported_arg_error_message("run_link")):
        store.create_model_version(name="mymodel", source="mysource", run_link="https://google.com")


def test_transition_model_version_stage_unsupported(store):
    name = "model_1"
    version = "5"
    expected_error = (
        f"{_expected_unsupported_method_error_message('transition_model_version_stage')}. "
        f"We recommend using aliases instead of stages for more flexible model deployment "
        f"management."
    )
    with pytest.raises(
        MlflowException,
        match=expected_error,
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


@mock_http_200
def test_search_model_versions_with_pagination(mock_http, store):
    store.search_model_versions(
        filter_string="name='model_12'", page_token="fake_page_token", max_results=123
    )
    _verify_requests(
        mock_http,
        "model-versions/search",
        "GET",
        SearchModelVersionsRequest(
            filter="name='model_12'", page_token="fake_page_token", max_results=123
        ),
    )


def test_search_model_versions_order_by_unsupported(store):
    with pytest.raises(MlflowException, match=_expected_unsupported_arg_error_message("order_by")):
        store.search_model_versions(
            filter_string="name='model_12'", page_token="fake_page_token", order_by=["name ASC"]
        )


@mock_http_200
def test_set_model_version_tag(mock_http, store):
    name = "model_1"
    tag = ModelVersionTag(key="key", value="value")
    store.set_model_version_tag(name=name, version="1", tag=tag)
    _verify_requests(
        mock_http,
        "model-versions/set-tag",
        "POST",
        SetModelVersionTagRequest(name=name, version="1", key=tag.key, value=tag.value),
    )


@mock_http_200
def test_delete_model_version_tag(mock_http, store):
    name = "model_1"
    store.delete_model_version_tag(name=name, version="1", key="key")
    _verify_requests(
        mock_http,
        "model-versions/delete-tag",
        "DELETE",
        DeleteModelVersionTagRequest(name=name, version="1", key="key"),
    )


@mock_http_200
@pytest.mark.parametrize("tags", [None, []])
def test_default_values_for_tags(store, tags):
    # No unsupported arg exceptions should be thrown
    store.create_registered_model(name="model_1", description="description", tags=tags)
    store.create_model_version(name="mymodel", source="source")


@mock_http_200
def test_set_registered_model_alias(mock_http, store):
    name = "model_1"
    alias = "test_alias"
    version = "1"
    store.set_registered_model_alias(name=name, alias=alias, version=version)
    _verify_requests(
        mock_http,
        "registered-models/alias",
        "POST",
        SetRegisteredModelAliasRequest(name=name, alias=alias, version=version),
    )


@mock_http_200
def test_delete_registered_model_alias(mock_http, store):
    name = "model_1"
    alias = "test_alias"
    store.delete_registered_model_alias(name=name, alias=alias)
    _verify_requests(
        mock_http,
        "registered-models/alias",
        "DELETE",
        DeleteRegisteredModelAliasRequest(name=name, alias=alias),
    )


@mock_http_200
def test_get_model_version_by_alias(mock_http, store):
    name = "model_1"
    alias = "test_alias"
    store.get_model_version_by_alias(name=name, alias=alias)
    _verify_requests(
        mock_http,
        "registered-models/alias",
        "GET",
        GetModelVersionByAliasRequest(name=name, alias=alias),
    )


@mock_http_200
@pytest.mark.parametrize("spark_session", ["main"], indirect=True)  # set the catalog name to "main"
def test_store_uses_catalog_and_schema_from_spark_session(mock_http, spark_session, store):
    name = "model_1"
    full_name = "main.default.model_1"
    store.get_registered_model(name=name)
    spark_session.sql.assert_any_call(_ACTIVE_CATALOG_QUERY)
    spark_session.sql.assert_any_call(_ACTIVE_SCHEMA_QUERY)
    assert spark_session.sql.call_count == 2
    _verify_requests(
        mock_http, "registered-models/get", "GET", GetRegisteredModelRequest(name=full_name)
    )


@mock_http_200
@pytest.mark.parametrize("spark_session", ["main"], indirect=True)
def test_store_uses_catalog_from_spark_session(mock_http, spark_session, store):
    name = "default.model_1"
    full_name = "main.default.model_1"
    store.get_registered_model(name=name)
    spark_session.sql.assert_any_call(_ACTIVE_CATALOG_QUERY)
    assert spark_session.sql.call_count == 1
    _verify_requests(
        mock_http, "registered-models/get", "GET", GetRegisteredModelRequest(name=full_name)
    )


@mock_http_200
@pytest.mark.parametrize("spark_session", ["hive_metastore", "spark_catalog"], indirect=True)
def test_store_ignores_hive_metastore_default_from_spark_session(mock_http, spark_session, store):
    name = "model_1"
    store.get_registered_model(name=name)
    spark_session.sql.assert_any_call(_ACTIVE_CATALOG_QUERY)
    assert spark_session.sql.call_count == 1
    _verify_requests(
        mock_http, "registered-models/get", "GET", GetRegisteredModelRequest(name=name)
    )


def test_store_use_presigned_url_store_when_disabled(monkeypatch):
    store_package = "mlflow.store._unity_catalog.registry.rest_store"
    monkeypatch.setenv("MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC", "false")
    monkeypatch.setenv("DATABRICKS_HOST", "my-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "my-token")

    uc_store = UcModelRegistryStore(store_uri="databricks-uc", tracking_uri="databricks-uc")
    model_version = ModelVersion(
        name="catalog.schema.model_1", version="1", storage_location="s3://some/storage/location"
    )
    creds = TemporaryCredentials(
        aws_temp_credentials=AwsCredentials(
            access_key_id="key", secret_access_key="secret", session_token="token"
        )
    )
    with (
        mock.patch(
            f"{store_package}.UcModelRegistryStore._get_temporary_model_version_write_credentials",
            return_value=creds,
        ) as temp_cred_mock,
        mock.patch(
            f"{store_package}.get_artifact_repo_from_storage_info",
            side_effect=get_artifact_repo_from_storage_info,
        ) as get_repo_mock,
    ):
        aws_store = uc_store._get_artifact_repo(model_version)

        assert type(aws_store) is OptimizedS3ArtifactRepository
        temp_cred_mock.assert_called_once_with(
            name=model_version.name, version=model_version.version
        )
        get_repo_mock.assert_called_once_with(
            storage_location=model_version.storage_location,
            scoped_token=creds,
            base_credential_refresh_def=ANY,
        )


def test_store_use_presigned_url_store_when_enabled(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "my-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "my-token")
    monkeypatch.setenv("MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC", "false")
    store_package = "mlflow.store._unity_catalog.registry.rest_store"
    creds = TemporaryCredentials(storage_mode=StorageMode.DEFAULT_STORAGE)
    with mock.patch(
        f"{store_package}.UcModelRegistryStore._get_temporary_model_version_write_credentials",
        return_value=creds,
    ):
        uc_store = UcModelRegistryStore(store_uri="databricks-uc", tracking_uri="databricks-uc")
        model_version = ModelVersion(name="catalog.schema.model_1", version="1")
        presigned_store = uc_store._get_artifact_repo(model_version)

    assert type(presigned_store) is PresignedUrlArtifactRepository


@mock_http_200
def test_create_and_update_registered_model_print_job_url(mock_http, store):
    name = "model_for_job_url_test"
    description = "test model with job id"
    deployment_job_id = "123"

    with (
        mock.patch(
            "mlflow.store._unity_catalog.registry.rest_store._print_databricks_deployment_job_url",
        ) as mock_print_url,
    ):
        # Should not print the job url when the deployment job id is None
        store.create_registered_model(name=name, description=description, deployment_job_id=None)
        store.create_registered_model(name=name, description=description, deployment_job_id="")
        mock_print_url.assert_not_called()
        # Should print the job url when the deployment job id is not None
        store.create_registered_model(
            name=name, description=description, deployment_job_id=deployment_job_id
        )
        mock_print_url.assert_called_once_with(model_name=name, job_id=deployment_job_id)
        mock_print_url.reset_mock()

        # Should not print the job url when the deployment job id is false-y
        store.update_registered_model(name=name, description=description, deployment_job_id=None)
        store.update_registered_model(name=name, description=description, deployment_job_id="")
        mock_print_url.assert_not_called()
        # Should print the job url when the deployment job id is not None
        store.update_registered_model(
            name=name, description=description, deployment_job_id=deployment_job_id
        )
        mock_print_url.assert_called_once_with(model_name=name, job_id=deployment_job_id)


@mock_http_200
def test_create_prompt_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    description = "A test prompt"
    tags = {"foo": "bar"}
    # Patch proto_info_to_mlflow_prompt_info to return a dummy Prompt
    with mock.patch(
        "mlflow.store._unity_catalog.registry.rest_store.proto_info_to_mlflow_prompt_info",
        return_value=Prompt(name=name, description=description, tags=tags),
    ) as proto_to_prompt:
        store.create_prompt(name=name, description=description, tags=tags)
        # Check that the endpoint was called correctly
        assert any(c[1]["endpoint"].endswith("/prompts") for c in mock_http.call_args_list)
        proto_to_prompt.assert_called()


@mock_http_200
def test_search_prompts_uc(mock_http, store, monkeypatch):
    # Patch proto_info_to_mlflow_prompt_info to return a dummy Prompt
    with mock.patch(
        "mlflow.store._unity_catalog.registry.rest_store.proto_info_to_mlflow_prompt_info",
        return_value=Prompt(name="prompt1", description="test prompt"),
    ) as proto_to_prompt:
        store.search_prompts(filter_string="catalog = 'test_catalog' AND schema = 'test_schema'")
        # Should call the correct endpoint for SearchPromptsRequest
        assert any("/prompts" in c[1]["endpoint"] for c in mock_http.call_args_list)
        # The utility function should NOT be called when there are no results (empty list)
        assert proto_to_prompt.call_count == 0  # Correct behavior for empty results


def test_search_prompts_with_results_uc(store, monkeypatch):
    # Create mock protobuf objects
    from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
        Prompt as ProtoPrompt,
    )
    from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
        PromptTag as ProtoPromptTag,
    )
    from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
        SearchPromptsResponse,
    )

    # Create mock prompt data
    mock_prompt_1 = ProtoPrompt(
        name="test_prompt_1",
        description="First test prompt",
        tags=[ProtoPromptTag(key="env", value="dev")],
    )
    mock_prompt_2 = ProtoPrompt(name="test_prompt_2", description="Second test prompt")

    # Create mock response
    mock_response = SearchPromptsResponse(
        prompts=[mock_prompt_1, mock_prompt_2], next_page_token="next_token_123"
    )

    # Expected conversion results
    expected_prompts = [
        Prompt(name="test_prompt_1", description="First test prompt", tags={"env": "dev"}),
        Prompt(name="test_prompt_2", description="Second test prompt", tags={}),
    ]

    with (
        mock.patch.object(store, "_call_endpoint", return_value=mock_response),
        mock.patch(
            "mlflow.store._unity_catalog.registry.rest_store.proto_info_to_mlflow_prompt_info",
            side_effect=expected_prompts,
        ) as mock_converter,
    ):
        # Call search_prompts
        result = store.search_prompts(
            max_results=10, filter_string="catalog = 'test_catalog' AND schema = 'test_schema'"
        )

        # Verify conversion function was called twice (once for each result)
        assert mock_converter.call_count == 2

        # Verify the results
        assert len(result) == 2
        assert result.token == "next_token_123"
        assert result[0].name == "test_prompt_1"
        assert result[1].name == "test_prompt_2"


@mock_http_200
def test_delete_prompt_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    store.delete_prompt(name=name)
    # Should call the correct endpoint for DeletePromptRequest
    assert any("/prompts" in c[1]["endpoint"] for c in mock_http.call_args_list)


@mock_http_200
def test_set_prompt_tag_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    key = "env"
    value = "prod"
    store.set_prompt_tag(name=name, key=key, value=value)
    # Should call the exact endpoint for SetPromptTagRequest with substituted path
    expected_endpoint = f"/api/2.0/mlflow/unity-catalog/prompts/{name}/tags"
    assert any(c[1]["endpoint"] == expected_endpoint for c in mock_http.call_args_list)


@mock_http_200
def test_delete_prompt_tag_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    key = "env"
    store.delete_prompt_tag(name=name, key=key)
    # Should call the exact endpoint for DeletePromptTagRequest with substituted path
    expected_endpoint = f"/api/2.0/mlflow/unity-catalog/prompts/{name}/tags/{key}"
    assert any(c[1]["endpoint"] == expected_endpoint for c in mock_http.call_args_list)


@mock_http_200
def test_create_prompt_version_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    template = "Hello {name}!"
    description = "A greeting prompt"
    tags = {"env": "test"}
    # Patch proto_to_mlflow_prompt to return a dummy PromptVersion
    with mock.patch(
        "mlflow.store._unity_catalog.registry.rest_store.proto_to_mlflow_prompt",
        return_value=PromptVersion(
            name=name, version=1, template=template, commit_message=description, tags=tags
        ),
    ) as proto_to_prompt:
        store.create_prompt_version(
            name=name, template=template, description=description, tags=tags
        )
        # Should call the correct endpoint for CreatePromptVersionRequest
        assert any(
            "/prompts/" in c[1]["endpoint"] and "/versions" in c[1]["endpoint"]
            for c in mock_http.call_args_list
        )
        proto_to_prompt.assert_called()


@mock_http_200
def test_create_prompt_version_with_response_format_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    template = "Generate a response for {query}"
    description = "A response generation prompt"
    tags = {"env": "test"}
    response_format = {"type": "object", "properties": {"answer": {"type": "string"}}}

    # Patch proto_to_mlflow_prompt to return a dummy PromptVersion
    with mock.patch(
        "mlflow.store._unity_catalog.registry.rest_store.proto_to_mlflow_prompt",
        return_value=PromptVersion(
            name=name,
            version=1,
            template=template,
            commit_message=description,
            tags=tags,
            response_format=response_format,
        ),
    ) as proto_to_prompt:
        store.create_prompt_version(
            name=name,
            template=template,
            description=description,
            tags=tags,
            response_format=response_format,
        )

    # Verify the correct endpoint is called
    assert any(
        "/prompts/" in c[1]["endpoint"] and "/versions" in c[1]["endpoint"]
        for c in mock_http.call_args_list
    )
    proto_to_prompt.assert_called()

    # Verify the HTTP request body contains the response_format in tags
    http_call_args = [
        c
        for c in mock_http.call_args_list
        if "/prompts/" in c[1]["endpoint"] and "/versions" in c[1]["endpoint"]
    ]
    assert len(http_call_args) == 1

    request_body = http_call_args[0][1]["json"]
    prompt_version = request_body["prompt_version"]

    tags_in_request = {tag["key"]: tag["value"] for tag in prompt_version.get("tags", [])}
    assert RESPONSE_FORMAT_TAG_KEY in tags_in_request

    expected_response_format = json.dumps(response_format)
    assert tags_in_request[RESPONSE_FORMAT_TAG_KEY] == expected_response_format
    assert tags_in_request["env"] == "test"
    assert tags_in_request[PROMPT_TYPE_TAG_KEY] == PROMPT_TYPE_TEXT


@mock_http_200
def test_create_prompt_version_with_multi_turn_template_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, can you help me with {task}?"},
        {"role": "assistant", "content": "Of course! I'd be happy to help you with {task}."},
        {"role": "user", "content": "Please provide a detailed explanation."},
    ]
    description = "A multi-turn conversation prompt"
    tags = {"type": "conversation", "env": "test"}

    # Patch proto_to_mlflow_prompt to return a dummy PromptVersion
    with mock.patch(
        "mlflow.store._unity_catalog.registry.rest_store.proto_to_mlflow_prompt",
        return_value=PromptVersion(
            name=name, version=1, template=template, commit_message=description, tags=tags
        ),
    ) as proto_to_prompt:
        store.create_prompt_version(
            name=name, template=template, description=description, tags=tags
        )

    # Verify the correct endpoint is called
    assert any(
        "/prompts/" in c[1]["endpoint"] and "/versions" in c[1]["endpoint"]
        for c in mock_http.call_args_list
    )
    proto_to_prompt.assert_called()

    # Verify the HTTP request body contains the multi-turn template
    http_call_args = [
        c
        for c in mock_http.call_args_list
        if "/prompts/" in c[1]["endpoint"] and "/versions" in c[1]["endpoint"]
    ]
    assert len(http_call_args) == 1

    request_body = http_call_args[0][1]["json"]
    prompt_version = request_body["prompt_version"]

    # Verify template was JSON-encoded properly
    template_in_request = json.loads(prompt_version["template"])
    assert template_in_request == template

    tags_in_request = {tag["key"]: tag["value"] for tag in prompt_version.get("tags", [])}
    assert tags_in_request["type"] == "conversation"
    assert tags_in_request["env"] == "test"
    assert tags_in_request[PROMPT_TYPE_TAG_KEY] == PROMPT_TYPE_CHAT


@mock_http_200
def test_get_prompt_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    # Patch proto_info_to_mlflow_prompt_info to return a dummy Prompt
    with mock.patch(
        "mlflow.store._unity_catalog.registry.rest_store.proto_info_to_mlflow_prompt_info",
        return_value=Prompt(name=name, description="test prompt", tags={}),
    ) as proto_to_prompt:
        store.get_prompt(name=name)
        # Should call the correct endpoint for GetPromptRequest
        assert any(
            "/prompts/" in c[1]["endpoint"] and "/versions/" not in c[1]["endpoint"]
            for c in mock_http.call_args_list
        )
        proto_to_prompt.assert_called()


@mock_http_200
def test_get_prompt_version_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    version = "1"
    # Patch proto_to_mlflow_prompt to return a dummy PromptVersion
    with mock.patch(
        "mlflow.store._unity_catalog.registry.rest_store.proto_to_mlflow_prompt",
        return_value=PromptVersion(name=name, version=1, template="Hello {name}!"),
    ) as proto_to_prompt:
        store.get_prompt_version(name=name, version=version)
        # Should call the correct endpoint for GetPromptVersionRequest
        assert any(
            "/prompts/" in c[1]["endpoint"] and "/versions/" in c[1]["endpoint"]
            for c in mock_http.call_args_list
        )
        proto_to_prompt.assert_called()


@mock_http_200
def test_delete_prompt_version_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    version = "1"
    store.delete_prompt_version(name=name, version=version)
    # Should call the correct endpoint for DeletePromptVersionRequest
    assert any(
        "/prompts/" in c[1]["endpoint"] and "/versions/" in c[1]["endpoint"]
        for c in mock_http.call_args_list
    )


@mock_http_200
def test_get_prompt_version_by_alias_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    alias = "latest"
    # Patch proto_to_mlflow_prompt to return a dummy PromptVersion
    with mock.patch(
        "mlflow.store._unity_catalog.registry.rest_store.proto_to_mlflow_prompt",
        return_value=PromptVersion(name=name, version=1, template="Hello {name}!"),
    ) as proto_to_prompt:
        store.get_prompt_version_by_alias(name=name, alias=alias)
        # Should call the correct endpoint for GetPromptVersionByAliasRequest
        assert any(
            "/prompts/" in c[1]["endpoint"] and "/versions/by-alias/" in c[1]["endpoint"]
            for c in mock_http.call_args_list
        )
        proto_to_prompt.assert_called()


def test_link_prompt_version_to_model_success(store):
    """Test successful Unity Catalog linking with API call."""

    with (
        mock.patch.object(store, "_edit_endpoint_and_call") as mock_edit_call,
        mock.patch.object(store, "_get_endpoint_from_method") as mock_get_endpoint,
        mock.patch(
            "mlflow.store.model_registry.abstract_store.AbstractStore.link_prompt_version_to_model"
        ) as mock_super_call,
    ):
        # Setup
        mock_get_endpoint.return_value = (
            "/api/2.0/mlflow/unity-catalog/prompts/link-to-model",
            "POST",
        )

        # Execute
        store.link_prompt_version_to_model("test_prompt", "1", "model_123")

        # Verify parent method was called
        mock_super_call.assert_called_once_with(
            name="test_prompt", version="1", model_id="model_123"
        )

        # Verify API call was made
        mock_edit_call.assert_called_once()
        call_args = mock_edit_call.call_args

        assert call_args[1]["name"] == "test_prompt"
        assert call_args[1]["version"] == "1"
        assert call_args[1]["model_id"] == "model_123"
        assert call_args[1]["proto_name"] == LinkPromptVersionsToModelsRequest


@mock.patch("mlflow.tracking._get_store")
def test_link_prompt_version_to_model_sets_tag(mock_get_tracking_store, store):
    """Test that linking a prompt version to a model sets the appropriate tag."""

    # Setup mocks
    mock_tracking_store = mock.Mock()
    mock_get_tracking_store.return_value = mock_tracking_store

    # Mock the prompt version
    mock_prompt_version = PromptVersion(
        name="test_prompt",
        version=1,
        template="Test template",
        creation_timestamp=1234567890,
    )

    with mock.patch.object(store, "get_prompt_version", return_value=mock_prompt_version):
        # Mock the logged model
        model_id = "model_123"
        logged_model = LoggedModel(
            experiment_id="exp_123",
            model_id=model_id,
            name="test_model",
            artifact_location="/path/to/artifacts",
            creation_timestamp=1234567890,
            last_updated_timestamp=1234567890,
            tags={},
        )
        mock_tracking_store.get_logged_model.return_value = logged_model

        # Mock the UC-specific API call to avoid real API calls
        with mock.patch.object(store, "_edit_endpoint_and_call"):
            with mock.patch.object(
                store, "_get_endpoint_from_method", return_value=("/api/test", "POST")
            ):
                # Execute
                store.link_prompt_version_to_model("test_prompt", "1", model_id)

        # Verify the tag was set
        mock_tracking_store.set_logged_model_tags.assert_called_once()
        call_args = mock_tracking_store.set_logged_model_tags.call_args
        assert call_args[0][0] == model_id

        logged_model_tags = call_args[0][1]
        assert len(logged_model_tags) == 1
        logged_model_tag = logged_model_tags[0]
        assert isinstance(logged_model_tag, LoggedModelTag)
        assert logged_model_tag.key == LINKED_PROMPTS_TAG_KEY

        expected_value = [{"name": "test_prompt", "version": "1"}]
        assert json.loads(logged_model_tag.value) == expected_value


def test_link_prompts_to_trace_success(store):
    """Test successful Unity Catalog linking prompts to a trace with API call."""

    with (
        mock.patch.object(store, "_edit_endpoint_and_call") as mock_edit_call,
        mock.patch.object(store, "_get_endpoint_from_method") as mock_get_endpoint,
        mock.patch(
            "mlflow.store.model_registry.abstract_store.AbstractStore.link_prompts_to_trace"
        ) as mock_super_call,
    ):
        # Setup
        mock_get_endpoint.return_value = (
            "/api/2.0/mlflow/unity-catalog/prompt-versions/links-to-traces",
            "POST",
        )

        prompt_versions = [
            PromptVersion(name="test_prompt", version=1, template="test", creation_timestamp=123)
        ]
        trace_id = "trace_123"

        # Execute
        store.link_prompts_to_trace(prompt_versions, trace_id)

        # Verify parent method was called
        mock_super_call.assert_called_once_with(prompt_versions=prompt_versions, trace_id=trace_id)

        # Verify API call was made
        mock_edit_call.assert_called_once()
        call_args = mock_edit_call.call_args

        assert call_args[1]["proto_name"] == LinkPromptsToTracesRequest


def test_link_prompt_version_to_run_success(store):
    """Test successful Unity Catalog linking prompt version to run with API call."""

    with (
        mock.patch.object(store, "_edit_endpoint_and_call") as mock_edit_call,
        mock.patch.object(store, "_get_endpoint_from_method") as mock_get_endpoint,
        mock.patch(
            "mlflow.store.model_registry.abstract_store.AbstractStore.link_prompt_version_to_run"
        ) as mock_super_call,
    ):
        # Setup
        mock_get_endpoint.return_value = (
            "/api/2.0/mlflow/unity-catalog/prompt-versions/links-to-runs",
            "POST",
        )

        # Execute
        store.link_prompt_version_to_run("test_prompt", "1", "run_123")

        # Verify parent method was called
        mock_super_call.assert_called_once_with(name="test_prompt", version="1", run_id="run_123")

        # Verify API call was made
        mock_edit_call.assert_called_once()
        call_args = mock_edit_call.call_args

        # Check that _edit_endpoint_and_call was called with correct parameters
        assert (
            call_args[1]["endpoint"]
            == "/api/2.0/mlflow/unity-catalog/prompt-versions/links-to-runs"
        )
        assert call_args[1]["method"] == "POST"
        assert call_args[1]["proto_name"] == LinkPromptVersionsToRunsRequest

        # Verify the request body contains correct prompt and run information
        req_body = json.loads(call_args[1]["req_body"])
        assert len(req_body["prompt_versions"]) == 1
        assert req_body["prompt_versions"][0]["name"] == "test_prompt"
        assert req_body["prompt_versions"][0]["version"] == "1"
        assert req_body["run_ids"] == ["run_123"]


@mock.patch("mlflow.tracking._get_store")
def test_link_prompt_version_to_run_sets_tag(mock_get_tracking_store, store):
    """Test that linking a prompt version to a run sets the appropriate tag."""

    # Setup mocks
    mock_tracking_store = mock.Mock()
    mock_get_tracking_store.return_value = mock_tracking_store

    # Mock the prompt version
    mock_prompt_version = PromptVersion(
        name="test_prompt",
        version=1,
        template="Test template",
        creation_timestamp=1234567890,
    )

    with mock.patch.object(store, "get_prompt_version", return_value=mock_prompt_version):
        # Mock the run
        run_id = "run_123"
        run_data = RunData(metrics=[], params=[], tags={})
        run_info = RunInfo(
            run_id=run_id,
            experiment_id="exp_123",
            user_id="user_123",
            status="FINISHED",
            start_time=1234567890,
            end_time=1234567890,
            lifecycle_stage="active",
        )
        run = Run(run_info=run_info, run_data=run_data)
        mock_tracking_store.get_run.return_value = run

        # Mock the UC-specific API call to avoid real API calls
        with mock.patch.object(store, "_edit_endpoint_and_call"):
            with mock.patch.object(
                store, "_get_endpoint_from_method", return_value=("/api/test", "POST")
            ):
                # Execute
                store.link_prompt_version_to_run("test_prompt", "1", run_id)

        # Verify the tag was set
        mock_tracking_store.set_tag.assert_called_once()
        call_args = mock_tracking_store.set_tag.call_args
        assert call_args[0][0] == run_id

        run_tag = call_args[0][1]
        assert isinstance(run_tag, RunTag)
        assert run_tag.key == LINKED_PROMPTS_TAG_KEY

        expected_value = [{"name": "test_prompt", "version": "1"}]
        assert json.loads(run_tag.value) == expected_value
