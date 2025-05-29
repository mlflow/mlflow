import json
from unittest import mock

import pytest

from mlflow.protos.unity_catalog_prompt_service_pb2 import (
    CreatePromptVersionRequest,
    CreatePromptVersionResponse,
    DeletePromptVersionRequest,
    DeletePromptVersionResponse,
    GetPromptVersionRequest,
    GetPromptVersionResponse,
    SearchPromptVersionsRequest,
    SearchPromptVersionsResponse,
    PromptVersionInfo,
    PromptTag,
)
from mlflow.store._unity_catalog.registry.uc_prompt_rest_store import UcPromptRestStore
from mlflow.utils.proto_json_utils import message_to_json

from tests.helper_functions import mock_http_200
from tests.store._unity_catalog.conftest import _REGISTRY_HOST_CREDS


@pytest.fixture
def store(mock_databricks_uc_host_creds):
    with mock.patch("mlflow.utils.databricks_utils.get_databricks_host_creds"):
        yield UcPromptRestStore(store_uri="databricks-uc")


def _verify_requests(mock_http, endpoint, method, proto_message):
    """Helper to verify REST requests."""
    if method == "GET":
        assert mock_http.call_args[0][0] == endpoint
        assert mock_http.call_args[0][1] == method
    else:
        assert mock_http.call_args[0][0] == endpoint
        assert mock_http.call_args[0][1] == method
        assert json.loads(mock_http.call_args[0][2]) == message_to_json(proto_message)


@mock_http_200
def test_create_prompt_version(mock_http, store):
    """Test creating a prompt version."""
    name = "catalog_1.schema_1.prompt_1"
    template = "This is version {{version}} of the template"
    description = "Test prompt version"
    tags = [{"key": "test_key", "value": "test_value"}]
    
    store.create_prompt_version(
        name=name,
        template=template,
        description=description,
        tags=tags,
    )
    
    _verify_requests(
        mock_http,
        f"prompts/{name}/versions",
        "POST",
        CreatePromptVersionRequest(
            name=name,
            template=template,
            description=description,
            tags=[PromptTag(key="test_key", value="test_value")],
        ),
    )


@mock_http_200
def test_get_prompt_version(mock_http, store):
    """Test getting a prompt version."""
    name = "catalog_1.schema_1.prompt_1"
    version = "1"
    
    store.get_prompt_version(name=name, version=version)
    
    _verify_requests(
        mock_http,
        f"prompts/{name}/versions/{version}",
        "GET",
        GetPromptVersionRequest(name=name, version=version),
    )


@mock_http_200
def test_delete_prompt_version(mock_http, store):
    """Test deleting a prompt version."""
    name = "catalog_1.schema_1.prompt_1"
    version = "1"
    
    store.delete_prompt_version(name=name, version=version)
    
    _verify_requests(
        mock_http,
        f"prompts/{name}/versions/{version}",
        "DELETE",
        DeletePromptVersionRequest(name=name, version=version),
    )


@mock_http_200
def test_search_prompt_versions(mock_http, store):
    """Test searching prompt versions."""
    name = "catalog_1.schema_1.prompt_1"
    filter_string = "version > '1'"
    max_results = 10
    page_token = "test_token"
    
    store.search_prompt_versions(
        name=name,
        filter_string=filter_string,
        max_results=max_results,
        page_token=page_token,
    )
    
    _verify_requests(
        mock_http,
        f"prompts/{name}/versions",
        "GET",
        SearchPromptVersionsRequest(
            name=name,
            filter=filter_string,
            max_results=max_results,
            page_token=page_token,
        ),
    )


@mock_http_200
def test_update_prompt_version(mock_http, store):
    """Test updating a prompt version."""
    name = "catalog_1.schema_1.prompt_1"
    version = "1"
    description = "Updated description"
    tags = [{"key": "updated_key", "value": "updated_value"}]
    
    store.update_prompt_version(
        name=name,
        version=version,
        description=description,
        tags=tags,
    )
    
    _verify_requests(
        mock_http,
        f"prompts/{name}/versions/{version}",
        "PATCH",
        UpdatePromptVersionRequest(
            name=name,
            version=version,
            description=description,
            tags=[PromptTag(key="updated_key", value="updated_value")],
        ),
    ) 