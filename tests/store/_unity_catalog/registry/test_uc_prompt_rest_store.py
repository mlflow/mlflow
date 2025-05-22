import json
from unittest import mock

import pytest

from mlflow.protos.unity_catalog_prompt_service_pb2 import (
    CreatePromptRequest,
    CreatePromptResponse,
    DeletePromptRequest,
    DeletePromptResponse,
    GetPromptRequest,
    GetPromptResponse,
    SearchPromptsRequest,
    SearchPromptsResponse,
    PromptInfo,
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
def test_create_prompt(mock_http, store):
    """Test creating a prompt."""
    name = "catalog_1.schema_1.prompt_1"
    description = "Test prompt"
    template = "This is a {{test}} template"
    tags = [{"key": "test_key", "value": "test_value"}]
    
    store.create_prompt(name=name, template=template, description=description, tags=tags)
    
    _verify_requests(
        mock_http,
        "prompts",
        "POST",
        CreatePromptRequest(
            name=name,
            description=description,
            template=template,
            tags=[PromptTag(key="test_key", value="test_value")],
        ),
    )


@mock_http_200
def test_get_prompt(mock_http, store):
    """Test getting a prompt."""
    name = "catalog_1.schema_1.prompt_1"
    store.get_prompt(name=name)
    
    _verify_requests(
        mock_http,
        f"prompts/{name}",
        "GET",
        GetPromptRequest(name=name),
    )


@mock_http_200
def test_delete_prompt(mock_http, store):
    """Test deleting a prompt."""
    name = "catalog_1.schema_1.prompt_1"
    store.delete_prompt(name=name)
    
    _verify_requests(
        mock_http,
        f"prompts/{name}",
        "DELETE",
        DeletePromptRequest(name=name),
    )


@mock_http_200
def test_search_prompts(mock_http, store):
    """Test searching prompts."""
    filter_string = "name LIKE 'test%'"
    max_results = 10
    page_token = "test_token"
    
    store.search_prompts(
        filter_string=filter_string,
        max_results=max_results,
        page_token=page_token,
    )
    
    _verify_requests(
        mock_http,
        "prompts/search",
        "POST",
        SearchPromptsRequest(
            filter=filter_string,
            max_results=max_results,
            page_token=page_token,
        ),
    )


@mock_http_200
def test_set_prompt_tag(mock_http, store):
    """Test setting a prompt tag."""
    name = "catalog_1.schema_1.prompt_1"
    key = "test_key"
    value = "test_value"
    
    store.set_prompt_tag(name=name, key=key, value=value)
    
    _verify_requests(
        mock_http,
        f"prompts/{name}/tags",
        "POST",
        SetPromptTagRequest(name=name, key=key, value=value),
    )


@mock_http_200
def test_delete_prompt_tag(mock_http, store):
    """Test deleting a prompt tag."""
    name = "catalog_1.schema_1.prompt_1"
    key = "test_key"
    
    store.delete_prompt_tag(name=name, key=key)
    
    _verify_requests(
        mock_http,
        f"prompts/{name}/tags/{key}",
        "DELETE",
        DeletePromptTagRequest(name=name, key=key),
    )


@mock_http_200
def test_set_prompt_version_tag(mock_http, store):
    """Test setting a prompt version tag."""
    name = "catalog_1.schema_1.prompt_1"
    version = "1"
    key = "test_key"
    value = "test_value"
    
    store.set_prompt_version_tag(name=name, version=version, key=key, value=value)
    
    _verify_requests(
        mock_http,
        f"prompts/{name}/versions/{version}/tags",
        "POST",
        SetPromptVersionTagRequest(name=name, version=version, key=key, value=value),
    )


@mock_http_200
def test_delete_prompt_version_tag(mock_http, store):
    """Test deleting a prompt version tag."""
    name = "catalog_1.schema_1.prompt_1"
    version = "1"
    key = "test_key"
    
    store.delete_prompt_version_tag(name=name, version=version, key=key)
    
    _verify_requests(
        mock_http,
        f"prompts/{name}/versions/{version}/tags/{key}",
        "DELETE",
        DeletePromptVersionTagRequest(name=name, version=version, key=key),
    )


@mock_http_200
def test_set_prompt_alias(mock_http, store):
    """Test setting a prompt alias."""
    name = "catalog_1.schema_1.prompt_1"
    alias = "test_alias"
    version = "1"
    
    store.set_prompt_alias(name=name, alias=alias, version=version)
    
    _verify_requests(
        mock_http,
        f"prompts/{name}/aliases/{alias}",
        "POST",
        SetPromptAliasRequest(name=name, alias=alias, version=version),
    )


@mock_http_200
def test_delete_prompt_alias(mock_http, store):
    """Test deleting a prompt alias."""
    name = "catalog_1.schema_1.prompt_1"
    alias = "test_alias"
    
    store.delete_prompt_alias(name=name, alias=alias)
    
    _verify_requests(
        mock_http,
        f"prompts/{name}/aliases/{alias}",
        "DELETE",
        DeletePromptAliasRequest(name=name, alias=alias),
    )


@mock_http_200
def test_get_prompt_version_by_alias(mock_http, store):
    """Test getting a prompt version by alias."""
    name = "catalog_1.schema_1.prompt_1"
    alias = "test_alias"
    
    store.get_prompt_version_by_alias(name=name, alias=alias)
    
    _verify_requests(
        mock_http,
        f"prompts/{name}/versions/by-alias/{alias}",
        "GET",
        GetPromptVersionByAliasRequest(name=name, alias=alias),
    ) 