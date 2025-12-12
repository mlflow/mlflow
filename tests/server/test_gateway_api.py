from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.gateway.config import (
    AWSBaseConfig,
    AWSIdAndKey,
    AWSRole,
    EndpointType,
    GeminiConfig,
    MistralConfig,
    OpenAIAPIType,
    OpenAIConfig,
)
from mlflow.gateway.providers.anthropic import AnthropicProvider
from mlflow.gateway.providers.bedrock import AmazonBedrockProvider
from mlflow.gateway.providers.gemini import GeminiProvider
from mlflow.gateway.providers.mistral import MistralProvider
from mlflow.gateway.providers.openai import OpenAIProvider
from mlflow.gateway.schemas import chat, embeddings
from mlflow.server.gateway_api import (
    _create_invocations_handler,
    _create_provider_from_endpoint_config,
    _register_gateway_endpoints,
    gateway_router,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

pytestmark = pytest.mark.notrackingurimock

TEST_PASSPHRASE = "test-passphrase-for-gateway-api-tests"


@pytest.fixture(autouse=True)
def set_kek_passphrase(monkeypatch):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", TEST_PASSPHRASE)


@pytest.fixture
def store(tmp_path: Path):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir(exist_ok=True)
    if db_uri_env := MLFLOW_TRACKING_URI.get():
        s = SqlAlchemyStore(db_uri_env, artifact_uri.as_uri())
        yield s
    else:
        db_path = tmp_path / "mlflow.db"
        db_uri = f"sqlite:///{db_path}"
        s = SqlAlchemyStore(db_uri, artifact_uri.as_uri())
        yield s


def test_create_provider_from_endpoint_config_openai(store: SqlAlchemyStore):
    # Create test data
    secret = store.create_gateway_secret(
        secret_name="openai-key",
        secret_value={"api_key": "sk-test-123"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="gpt-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4o",
    )
    endpoint = store.create_gateway_endpoint(
        name="test-openai-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    provider = _create_provider_from_endpoint_config(
        endpoint.endpoint_id, store, EndpointType.LLM_V1_CHAT
    )

    assert isinstance(provider, OpenAIProvider)
    assert isinstance(provider.config.model.config, OpenAIConfig)
    assert provider.config.model.config.openai_api_key == "sk-test-123"


def test_create_provider_from_endpoint_config_azure_openai(store: SqlAlchemyStore):
    # Test Azure OpenAI configuration
    secret = store.create_gateway_secret(
        secret_name="azure-openai-key",
        secret_value={"api_key": "azure-api-key-test"},
        provider="openai",
        auth_config={
            "api_type": "azure",
            "api_base": "https://my-resource.openai.azure.com",
            "deployment_name": "gpt-4-deployment",
            "api_version": "2024-02-01",
        },
    )
    model_def = store.create_gateway_model_definition(
        name="azure-gpt-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    endpoint = store.create_gateway_endpoint(
        name="test-azure-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    provider = _create_provider_from_endpoint_config(
        endpoint.endpoint_id, store, EndpointType.LLM_V1_CHAT
    )

    assert isinstance(provider, OpenAIProvider)
    assert isinstance(provider.config.model.config, OpenAIConfig)
    assert provider.config.model.config.openai_api_type == OpenAIAPIType.AZURE
    assert provider.config.model.config.openai_api_base == "https://my-resource.openai.azure.com"
    assert provider.config.model.config.openai_deployment_name == "gpt-4-deployment"
    assert provider.config.model.config.openai_api_version == "2024-02-01"
    assert provider.config.model.config.openai_api_key == "azure-api-key-test"


def test_create_provider_from_endpoint_config_azure_openai_with_azuread(store: SqlAlchemyStore):
    # Test Azure OpenAI with AzureAD authentication
    secret = store.create_gateway_secret(
        secret_name="azuread-openai-key",
        secret_value={"api_key": "azuread-api-key-test"},
        provider="openai",
        auth_config={
            "api_type": "azuread",
            "api_base": "https://my-resource-ad.openai.azure.com",
            "deployment_name": "gpt-4-deployment-ad",
            "api_version": "2024-02-01",
        },
    )
    model_def = store.create_gateway_model_definition(
        name="azuread-gpt-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    endpoint = store.create_gateway_endpoint(
        name="test-azuread-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    provider = _create_provider_from_endpoint_config(
        endpoint.endpoint_id, store, EndpointType.LLM_V1_CHAT
    )

    assert isinstance(provider, OpenAIProvider)
    assert isinstance(provider.config.model.config, OpenAIConfig)
    assert provider.config.model.config.openai_api_type == OpenAIAPIType.AZUREAD
    assert provider.config.model.config.openai_api_base == "https://my-resource-ad.openai.azure.com"
    assert provider.config.model.config.openai_deployment_name == "gpt-4-deployment-ad"
    assert provider.config.model.config.openai_api_version == "2024-02-01"
    assert provider.config.model.config.openai_api_key == "azuread-api-key-test"


def test_create_provider_from_endpoint_config_anthropic(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="anthropic-key",
        secret_value={"api_key": "sk-ant-test"},
        provider="anthropic",
    )
    model_def = store.create_gateway_model_definition(
        name="claude-model",
        secret_id=secret.secret_id,
        provider="anthropic",
        model_name="claude-3-sonnet",
    )
    endpoint = store.create_gateway_endpoint(
        name="test-anthropic-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    provider = _create_provider_from_endpoint_config(
        endpoint.endpoint_id, store, EndpointType.LLM_V1_CHAT
    )

    assert isinstance(provider, AnthropicProvider)
    assert provider.config.model.config.anthropic_api_key == "sk-ant-test"


def test_create_provider_from_endpoint_config_bedrock_base_config(store: SqlAlchemyStore):
    # Test Bedrock with base config (default credentials chain)
    secret = store.create_gateway_secret(
        secret_name="bedrock-base-key",
        secret_value={"api_key": "placeholder"},
        provider="bedrock",
        auth_config={"aws_region": "us-east-1"},
    )
    model_def = store.create_gateway_model_definition(
        name="bedrock-base-model",
        secret_id=secret.secret_id,
        provider="bedrock",
        model_name="anthropic.claude-3-sonnet",
    )
    endpoint = store.create_gateway_endpoint(
        name="test-bedrock-base-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    provider = _create_provider_from_endpoint_config(
        endpoint.endpoint_id, store, EndpointType.LLM_V1_CHAT
    )

    assert isinstance(provider, AmazonBedrockProvider)
    assert isinstance(provider.config.model.config.aws_config, AWSBaseConfig)
    assert provider.config.model.config.aws_config.aws_region == "us-east-1"


def test_create_provider_from_endpoint_config_bedrock_access_keys(store: SqlAlchemyStore):
    # Test Bedrock with access key authentication
    secret = store.create_gateway_secret(
        secret_name="bedrock-keys",
        secret_value={
            "aws_access_key_id": "AKIA1234567890",
            "aws_secret_access_key": "secret-key-value",
            "aws_session_token": "session-token",
        },
        provider="bedrock",
        auth_config={"aws_region": "us-west-2"},
    )
    model_def = store.create_gateway_model_definition(
        name="bedrock-keys-model",
        secret_id=secret.secret_id,
        provider="bedrock",
        model_name="anthropic.claude-3-sonnet",
    )
    endpoint = store.create_gateway_endpoint(
        name="test-bedrock-keys-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    provider = _create_provider_from_endpoint_config(
        endpoint.endpoint_id, store, EndpointType.LLM_V1_CHAT
    )

    assert isinstance(provider, AmazonBedrockProvider)
    assert isinstance(provider.config.model.config.aws_config, AWSIdAndKey)
    assert provider.config.model.config.aws_config.aws_access_key_id == "AKIA1234567890"
    assert provider.config.model.config.aws_config.aws_secret_access_key == "secret-key-value"
    assert provider.config.model.config.aws_config.aws_session_token == "session-token"
    assert provider.config.model.config.aws_config.aws_region == "us-west-2"


def test_create_provider_from_endpoint_config_bedrock_role(store: SqlAlchemyStore):
    # Test Bedrock with role-based authentication
    secret = store.create_gateway_secret(
        secret_name="bedrock-role-key",
        secret_value={"api_key": "placeholder"},
        provider="bedrock",
        auth_config={
            "aws_role_arn": "arn:aws:iam::123456789012:role/MyBedrockRole",
            "session_length_seconds": 3600,
            "aws_region": "eu-west-1",
        },
    )
    model_def = store.create_gateway_model_definition(
        name="bedrock-role-model",
        secret_id=secret.secret_id,
        provider="bedrock",
        model_name="anthropic.claude-3-sonnet",
    )
    endpoint = store.create_gateway_endpoint(
        name="test-bedrock-role-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    provider = _create_provider_from_endpoint_config(
        endpoint.endpoint_id, store, EndpointType.LLM_V1_CHAT
    )

    assert isinstance(provider, AmazonBedrockProvider)
    assert isinstance(provider.config.model.config.aws_config, AWSRole)
    assert (
        provider.config.model.config.aws_config.aws_role_arn
        == "arn:aws:iam::123456789012:role/MyBedrockRole"
    )
    assert provider.config.model.config.aws_config.session_length_seconds == 3600
    assert provider.config.model.config.aws_config.aws_region == "eu-west-1"


def test_create_provider_from_endpoint_config_mistral(store: SqlAlchemyStore):
    # Test Mistral provider
    secret = store.create_gateway_secret(
        secret_name="mistral-key",
        secret_value={"api_key": "mistral-test-key"},
        provider="mistral",
    )
    model_def = store.create_gateway_model_definition(
        name="mistral-model",
        secret_id=secret.secret_id,
        provider="mistral",
        model_name="mistral-large-latest",
    )
    endpoint = store.create_gateway_endpoint(
        name="test-mistral-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    provider = _create_provider_from_endpoint_config(
        endpoint.endpoint_id, store, EndpointType.LLM_V1_CHAT
    )

    assert isinstance(provider, MistralProvider)
    assert isinstance(provider.config.model.config, MistralConfig)
    assert provider.config.model.config.mistral_api_key == "mistral-test-key"


def test_create_provider_from_endpoint_config_gemini(store: SqlAlchemyStore):
    # Test Gemini provider
    secret = store.create_gateway_secret(
        secret_name="gemini-key",
        secret_value={"api_key": "gemini-test-key"},
        provider="gemini",
    )
    model_def = store.create_gateway_model_definition(
        name="gemini-model",
        secret_id=secret.secret_id,
        provider="gemini",
        model_name="gemini-1.5-pro",
    )
    endpoint = store.create_gateway_endpoint(
        name="test-gemini-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    provider = _create_provider_from_endpoint_config(
        endpoint.endpoint_id, store, EndpointType.LLM_V1_CHAT
    )

    assert isinstance(provider, GeminiProvider)
    assert isinstance(provider.config.model.config, GeminiConfig)
    assert provider.config.model.config.gemini_api_key == "gemini-test-key"


def test_create_provider_from_endpoint_config_nonexistent_endpoint(store: SqlAlchemyStore):
    from mlflow.exceptions import MlflowException

    with pytest.raises(MlflowException, match="not found"):
        _create_provider_from_endpoint_config("nonexistent-id", store, EndpointType.LLM_V1_CHAT)


def test_register_gateway_endpoints(store: SqlAlchemyStore):
    # Create test endpoints
    secret = store.create_gateway_secret(
        secret_name="test-key",
        secret_value={"api_key": "sk-test"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="test-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    store.create_gateway_endpoint(
        name="test-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    # Register endpoints
    router = _register_gateway_endpoints(store)

    # Check that routes were registered
    route_paths = [route.path for route in router.routes]
    assert "/gateway/test-endpoint/mlflow/invocations" in route_paths


@pytest.mark.asyncio
async def test_invocations_handler_chat(store: SqlAlchemyStore):
    # Create test data
    secret = store.create_gateway_secret(
        secret_name="chat-key",
        secret_value={"api_key": "sk-test-chat"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="chat-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    endpoint = store.create_gateway_endpoint(
        name="chat-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    # Create handler
    handler = _create_invocations_handler(endpoint.endpoint_id, store)

    # Mock the provider's chat method
    mock_response = chat.ResponsePayload(
        id="test-id",
        object="chat.completion",
        created=1234567890,
        model="gpt-4",
        choices=[
            chat.Choice(
                index=0,
                message=chat.ResponseMessage(role="assistant", content="Hello!"),
                finish_reason="stop",
            )
        ],
        usage=chat.ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    # Create a mock request with chat payload
    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "stream": False,
        }
    )

    # Patch the provider creation to return a mocked provider
    with patch(
        "mlflow.server.gateway_api._create_provider_from_endpoint_config"
    ) as mock_create_provider:
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(return_value=mock_response)
        mock_create_provider.return_value = mock_provider

        # Call the handler
        response = await handler(mock_request)

        # Verify
        assert response.id == "test-id"
        assert response.choices[0].message.content == "Hello!"
        assert mock_provider.chat.called


@pytest.mark.asyncio
async def test_invocations_handler_embeddings(store: SqlAlchemyStore):
    # Create test data
    secret = store.create_gateway_secret(
        secret_name="embed-key",
        secret_value={"api_key": "sk-test-embed"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="embed-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="text-embedding-ada-002",
    )
    endpoint = store.create_gateway_endpoint(
        name="embed-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    # Create handler
    handler = _create_invocations_handler(endpoint.endpoint_id, store)

    # Mock the provider's embeddings method
    mock_response = embeddings.ResponsePayload(
        object="list",
        data=[embeddings.EmbeddingObject(embedding=[0.1, 0.2, 0.3], index=0)],
        model="text-embedding-ada-002",
        usage=embeddings.EmbeddingsUsage(prompt_tokens=5, total_tokens=5),
    )

    # Create a mock request with embeddings payload
    mock_request = MagicMock()
    mock_request.json = AsyncMock(return_value={"input": "test text"})

    # Patch the provider creation to return a mocked provider
    with patch(
        "mlflow.server.gateway_api._create_provider_from_endpoint_config"
    ) as mock_create_provider:
        mock_provider = MagicMock()
        mock_provider.embeddings = AsyncMock(return_value=mock_response)
        mock_create_provider.return_value = mock_provider

        # Call the handler
        response = await handler(mock_request)

        # Verify
        assert response.object == "list"
        assert len(response.data) == 1
        assert response.data[0].embedding == [0.1, 0.2, 0.3]
        assert mock_provider.embeddings.called


def test_gateway_router_initialization():
    assert gateway_router is not None
    assert gateway_router.prefix == "/gateway"


@pytest.mark.asyncio
async def test_invocations_handler_invalid_json(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="test-key",
        secret_value={"api_key": "sk-test"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="test-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    endpoint = store.create_gateway_endpoint(
        name="test-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    handler = _create_invocations_handler(endpoint.endpoint_id, store)

    # Mock request that raises exception when parsing JSON
    mock_request = MagicMock()
    mock_request.json = AsyncMock(side_effect=ValueError("Invalid JSON"))

    with pytest.raises(HTTPException, match="Invalid JSON payload: Invalid JSON") as exc_info:
        await handler(mock_request)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_invocations_handler_missing_fields(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="test-key",
        secret_value={"api_key": "sk-test"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="test-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    endpoint = store.create_gateway_endpoint(
        name="test-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    handler = _create_invocations_handler(endpoint.endpoint_id, store)

    # Create request with neither messages nor input
    mock_request = MagicMock()
    mock_request.json = AsyncMock(return_value={"temperature": 0.7})

    with patch(
        "mlflow.server.gateway_api._create_provider_from_endpoint_config"
    ) as mock_create_provider:
        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider

        with pytest.raises(
            HTTPException, match="Invalid request: payload format must be either chat or embeddings"
        ) as exc_info:
            await handler(mock_request)

        assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_invocations_handler_invalid_chat_payload(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="test-key",
        secret_value={"api_key": "sk-test"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="test-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    endpoint = store.create_gateway_endpoint(
        name="test-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    handler = _create_invocations_handler(endpoint.endpoint_id, store)

    # Create request with invalid messages structure
    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "messages": "not a list",  # Should be a list
            "stream": False,
        }
    )

    with patch(
        "mlflow.server.gateway_api._create_provider_from_endpoint_config"
    ) as mock_create_provider:
        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider

        with pytest.raises(HTTPException, match="Invalid chat payload") as exc_info:
            await handler(mock_request)

        assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_invocations_handler_invalid_embeddings_payload(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="test-key",
        secret_value={"api_key": "sk-test"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="test-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="text-embedding-ada-002",
    )
    endpoint = store.create_gateway_endpoint(
        name="test-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    handler = _create_invocations_handler(endpoint.endpoint_id, store)

    # Create request with invalid input structure
    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "input": 123,  # Should be string or list of strings
        }
    )

    with patch(
        "mlflow.server.gateway_api._create_provider_from_endpoint_config"
    ) as mock_create_provider:
        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider

        with pytest.raises(HTTPException, match="Invalid embeddings payload") as exc_info:
            await handler(mock_request)

        assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_invocations_handler_streaming(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="test-key",
        secret_value={"api_key": "sk-test"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="test-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    endpoint = store.create_gateway_endpoint(
        name="test-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    handler = _create_invocations_handler(endpoint.endpoint_id, store)

    # Create streaming request
    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
    )

    # Mock streaming response
    mock_streaming_response = MagicMock()

    with (
        patch(
            "mlflow.server.gateway_api._create_provider_from_endpoint_config"
        ) as mock_create_provider,
        patch(
            "mlflow.server.gateway_api.make_streaming_response",
            return_value=mock_streaming_response,
        ) as mock_make_streaming,
    ):
        mock_provider = MagicMock()
        mock_provider.chat_stream = MagicMock(return_value="mock_stream")
        mock_create_provider.return_value = mock_provider

        response = await handler(mock_request)

        # Verify streaming was called
        assert mock_provider.chat_stream.called
        assert mock_make_streaming.called
        assert response == mock_streaming_response


def test_create_provider_from_endpoint_config_no_models(store: SqlAlchemyStore):
    from mlflow.exceptions import MlflowException
    from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig

    # Create a minimal endpoint to get an endpoint_id
    secret = store.create_gateway_secret(
        secret_name="test-key",
        secret_value={"api_key": "sk-test"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="test-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    endpoint = store.create_gateway_endpoint(
        name="test-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    # Mock get_endpoint_config to return an empty models list
    with patch(
        "mlflow.server.gateway_api.get_endpoint_config",
        return_value=GatewayEndpointConfig(
            endpoint_id=endpoint.endpoint_id, endpoint_name="test-endpoint", models=[]
        ),
    ):
        with pytest.raises(MlflowException, match="has no models configured"):
            _create_provider_from_endpoint_config(
                endpoint.endpoint_id, store, EndpointType.LLM_V1_CHAT
            )
