from pathlib import Path
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.gateway.config import (
    AWSBaseConfig,
    AWSIdAndKey,
    AWSRole,
    EndpointType,
    GeminiConfig,
    LiteLLMConfig,
    MistralConfig,
    OpenAIAPIType,
    OpenAIConfig,
)
from mlflow.gateway.providers.anthropic import AnthropicProvider
from mlflow.gateway.providers.bedrock import AmazonBedrockProvider
from mlflow.gateway.providers.gemini import GeminiProvider
from mlflow.gateway.providers.litellm import LiteLLMProvider
from mlflow.gateway.providers.mistral import MistralProvider
from mlflow.gateway.providers.openai import OpenAIProvider
from mlflow.gateway.schemas import chat, embeddings
from mlflow.server.gateway_api import (
    _create_provider_from_endpoint_name,
    anthropic_passthrough_messages,
    chat_completions,
    gateway_router,
    gemini_passthrough_generate_content,
    gemini_passthrough_stream_generate_content,
    invocations,
    openai_passthrough_chat,
    openai_passthrough_embeddings,
    openai_passthrough_responses,
)
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig
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
    db_path = tmp_path / "mlflow.db"
    db_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(db_uri)
    yield SqlAlchemyStore(db_uri, artifact_uri.as_uri())
    mlflow.set_tracking_uri(None)


def test_create_provider_from_endpoint_name_openai(store: SqlAlchemyStore):
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

    provider = _create_provider_from_endpoint_name(store, endpoint.name, EndpointType.LLM_V1_CHAT)

    assert isinstance(provider, OpenAIProvider)
    assert isinstance(provider.config.model.config, OpenAIConfig)
    assert provider.config.model.config.openai_api_key == "sk-test-123"


def test_create_provider_from_endpoint_name_azure_openai(store: SqlAlchemyStore):
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

    provider = _create_provider_from_endpoint_name(store, endpoint.name, EndpointType.LLM_V1_CHAT)

    assert isinstance(provider, OpenAIProvider)
    assert isinstance(provider.config.model.config, OpenAIConfig)
    assert provider.config.model.config.openai_api_type == OpenAIAPIType.AZURE
    assert provider.config.model.config.openai_api_base == "https://my-resource.openai.azure.com"
    assert provider.config.model.config.openai_deployment_name == "gpt-4-deployment"
    assert provider.config.model.config.openai_api_version == "2024-02-01"
    assert provider.config.model.config.openai_api_key == "azure-api-key-test"


def test_create_provider_from_endpoint_name_azure_openai_with_azuread(store: SqlAlchemyStore):
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

    provider = _create_provider_from_endpoint_name(store, endpoint.name, EndpointType.LLM_V1_CHAT)

    assert isinstance(provider, OpenAIProvider)
    assert isinstance(provider.config.model.config, OpenAIConfig)
    assert provider.config.model.config.openai_api_type == OpenAIAPIType.AZUREAD
    assert provider.config.model.config.openai_api_base == "https://my-resource-ad.openai.azure.com"
    assert provider.config.model.config.openai_deployment_name == "gpt-4-deployment-ad"
    assert provider.config.model.config.openai_api_version == "2024-02-01"
    assert provider.config.model.config.openai_api_key == "azuread-api-key-test"


def test_create_provider_from_endpoint_name_anthropic(store: SqlAlchemyStore):
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

    provider = _create_provider_from_endpoint_name(store, endpoint.name, EndpointType.LLM_V1_CHAT)

    assert isinstance(provider, AnthropicProvider)
    assert provider.config.model.config.anthropic_api_key == "sk-ant-test"


def test_create_provider_from_endpoint_name_bedrock_base_config(store: SqlAlchemyStore):
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

    provider = _create_provider_from_endpoint_name(store, endpoint.name, EndpointType.LLM_V1_CHAT)

    assert isinstance(provider, AmazonBedrockProvider)
    assert isinstance(provider.config.model.config.aws_config, AWSBaseConfig)
    assert provider.config.model.config.aws_config.aws_region == "us-east-1"


def test_create_provider_from_endpoint_name_bedrock_access_keys(store: SqlAlchemyStore):
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

    provider = _create_provider_from_endpoint_name(store, endpoint.name, EndpointType.LLM_V1_CHAT)

    assert isinstance(provider, AmazonBedrockProvider)
    assert isinstance(provider.config.model.config.aws_config, AWSIdAndKey)
    assert provider.config.model.config.aws_config.aws_access_key_id == "AKIA1234567890"
    assert provider.config.model.config.aws_config.aws_secret_access_key == "secret-key-value"
    assert provider.config.model.config.aws_config.aws_session_token == "session-token"
    assert provider.config.model.config.aws_config.aws_region == "us-west-2"


def test_create_provider_from_endpoint_name_bedrock_role(store: SqlAlchemyStore):
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

    provider = _create_provider_from_endpoint_name(store, endpoint.name, EndpointType.LLM_V1_CHAT)

    assert isinstance(provider, AmazonBedrockProvider)
    assert isinstance(provider.config.model.config.aws_config, AWSRole)
    assert (
        provider.config.model.config.aws_config.aws_role_arn
        == "arn:aws:iam::123456789012:role/MyBedrockRole"
    )
    assert provider.config.model.config.aws_config.session_length_seconds == 3600
    assert provider.config.model.config.aws_config.aws_region == "eu-west-1"


def test_create_provider_from_endpoint_name_mistral(store: SqlAlchemyStore):
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

    provider = _create_provider_from_endpoint_name(store, endpoint.name, EndpointType.LLM_V1_CHAT)

    assert isinstance(provider, MistralProvider)
    assert isinstance(provider.config.model.config, MistralConfig)
    assert provider.config.model.config.mistral_api_key == "mistral-test-key"


def test_create_provider_from_endpoint_name_gemini(store: SqlAlchemyStore):
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

    provider = _create_provider_from_endpoint_name(store, endpoint.name, EndpointType.LLM_V1_CHAT)

    assert isinstance(provider, GeminiProvider)
    assert isinstance(provider.config.model.config, GeminiConfig)
    assert provider.config.model.config.gemini_api_key == "gemini-test-key"


def test_create_provider_from_endpoint_name_litellm(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="litellm-key",
        secret_value={"api_key": "litellm-test-key"},
        provider="litellm",
    )
    model_def = store.create_gateway_model_definition(
        name="litellm-model",
        secret_id=secret.secret_id,
        provider="litellm",
        model_name="claude-3-5-sonnet-20241022",
    )
    endpoint = store.create_gateway_endpoint(
        name="test-litellm-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    provider = _create_provider_from_endpoint_name(store, endpoint.name, EndpointType.LLM_V1_CHAT)

    assert isinstance(provider, LiteLLMProvider)
    assert isinstance(provider.config.model.config, LiteLLMConfig)
    assert provider.config.model.config.litellm_api_key == "litellm-test-key"
    assert provider.config.model.config.litellm_provider == "litellm"


def test_create_provider_from_endpoint_name_litellm_with_api_base(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="litellm-custom-key",
        secret_value={"api_key": "litellm-custom-key"},
        provider="litellm",
        auth_config={"api_base": "https://custom-api.example.com"},
    )
    model_def = store.create_gateway_model_definition(
        name="litellm-custom-model",
        secret_id=secret.secret_id,
        provider="litellm",
        model_name="custom-model",
    )
    endpoint = store.create_gateway_endpoint(
        name="test-litellm-custom-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    provider = _create_provider_from_endpoint_name(store, endpoint.name, EndpointType.LLM_V1_CHAT)

    assert isinstance(provider, LiteLLMProvider)
    assert isinstance(provider.config.model.config, LiteLLMConfig)
    assert provider.config.model.config.litellm_api_key == "litellm-custom-key"
    assert provider.config.model.config.litellm_api_base == "https://custom-api.example.com"
    assert provider.config.model.config.litellm_provider == "litellm"


def test_create_provider_from_endpoint_name_nonexistent_endpoint(store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match="not found"):
        _create_provider_from_endpoint_name(store, "nonexistent-id", EndpointType.LLM_V1_CHAT)


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
        "mlflow.server.gateway_api._create_provider_from_endpoint_name"
    ) as mock_create_provider:
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(return_value=mock_response)
        mock_create_provider.return_value = mock_provider

        # Call the handler
        response = await invocations(endpoint.name, mock_request)

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
        "mlflow.server.gateway_api._create_provider_from_endpoint_name"
    ) as mock_create_provider:
        mock_provider = MagicMock()
        mock_provider.embeddings = AsyncMock(return_value=mock_response)
        mock_create_provider.return_value = mock_provider

        # Call the handler
        response = await invocations(endpoint.name, mock_request)

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

    # Mock request that raises exception when parsing JSON
    mock_request = MagicMock()
    mock_request.json = AsyncMock(side_effect=ValueError("Invalid JSON"))

    with pytest.raises(HTTPException, match="Invalid JSON payload: Invalid JSON") as exc_info:
        await invocations(endpoint.name, mock_request)

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

    # Create request with neither messages nor input
    mock_request = MagicMock()
    mock_request.json = AsyncMock(return_value={"temperature": 0.7})

    with patch(
        "mlflow.server.gateway_api._create_provider_from_endpoint_name"
    ) as mock_create_provider:
        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider

        with pytest.raises(
            HTTPException, match="Invalid request: payload format must be either chat or embeddings"
        ) as exc_info:
            await invocations(endpoint.name, mock_request)

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

    # Create request with invalid messages structure
    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "messages": "not a list",  # Should be a list
            "stream": False,
        }
    )

    with patch(
        "mlflow.server.gateway_api._create_provider_from_endpoint_name"
    ) as mock_create_provider:
        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider

        with pytest.raises(HTTPException, match="Invalid chat payload") as exc_info:
            await invocations(endpoint.name, mock_request)

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

    # Create request with invalid input structure
    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "input": 123,  # Should be string or list of strings
        }
    )

    with patch(
        "mlflow.server.gateway_api._create_provider_from_endpoint_name"
    ) as mock_create_provider:
        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider

        with pytest.raises(HTTPException, match="Invalid embeddings payload") as exc_info:
            await invocations(endpoint.name, mock_request)

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
            "mlflow.server.gateway_api._create_provider_from_endpoint_name"
        ) as mock_create_provider,
        patch(
            "mlflow.server.gateway_api.make_streaming_response",
            return_value=mock_streaming_response,
        ) as mock_make_streaming,
    ):
        mock_provider = MagicMock()
        mock_provider.chat_stream = MagicMock(return_value="mock_stream")
        mock_create_provider.return_value = mock_provider

        response = await invocations(endpoint.name, mock_request)

        # Verify streaming was called
        assert mock_provider.chat_stream.called
        assert mock_make_streaming.called
        assert response == mock_streaming_response


def test_create_provider_from_endpoint_name_no_models(store: SqlAlchemyStore):
    # Create a minimal endpoint to get an endpoint_name
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
            _create_provider_from_endpoint_name(store, endpoint.name, EndpointType.LLM_V1_CHAT)


# =============================================================================
# OpenAI-compatible chat completions endpoint tests
# =============================================================================


@pytest.mark.asyncio
async def test_chat_completions_endpoint(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="openai-compat-key",
        secret_value={"api_key": "sk-test"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="openai-compat-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    store.create_gateway_endpoint(
        name="my-chat-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    # Mock the provider's chat method
    mock_response = chat.ResponsePayload(
        id="test-id",
        object="chat.completion",
        created=1234567890,
        model="gpt-4",
        choices=[
            chat.Choice(
                index=0,
                message=chat.ResponseMessage(role="assistant", content="Hello from OpenAI!"),
                finish_reason="stop",
            )
        ],
        usage=chat.ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    # Create a mock request with OpenAI-compatible format
    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "model": "my-chat-endpoint",  # Endpoint name via model parameter
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "stream": False,
        }
    )

    # Patch the provider creation to return a mocked provider
    with patch(
        "mlflow.server.gateway_api._create_provider_from_endpoint_name"
    ) as mock_create_provider:
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(return_value=mock_response)
        mock_create_provider.return_value = mock_provider

        # Call the handler
        response = await chat_completions(mock_request)

        # Verify
        assert response.id == "test-id"
        assert response.choices[0].message.content == "Hello from OpenAI!"
        assert mock_provider.chat.called


@pytest.mark.asyncio
async def test_chat_completions_endpoint_streaming(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="stream-key",
        secret_value={"api_key": "sk-test"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="stream-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    store.create_gateway_endpoint(
        name="stream-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    # Create streaming request
    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "model": "stream-endpoint",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
    )

    # Mock streaming response
    mock_streaming_response = MagicMock()

    with (
        patch(
            "mlflow.server.gateway_api._create_provider_from_endpoint_name"
        ) as mock_create_provider,
        patch(
            "mlflow.server.gateway_api.make_streaming_response",
            return_value=mock_streaming_response,
        ) as mock_make_streaming,
    ):
        mock_provider = MagicMock()
        mock_provider.chat_stream = MagicMock(return_value="mock_stream")
        mock_create_provider.return_value = mock_provider

        response = await chat_completions(mock_request)

        # Verify streaming was called
        assert mock_provider.chat_stream.called
        assert mock_make_streaming.called
        assert response == mock_streaming_response


@pytest.mark.asyncio
async def test_chat_completions_endpoint_missing_model_parameter(store: SqlAlchemyStore):
    # Create request without model parameter
    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "messages": [{"role": "user", "content": "Hi"}],
        }
    )

    with pytest.raises(HTTPException, match="Missing required 'model' parameter") as exc_info:
        await chat_completions(mock_request)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_chat_completions_endpoint_missing_messages(store: SqlAlchemyStore):
    # Create request without messages
    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "model": "my-endpoint",
            "temperature": 0.7,
        }
    )

    with pytest.raises(HTTPException, match="Invalid chat payload") as exc_info:
        await chat_completions(mock_request)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_chat_completions_endpoint_invalid_json(store: SqlAlchemyStore):
    mock_request = MagicMock()
    mock_request.json = AsyncMock(side_effect=ValueError("Invalid JSON"))

    with pytest.raises(HTTPException, match="Invalid JSON payload: Invalid JSON") as exc_info:
        await chat_completions(mock_request)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_openai_passthrough_chat(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="openai-passthrough-key",
        secret_value={"api_key": "sk-test-passthrough"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="openai-passthrough-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4o",
    )
    store.create_gateway_endpoint(
        name="openai-passthrough-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    # Mock OpenAI API response
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from passthrough!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    # Create mock request
    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "model": "openai-passthrough-endpoint",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )

    # Mock send_request directly
    with mock.patch(
        "mlflow.gateway.providers.openai.send_request", return_value=mock_response
    ) as mock_send:
        response = await openai_passthrough_chat(mock_request)

        # Verify send_request was called
        assert mock_send.called
        call_kwargs = mock_send.call_args[1]
        assert call_kwargs["path"] == "chat/completions"
        assert call_kwargs["payload"]["model"] == "gpt-4o"
        assert call_kwargs["payload"]["messages"] == [{"role": "user", "content": "Hello"}]

        # Verify response is raw OpenAI format
        assert response["id"] == "chatcmpl-123"
        assert response["model"] == "gpt-4o"
        assert response["choices"][0]["message"]["content"] == "Hello from passthrough!"


@pytest.mark.asyncio
async def test_openai_passthrough_embeddings(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="openai-embed-passthrough-key",
        secret_value={"api_key": "sk-test-embed"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="openai-embed-passthrough-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="text-embedding-3-small",
    )
    store.create_gateway_endpoint(
        name="openai-embed-passthrough-endpoint",
        model_definition_ids=[model_def.model_definition_id],
    )

    # Mock OpenAI API response
    mock_response = {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }

    # Create mock request
    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "model": "openai-embed-passthrough-endpoint",
            "input": "Test input",
        }
    )

    # Mock send_request directly
    with mock.patch(
        "mlflow.gateway.providers.openai.send_request", return_value=mock_response
    ) as mock_send:
        response = await openai_passthrough_embeddings(mock_request)

        # Verify send_request was called
        assert mock_send.called
        call_kwargs = mock_send.call_args[1]
        assert call_kwargs["path"] == "embeddings"
        assert call_kwargs["payload"]["model"] == "text-embedding-3-small"
        assert call_kwargs["payload"]["input"] == "Test input"

        # Verify response is raw OpenAI format
        assert response["model"] == "text-embedding-3-small"
        assert response["data"][0]["embedding"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_openai_passthrough_responses(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="openai-responses-key",
        secret_value={"api_key": "sk-test-responses"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="openai-responses-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4o",
    )
    store.create_gateway_endpoint(
        name="openai-responses-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    # Mock OpenAI Responses API response (using correct Responses API schema)
    mock_response = {
        "id": "resp-123",
        "object": "response",
        "created": 1234567890,
        "model": "gpt-4o",
        "status": "completed",
        "output": [
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Response from Responses API"}],
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    # Create mock request
    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "model": "openai-responses-endpoint",
            "input": [{"role": "user", "content": "Hello"}],
            "instructions": "You are a helpful assistant",
            "response_format": {"type": "text"},
        }
    )

    # Mock send_request directly
    with mock.patch(
        "mlflow.gateway.providers.openai.send_request", return_value=mock_response
    ) as mock_send:
        response = await openai_passthrough_responses(mock_request)

        # Verify send_request was called
        assert mock_send.called
        call_kwargs = mock_send.call_args[1]
        assert call_kwargs["path"] == "responses"
        assert call_kwargs["payload"]["model"] == "gpt-4o"
        assert call_kwargs["payload"]["input"] == [{"role": "user", "content": "Hello"}]
        assert call_kwargs["payload"]["instructions"] == "You are a helpful assistant"
        assert call_kwargs["payload"]["response_format"] == {"type": "text"}

        # Verify response is raw OpenAI Responses API format
        assert response["id"] == "resp-123"
        assert response["object"] == "response"
        assert response["status"] == "completed"
        assert response["output"][0]["content"][0]["text"] == "Response from Responses API"


@pytest.mark.asyncio
async def test_openai_passthrough_chat_streaming(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="openai-stream-passthrough-key",
        secret_value={"api_key": "sk-test-stream"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="openai-stream-passthrough-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4o",
    )
    store.create_gateway_endpoint(
        name="openai-stream-passthrough-endpoint",
        model_definition_ids=[model_def.model_definition_id],
    )

    # Create mock request with streaming enabled
    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "model": "openai-stream-passthrough-endpoint",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
    )

    # Mock streaming response chunks
    mock_stream_chunks = [
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n',  # noqa: E501
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n',  # noqa: E501
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',  # noqa: E501
    ]

    async def mock_stream_generator():
        for chunk in mock_stream_chunks:
            yield chunk

    with mock.patch(
        "mlflow.gateway.providers.openai.send_stream_request",
        return_value=mock_stream_generator(),
    ) as mock_send_stream:
        response = await openai_passthrough_chat(mock_request)

        assert mock_send_stream.called
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"

        chunks = [chunk async for chunk in response.body_iterator]

        assert len(chunks) == 3
        assert b"Hello" in chunks[0]
        assert b"world" in chunks[1]
        assert b"stop" in chunks[2]


@pytest.mark.asyncio
async def test_openai_passthrough_responses_streaming(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="openai-responses-stream-key",
        secret_value={"api_key": "sk-test-responses-stream"},
        provider="openai",
    )
    model_def = store.create_gateway_model_definition(
        name="openai-responses-stream-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4o",
    )
    store.create_gateway_endpoint(
        name="openai-responses-stream-endpoint",
        model_definition_ids=[model_def.model_definition_id],
    )

    # Create mock request with streaming enabled
    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "model": "openai-responses-stream-endpoint",
            "input": [{"type": "text", "text": "Hello"}],
            "instructions": "You are a helpful assistant",
            "stream": True,
        }
    )

    # Mock streaming response chunks for Responses API
    mock_stream_chunks = [
        b'data: {"type":"response.created","response":{"id":"resp_1","object":"response","created_at":1741290958,"status":"in_progress","error":null,"incomplete_details":null,"instructions":"You are a helpful assistant.","max_output_tokens":null,"model":"gpt-4.1-2025-04-14","output":[],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{"effort":null,"summary":null},"store":true,"temperature":1.0,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1.0,"truncation":"disabled","usage":null,"user":null,"metadata":{}}}\n\n',  # noqa: E501
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","status":"in_progress","role":"assistant","content":[]}}\n\n',  # noqa: E501
        b'data: {"type":"response.content_part.added","item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}\n\n',  # noqa: E501
        b'data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"Hi"}\n\n',  # noqa: E501
        b'data: {"type":"response.output_text.done","item_id":"msg_1","output_index":0,"content_index":0,"text":"Hi there! How can I assist you today?"}\n\n',  # noqa: E501
        b'data: {"type":"response.content_part.done","item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"Hi there! How can I assist you today?","annotations":[]}}\n\n',  # noqa: E501
        b'data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hi there! How can I assist you today?","annotations":[]}]}}\n\n',  # noqa: E501
        b'data: {"type":"response.completed","response":{"id":"resp_1","object":"response","created_at":1741290958,"status":"completed","error":null,"incomplete_details":null,"instructions":"You are a helpful assistant.","max_output_tokens":null,"model":"gpt-4.1-2025-04-14","output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hi there! How can I assist you today?","annotations":[]}]}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{"effort":null,"summary":null},"store":true,"temperature":1.0,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1.0,"truncation":"disabled","usage":{"input_tokens":37,"output_tokens":11,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":48},"user":null,"metadata":{}}}\n\n',  # noqa: E501
    ]

    async def mock_stream_generator():
        for chunk in mock_stream_chunks:
            yield chunk

    with mock.patch(
        "mlflow.gateway.providers.openai.send_stream_request",
        return_value=mock_stream_generator(),
    ) as mock_send_stream:
        response = await openai_passthrough_responses(mock_request)

        assert mock_send_stream.called
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"

        chunks = [chunk async for chunk in response.body_iterator]

        assert len(chunks) == 8
        assert b"response.created" in chunks[0]
        assert b"response.output_item.added" in chunks[1]
        assert b"response.content_part.added" in chunks[2]
        assert b"response.output_text.delta" in chunks[3]
        assert b"response.output_text.done" in chunks[4]
        assert b"response.content_part.done" in chunks[5]
        assert b"response.output_item.done" in chunks[6]
        assert b"response.completed" in chunks[7]


# =============================================================================
# Anthropic Messages API passthrough endpoint tests
# =============================================================================


@pytest.mark.asyncio
async def test_anthropic_passthrough_messages(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="anthropic-passthrough-key",
        secret_value={"api_key": "sk-ant-test"},
        provider="anthropic",
    )
    model_def = store.create_gateway_model_definition(
        name="anthropic-passthrough-model",
        secret_id=secret.secret_id,
        provider="anthropic",
        model_name="claude-3-5-sonnet-20241022",
    )
    store.create_gateway_endpoint(
        name="anthropic-passthrough-endpoint",
        model_definition_ids=[model_def.model_definition_id],
    )

    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "model": "anthropic-passthrough-endpoint",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
        }
    )

    mock_response = {
        "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello! How can I assist you today?"}],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }

    with mock.patch(
        "mlflow.gateway.providers.anthropic.send_request", return_value=mock_response
    ) as mock_send:
        response = await anthropic_passthrough_messages(mock_request)

        assert mock_send.called
        call_args = mock_send.call_args
        assert call_args[1]["path"] == "messages"
        assert call_args[1]["payload"]["model"] == "claude-3-5-sonnet-20241022"
        assert call_args[1]["payload"]["messages"] == [{"role": "user", "content": "Hello"}]
        assert call_args[1]["payload"]["max_tokens"] == 1024

        assert response["id"] == "msg_01XFDUDYJgAACzvnptvVoYEL"
        assert response["model"] == "claude-3-5-sonnet-20241022"
        assert response["content"][0]["text"] == "Hello! How can I assist you today?"


@pytest.mark.asyncio
async def test_anthropic_passthrough_messages_streaming(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="anthropic-stream-passthrough-key",
        secret_value={"api_key": "sk-ant-test-stream"},
        provider="anthropic",
    )
    model_def = store.create_gateway_model_definition(
        name="anthropic-stream-passthrough-model",
        secret_id=secret.secret_id,
        provider="anthropic",
        model_name="claude-3-5-sonnet-20241022",
    )
    store.create_gateway_endpoint(
        name="anthropic-stream-passthrough-endpoint",
        model_definition_ids=[model_def.model_definition_id],
    )

    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "model": "anthropic-stream-passthrough-endpoint",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
            "stream": True,
        }
    )

    mock_stream_chunks = [
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_01XFDUDYJgAACzvnptvVoYEL","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}\n\n',  # noqa: E501
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',  # noqa: E501
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',  # noqa: E501
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}\n\n',  # noqa: E501
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":20}}\n\n',  # noqa: E501
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]

    async def mock_stream_generator():
        for chunk in mock_stream_chunks:
            yield chunk

    with mock.patch(
        "mlflow.gateway.providers.anthropic.send_stream_request",
        return_value=mock_stream_generator(),
    ) as mock_send_stream:
        response = await anthropic_passthrough_messages(mock_request)

        assert mock_send_stream.called
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"

        chunks = [chunk async for chunk in response.body_iterator]

        assert len(chunks) == 7
        assert b"message_start" in chunks[0]
        assert b"content_block_start" in chunks[1]
        assert b"content_block_delta" in chunks[2]
        assert b"Hello" in chunks[2]
        assert b"content_block_delta" in chunks[3]
        assert b"!" in chunks[3]
        assert b"content_block_stop" in chunks[4]
        assert b"message_delta" in chunks[5]
        assert b"message_stop" in chunks[6]


# =============================================================================
# Gemini generateContent/streamGenerateContent passthrough endpoint tests
# =============================================================================


@pytest.mark.asyncio
async def test_gemini_passthrough_generate_content(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="gemini-passthrough-key",
        secret_value={"api_key": "test-key"},
        provider="gemini",
    )
    model_def = store.create_gateway_model_definition(
        name="gemini-passthrough-model",
        secret_id=secret.secret_id,
        provider="gemini",
        model_name="gemini-2.0-flash",
    )
    store.create_gateway_endpoint(
        name="gemini-passthrough-endpoint",
        model_definition_ids=[model_def.model_definition_id],
    )

    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "Hello"}],
                }
            ]
        }
    )

    mock_response = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello! How can I assist you today?"}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 10,
            "totalTokenCount": 15,
        },
    }

    with mock.patch(
        "mlflow.gateway.providers.gemini.send_request", return_value=mock_response
    ) as mock_send:
        response = await gemini_passthrough_generate_content(
            "gemini-passthrough-endpoint", mock_request
        )

        assert mock_send.called
        call_args = mock_send.call_args
        assert call_args[1]["path"] == "gemini-2.0-flash:generateContent"
        assert call_args[1]["payload"]["contents"] == [
            {"role": "user", "parts": [{"text": "Hello"}]}
        ]

        assert (
            response["candidates"][0]["content"]["parts"][0]["text"]
            == "Hello! How can I assist you today?"
        )
        assert response["usageMetadata"]["totalTokenCount"] == 15


@pytest.mark.asyncio
async def test_gemini_passthrough_stream_generate_content(store: SqlAlchemyStore):
    secret = store.create_gateway_secret(
        secret_name="gemini-stream-passthrough-key",
        secret_value={"api_key": "test-stream-key"},
        provider="gemini",
    )
    model_def = store.create_gateway_model_definition(
        name="gemini-stream-passthrough-model",
        secret_id=secret.secret_id,
        provider="gemini",
        model_name="gemini-2.0-flash",
    )
    store.create_gateway_endpoint(
        name="gemini-stream-passthrough-endpoint",
        model_definition_ids=[model_def.model_definition_id],
    )

    mock_request = MagicMock()
    mock_request.json = AsyncMock(
        return_value={
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "Hello"}],
                }
            ]
        }
    )

    mock_stream_chunks = [
        b'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}\n\n',
        b'data: {"candidates":[{"content":{"parts":[{"text":"!"}],"role":"model"}}]}\n\n',
        b'data: {"candidates":[{"content":{"parts":[{"text":" How can I help you?"}],"role":"model"},"finishReason":"STOP"}]}\n\n',  # noqa: E501
    ]

    async def mock_stream_generator():
        for chunk in mock_stream_chunks:
            yield chunk

    with mock.patch(
        "mlflow.gateway.providers.gemini.send_stream_request",
        return_value=mock_stream_generator(),
    ) as mock_send_stream:
        response = await gemini_passthrough_stream_generate_content(
            "gemini-stream-passthrough-endpoint", mock_request
        )

        assert mock_send_stream.called
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"

        chunks = [chunk async for chunk in response.body_iterator]

        assert len(chunks) == 3
        assert b"Hello" in chunks[0]
        assert b"!" in chunks[1]
        assert b"How can I help you?" in chunks[2]
        assert b"STOP" in chunks[2]
