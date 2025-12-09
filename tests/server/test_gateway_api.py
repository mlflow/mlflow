"""
Tests for database-backed Gateway API endpoints.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.gateway.config import EndpointType
from mlflow.gateway.schemas import chat, embeddings
from mlflow.server.gateway_api import (
    _create_invocations_handler,
    _create_provider_from_endpoint_config,
    gateway_router,
    register_gateway_endpoints,
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
    secret = store.create_secret(
        secret_name="openai-key",
        secret_value="sk-test-123",
        provider="openai",
        credential_name="OPENAI_API_KEY",
    )
    model_def = store.create_model_definition(
        name="gpt-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4o",
    )
    endpoint = store.create_endpoint(
        name="test-openai-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    # Create provider
    provider, endpoint_type = _create_provider_from_endpoint_config(endpoint.endpoint_id, store)

    assert provider is not None
    assert endpoint_type == EndpointType.LLM_V1_CHAT


def test_create_provider_from_endpoint_config_anthropic(store: SqlAlchemyStore):
    secret = store.create_secret(
        secret_name="anthropic-key",
        secret_value="sk-ant-test",
        provider="anthropic",
    )
    model_def = store.create_model_definition(
        name="claude-model",
        secret_id=secret.secret_id,
        provider="anthropic",
        model_name="claude-3-sonnet",
    )
    endpoint = store.create_endpoint(
        name="test-anthropic-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    provider, endpoint_type = _create_provider_from_endpoint_config(endpoint.endpoint_id, store)

    assert provider is not None
    assert endpoint_type == EndpointType.LLM_V1_CHAT


def test_create_provider_from_endpoint_config_nonexistent_endpoint(store: SqlAlchemyStore):
    from mlflow.exceptions import MlflowException

    with pytest.raises(MlflowException, match="not found"):
        _create_provider_from_endpoint_config("nonexistent-id", store)


def test_register_gateway_endpoints(store: SqlAlchemyStore):
    # Create test endpoints
    secret = store.create_secret(
        secret_name="test-key",
        secret_value="sk-test",
        provider="openai",
    )
    model_def = store.create_model_definition(
        name="test-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    endpoint = store.create_endpoint(
        name="test-endpoint", model_definition_ids=[model_def.model_definition_id]
    )

    # Register endpoints
    router = register_gateway_endpoints(store)

    # Check that routes were registered
    route_paths = [route.path for route in router.routes]
    assert "/gateway/test-endpoint/mlflow/invocations" in route_paths
    # Should NOT have separate embeddings endpoint
    assert "/gateway/test-endpoint/mlflow/embeddings" not in route_paths


@pytest.mark.asyncio
async def test_invocations_handler_chat(store: SqlAlchemyStore):
    # Create test data
    secret = store.create_secret(
        secret_name="chat-key",
        secret_value="sk-test-chat",
        provider="openai",
    )
    model_def = store.create_model_definition(
        name="chat-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    endpoint = store.create_endpoint(
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
        mock_create_provider.return_value = (mock_provider, EndpointType.LLM_V1_CHAT)

        # Call the handler
        response = await handler(mock_request)

        # Verify
        assert response.id == "test-id"
        assert response.choices[0].message.content == "Hello!"
        assert mock_provider.chat.called


@pytest.mark.asyncio
async def test_invocations_handler_embeddings(store: SqlAlchemyStore):
    # Create test data
    secret = store.create_secret(
        secret_name="embed-key",
        secret_value="sk-test-embed",
        provider="openai",
    )
    model_def = store.create_model_definition(
        name="embed-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="text-embedding-ada-002",
    )
    endpoint = store.create_endpoint(
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
        mock_create_provider.return_value = (mock_provider, EndpointType.LLM_V1_EMBEDDINGS)

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
