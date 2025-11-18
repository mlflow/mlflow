"""
Tests for secret management entities: Secret, Endpoint, and EndpointBinding.

These entities form the core of MLflow's secret management system:
- Secret: Stores encrypted credentials (API keys)
- Endpoint: Maps secrets to model configurations
- EndpointBinding: Binds endpoints to resources (jobs, workspaces)
"""

from mlflow.entities import (
    Endpoint,
    EndpointBinding,
    EndpointBindingListItem,
    EndpointModel,
    EndpointTag,
    Secret,
    SecretResourceType,
    SecretTag,
    SecretWithEndpointAndBinding,
)


def test_secret_creation():
    secret = Secret(
        secret_id="test-secret-id",
        secret_name="my_api_key",
        masked_value="sk-...abc123",
        is_shared=True,
        created_at=1234567890000,
        last_updated_at=1234567890000,
        created_by="test_user",
        last_updated_by="test_user",
        provider="openai",
        tags=[],
    )

    assert secret.secret_id == "test-secret-id"
    assert secret.secret_name == "my_api_key"
    assert secret.masked_value == "sk-...abc123"
    assert secret.is_shared is True
    assert secret.created_at == 1234567890000
    assert secret.last_updated_at == 1234567890000
    assert secret.created_by == "test_user"
    assert secret.last_updated_by == "test_user"
    assert secret.provider == "openai"
    assert secret.tags == []


def test_secret_with_tags():
    tag1 = SecretTag(key="team", value="ml-platform")
    tag2 = SecretTag(key="cost-center", value="engineering")

    secret = Secret(
        secret_id="test-secret-id",
        secret_name="shared_key",
        masked_value="sk-...xyz",
        is_shared=True,
        created_at=1234567890000,
        last_updated_at=1234567890000,
        tags=[tag1, tag2],
    )

    assert len(secret.tags) == 2
    assert secret.tags[0].key == "team"
    assert secret.tags[0].value == "ml-platform"
    assert secret.tags[1].key == "cost-center"
    assert secret.tags[1].value == "engineering"


def test_secret_private():
    secret = Secret(
        secret_id="private-secret-id",
        secret_name="private_key",
        masked_value="key-...123",
        is_shared=False,
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    assert secret.is_shared is False
    assert secret.created_by is None
    assert secret.last_updated_by is None
    assert secret.provider is None


def test_secret_tag():
    tag = SecretTag(key="environment", value="production")

    assert tag.key == "environment"
    assert tag.value == "production"


def test_endpoint_creation():
    endpoint = Endpoint(
        endpoint_id="test-endpoint-id",
        created_at=1234567890000,
        last_updated_at=1234567890000,
        name="Test Endpoint",
        description="A test endpoint",
        created_by="test_user",
        last_updated_by="test_user",
        models=[],
        tags=[],
    )

    assert endpoint.endpoint_id == "test-endpoint-id"
    assert endpoint.created_at == 1234567890000
    assert endpoint.last_updated_at == 1234567890000
    assert endpoint.name == "Test Endpoint"
    assert endpoint.description == "A test endpoint"
    assert endpoint.created_by == "test_user"
    assert endpoint.last_updated_by == "test_user"
    assert endpoint.models == []
    assert endpoint.tags == []


def test_endpoint_with_tags():
    tag1 = EndpointTag(key="env", value="prod")
    tag2 = EndpointTag(key="team", value="ml-platform")

    endpoint = Endpoint(
        endpoint_id="test-endpoint-id",
        created_at=1234567890000,
        last_updated_at=1234567890000,
        tags=[tag1, tag2],
    )

    assert len(endpoint.tags) == 2
    assert endpoint.tags[0].key == "env"
    assert endpoint.tags[0].value == "prod"
    assert endpoint.tags[1].key == "team"
    assert endpoint.tags[1].value == "ml-platform"


def test_endpoint_minimal():
    endpoint = Endpoint(
        endpoint_id="test-endpoint-id",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    assert endpoint.endpoint_id == "test-endpoint-id"
    assert endpoint.name is None
    assert endpoint.description is None
    assert endpoint.created_by is None
    assert endpoint.last_updated_by is None
    assert endpoint.models == []
    assert endpoint.tags == []


def test_endpoint_tag():
    tag = EndpointTag(key="environment", value="production")

    assert tag.key == "environment"
    assert tag.value == "production"


def test_endpoint_tag_with_none_value():
    tag = EndpointTag(key="optional_key", value=None)

    assert tag.key == "optional_key"
    assert tag.value is None


def test_secret_binding_creation():
    binding = EndpointBinding(
        binding_id="test-binding-id",
        endpoint_id="test-endpoint-id",
        resource_type="GLOBAL",
        resource_id="workspace",
        field_name="OPENAI_API_KEY",
        created_at=1234567890000,
        last_updated_at=1234567890000,
        created_by="test_user",
        last_updated_by="test_user",
    )

    assert binding.binding_id == "test-binding-id"
    assert binding.endpoint_id == "test-endpoint-id"
    assert binding.resource_type == "GLOBAL"
    assert binding.resource_id == "workspace"
    assert binding.field_name == "OPENAI_API_KEY"
    assert binding.created_at == 1234567890000
    assert binding.last_updated_at == 1234567890000
    assert binding.created_by == "test_user"
    assert binding.last_updated_by == "test_user"


def test_secret_binding_minimal():
    binding = EndpointBinding(
        binding_id="test-binding-id",
        endpoint_id="test-endpoint-id",
        resource_type="SCORER_JOB",
        resource_id="job_123",
        field_name="llm_api_key",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    assert binding.created_by is None
    assert binding.last_updated_by is None


def test_secret_binding_list_item():
    model1 = EndpointModel(
        model_id="model-1",
        endpoint_id="test-endpoint-id",
        model_name="gpt-4-turbo",
        secret_id="secret-1",
        routing_config='{"weight": 0.6}',
        created_at=1234567890000,
        last_updated_at=1234567890000,
        secret_name="openai_key",
        provider="openai",
    )
    model2 = EndpointModel(
        model_id="model-2",
        endpoint_id="test-endpoint-id",
        model_name="claude-3-5-sonnet",
        secret_id="secret-2",
        routing_config='{"weight": 0.4}',
        created_at=1234567890000,
        last_updated_at=1234567890000,
        secret_name="anthropic_key",
        provider="anthropic",
    )

    list_item = EndpointBindingListItem(
        binding_id="test-binding-id",
        endpoint_id="test-endpoint-id",
        resource_type="SCORER_JOB",
        resource_id="job_123",
        field_name="llm_api_key",
        created_at=1234567890000,
        last_updated_at=1234567890000,
        endpoint_name="GPT-4 Turbo Endpoint",
        endpoint_description="Multi-model endpoint",
        models=[model1, model2],
    )

    assert list_item.binding_id == "test-binding-id"
    assert list_item.endpoint_name == "GPT-4 Turbo Endpoint"
    assert list_item.endpoint_description == "Multi-model endpoint"
    assert len(list_item.models) == 2
    assert list_item.models[0].model_name == "gpt-4-turbo"
    assert list_item.models[0].provider == "openai"
    assert list_item.models[0].routing_config == '{"weight": 0.6}'
    assert list_item.models[1].model_name == "claude-3-5-sonnet"
    assert list_item.models[1].provider == "anthropic"
    assert list_item.models[1].routing_config == '{"weight": 0.4}'
    assert {m.provider for m in list_item.models if m.provider} == {"openai", "anthropic"}
    assert isinstance(list_item, EndpointBinding)


def test_secret_resource_type_enum():
    assert SecretResourceType.GLOBAL == "GLOBAL"
    assert SecretResourceType.SCORER_JOB == "SCORER_JOB"


def test_secret_binding_with_different_resource_types():
    binding1 = EndpointBinding(
        binding_id="binding-1",
        endpoint_id="endpoint-1",
        resource_type=SecretResourceType.GLOBAL,
        resource_id="workspace",
        field_name="API_KEY",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    binding2 = EndpointBinding(
        binding_id="binding-2",
        endpoint_id="endpoint-1",
        resource_type=SecretResourceType.SCORER_JOB,
        resource_id="job_456",
        field_name="llm_key",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    assert binding1.resource_type == "GLOBAL"
    assert binding2.resource_type == "SCORER_JOB"
    assert binding1.endpoint_id == binding2.endpoint_id


def test_secret_with_endpoint_and_binding():
    secret = Secret(
        secret_id="test-secret-id",
        secret_name="my_key",
        masked_value="sk-...abc",
        is_shared=True,
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    endpoint = Endpoint(
        endpoint_id="test-endpoint-id",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    binding = EndpointBinding(
        binding_id="test-binding-id",
        endpoint_id="test-endpoint-id",
        resource_type="GLOBAL",
        resource_id="workspace",
        field_name="OPENAI_API_KEY",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    wrapper = SecretWithEndpointAndBinding(
        secret=secret,
        endpoint=endpoint,
        binding=binding,
    )

    assert wrapper.secret.secret_id == "test-secret-id"
    assert wrapper.endpoint.endpoint_id == "test-endpoint-id"
    assert wrapper.binding.binding_id == "test-binding-id"
    assert wrapper.secret.secret_name == "my_key"
    assert wrapper.binding.resource_type == "GLOBAL"
