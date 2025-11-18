from mlflow.entities import (
    Endpoint,
    EndpointTag,
    Secret,
    SecretBinding,
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
        secret_id="test-secret-id",
        model_name="gpt-4",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    binding = SecretBinding(
        binding_id="test-binding-id",
        endpoint_id="test-endpoint-id",
        secret_id="test-secret-id",
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
    assert wrapper.endpoint.model_name == "gpt-4"
    assert wrapper.binding.resource_type == "GLOBAL"
