from mlflow.entities import GatewaySecretInfo


def test_secret_creation_full():
    secret = GatewaySecretInfo(
        secret_id="test-secret-id",
        secret_name="my_api_key",
        masked_value="sk-...abc123",
        created_at=1234567890000,
        last_updated_at=1234567890000,
        provider="openai",
        created_by="test_user",
        last_updated_by="test_user",
    )

    assert secret.secret_id == "test-secret-id"
    assert secret.secret_name == "my_api_key"
    assert secret.masked_value == "sk-...abc123"
    assert secret.created_at == 1234567890000
    assert secret.last_updated_at == 1234567890000
    assert secret.provider == "openai"
    assert secret.created_by == "test_user"
    assert secret.last_updated_by == "test_user"


def test_secret_creation_minimal():
    secret = GatewaySecretInfo(
        secret_id="minimal-secret-id",
        secret_name="minimal_key",
        masked_value="key-...xyz",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    assert secret.secret_id == "minimal-secret-id"
    assert secret.secret_name == "minimal_key"
    assert secret.masked_value == "key-...xyz"
    assert secret.created_at == 1234567890000
    assert secret.last_updated_at == 1234567890000
    assert secret.provider is None
    assert secret.created_by is None
    assert secret.last_updated_by is None


def test_secret_with_provider():
    providers = ["openai", "anthropic", "cohere", "bedrock"]

    for provider in providers:
        secret = GatewaySecretInfo(
            secret_id=f"{provider}-secret-id",
            secret_name=f"{provider}_key",
            masked_value=f"key-...{provider}",
            created_at=1234567890000,
            last_updated_at=1234567890000,
            provider=provider,
        )

        assert secret.provider == provider
        assert secret.secret_name == f"{provider}_key"


def test_secret_masked_value_formats():
    test_cases = [
        "sk-...abc123",
        "***",
        "AKIAIO...EXAMPLE",
        "glpat-...xyz",
    ]

    for masked_value in test_cases:
        secret = GatewaySecretInfo(
            secret_id="test-id",
            secret_name="test_key",
            masked_value=masked_value,
            created_at=1234567890000,
            last_updated_at=1234567890000,
        )

        assert secret.masked_value == masked_value


def test_secret_audit_fields():
    secret = GatewaySecretInfo(
        secret_id="audit-secret-id",
        secret_name="audit_key",
        masked_value="key-...audit",
        created_at=1234567890000,
        last_updated_at=9876543210000,
        created_by="user_1",
        last_updated_by="user_2",
    )

    assert secret.created_at == 1234567890000
    assert secret.last_updated_at == 9876543210000
    assert secret.created_by == "user_1"
    assert secret.last_updated_by == "user_2"


def test_secret_proto_round_trip():
    secret = GatewaySecretInfo(
        secret_id="secret-proto",
        secret_name="proto_api_key",
        masked_value="sk-...proto",
        created_at=1234567890000,
        last_updated_at=1234567891000,
        provider="openai",
        created_by="proto_user",
        last_updated_by="proto_user_2",
    )

    proto = secret.to_proto()
    restored = GatewaySecretInfo.from_proto(proto)

    assert restored.secret_id == secret.secret_id
    assert restored.secret_name == secret.secret_name
    assert restored.masked_value == secret.masked_value
    assert restored.created_at == secret.created_at
    assert restored.last_updated_at == secret.last_updated_at
    assert restored.provider == secret.provider
    assert restored.created_by == secret.created_by
    assert restored.last_updated_by == secret.last_updated_by


def test_secret_with_auth_config():
    auth_config = {"region": "us-west-2", "project_id": "my-project"}
    secret = GatewaySecretInfo(
        secret_id="auth-config-secret",
        secret_name="bedrock_key",
        masked_value="key-...bedrock",
        created_at=1234567890000,
        last_updated_at=1234567890000,
        provider="bedrock",
        auth_config=auth_config,
    )

    assert secret.auth_config == auth_config
    assert secret.auth_config["region"] == "us-west-2"
    assert secret.auth_config["project_id"] == "my-project"


def test_secret_auth_config_proto_round_trip():
    auth_config = {"region": "eu-central-1", "api_version": "2024-01"}
    secret = GatewaySecretInfo(
        secret_id="auth-config-proto",
        secret_name="config_key",
        masked_value="key-...config",
        created_at=1234567890000,
        last_updated_at=1234567891000,
        provider="anthropic",
        auth_config=auth_config,
        created_by="config_user",
        last_updated_by="config_user",
    )

    proto = secret.to_proto()
    restored = GatewaySecretInfo.from_proto(proto)

    assert restored.auth_config == secret.auth_config
    assert restored.auth_config["region"] == "eu-central-1"
    assert restored.auth_config["api_version"] == "2024-01"
