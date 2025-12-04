from mlflow.entities import (
    GatewayEndpoint,
    GatewayEndpointBinding,
    GatewayEndpointConfig,
    GatewayEndpointModelMapping,
    GatewayModelConfig,
    GatewayModelDefinition,
    GatewayResourceType,
)


def test_model_definition_creation_full():
    model_def = GatewayModelDefinition(
        model_definition_id="model-def-123",
        name="GPT-4o Production",
        secret_id="secret-789",
        secret_name="openai_api_key",
        provider="openai",
        model_name="gpt-4o",
        created_at=1234567890000,
        last_updated_at=1234567890000,
        created_by="test_user",
        last_updated_by="test_user",
    )

    assert model_def.model_definition_id == "model-def-123"
    assert model_def.name == "GPT-4o Production"
    assert model_def.secret_id == "secret-789"
    assert model_def.secret_name == "openai_api_key"
    assert model_def.provider == "openai"
    assert model_def.model_name == "gpt-4o"
    assert model_def.created_at == 1234567890000
    assert model_def.last_updated_at == 1234567890000
    assert model_def.created_by == "test_user"
    assert model_def.last_updated_by == "test_user"


def test_model_definition_creation_minimal():
    model_def = GatewayModelDefinition(
        model_definition_id="model-def-123",
        name="Anthropic Claude",
        secret_id="secret-789",
        secret_name="api_key",
        provider="anthropic",
        model_name="claude-3-5-sonnet-20241022",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    assert model_def.model_definition_id == "model-def-123"
    assert model_def.created_by is None
    assert model_def.last_updated_by is None


def test_model_definition_various_providers():
    providers_and_models = [
        ("openai", "gpt-4o"),
        ("anthropic", "claude-3-5-sonnet-20241022"),
        ("cohere", "command-r-plus"),
        ("bedrock", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
    ]

    for provider, model_name in providers_and_models:
        model_def = GatewayModelDefinition(
            model_definition_id=f"model-def-{provider}",
            name=f"{provider} model",
            secret_id=f"secret-{provider}",
            secret_name=f"{provider}_key",
            provider=provider,
            model_name=model_name,
            created_at=1234567890000,
            last_updated_at=1234567890000,
        )

        assert model_def.provider == provider
        assert model_def.model_name == model_name


def test_endpoint_model_mapping_creation():
    model_def = GatewayModelDefinition(
        model_definition_id="model-def-1",
        name="GPT-4o",
        secret_id="secret-1",
        secret_name="openai_key",
        provider="openai",
        model_name="gpt-4o",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    mapping = GatewayEndpointModelMapping(
        mapping_id="mapping-123",
        endpoint_id="endpoint-456",
        model_definition_id="model-def-1",
        model_definition=model_def,
        weight=1,
        created_at=1234567890000,
        created_by="test_user",
    )

    assert mapping.mapping_id == "mapping-123"
    assert mapping.endpoint_id == "endpoint-456"
    assert mapping.model_definition_id == "model-def-1"
    assert mapping.model_definition is not None
    assert mapping.model_definition.name == "GPT-4o"
    assert mapping.weight == 1
    assert mapping.created_at == 1234567890000
    assert mapping.created_by == "test_user"


def test_endpoint_model_mapping_without_model_definition():
    mapping = GatewayEndpointModelMapping(
        mapping_id="mapping-123",
        endpoint_id="endpoint-456",
        model_definition_id="model-def-1",
        model_definition=None,
        weight=2,
        created_at=1234567890000,
    )

    assert mapping.mapping_id == "mapping-123"
    assert mapping.model_definition is None
    assert mapping.weight == 2
    assert mapping.created_by is None


def test_endpoint_creation_full():
    model_def = GatewayModelDefinition(
        model_definition_id="model-def-1",
        name="GPT-4o",
        secret_id="secret-1",
        secret_name="openai_key",
        provider="openai",
        model_name="gpt-4o",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    mapping = GatewayEndpointModelMapping(
        mapping_id="mapping-1",
        endpoint_id="endpoint-1",
        model_definition_id="model-def-1",
        model_definition=model_def,
        weight=1,
        created_at=1234567890000,
    )

    endpoint = GatewayEndpoint(
        endpoint_id="endpoint-1",
        name="Production LLM Endpoint",
        created_at=1234567890000,
        last_updated_at=1234567890000,
        model_mappings=[mapping],
        created_by="test_user",
        last_updated_by="test_user",
    )

    assert endpoint.endpoint_id == "endpoint-1"
    assert endpoint.name == "Production LLM Endpoint"
    assert endpoint.created_at == 1234567890000
    assert endpoint.last_updated_at == 1234567890000
    assert len(endpoint.model_mappings) == 1
    assert endpoint.model_mappings[0].mapping_id == "mapping-1"
    assert endpoint.created_by == "test_user"
    assert endpoint.last_updated_by == "test_user"


def test_endpoint_creation_minimal():
    endpoint = GatewayEndpoint(
        endpoint_id="endpoint-minimal",
        name="Minimal Endpoint",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    assert endpoint.endpoint_id == "endpoint-minimal"
    assert endpoint.name == "Minimal Endpoint"
    assert endpoint.model_mappings == []
    assert endpoint.created_by is None
    assert endpoint.last_updated_by is None


def test_endpoint_with_multiple_model_mappings():
    model_def1 = GatewayModelDefinition(
        model_definition_id="model-def-1",
        name="GPT-4o",
        secret_id="secret-1",
        secret_name="openai_key",
        provider="openai",
        model_name="gpt-4o",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    model_def2 = GatewayModelDefinition(
        model_definition_id="model-def-2",
        name="Claude Sonnet",
        secret_id="secret-2",
        secret_name="anthropic_key",
        provider="anthropic",
        model_name="claude-3-5-sonnet-20241022",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    mapping1 = GatewayEndpointModelMapping(
        mapping_id="mapping-1",
        endpoint_id="endpoint-multi",
        model_definition_id="model-def-1",
        model_definition=model_def1,
        weight=1,
        created_at=1234567890000,
    )

    mapping2 = GatewayEndpointModelMapping(
        mapping_id="mapping-2",
        endpoint_id="endpoint-multi",
        model_definition_id="model-def-2",
        model_definition=model_def2,
        weight=1,
        created_at=1234567890000,
    )

    endpoint = GatewayEndpoint(
        endpoint_id="endpoint-multi",
        name="Multi-Model Endpoint",
        created_at=1234567890000,
        last_updated_at=1234567890000,
        model_mappings=[mapping1, mapping2],
    )

    assert len(endpoint.model_mappings) == 2
    assert endpoint.model_mappings[0].model_definition.provider == "openai"
    assert endpoint.model_mappings[1].model_definition.provider == "anthropic"
    providers = {m.model_definition.provider for m in endpoint.model_mappings}
    assert providers == {"openai", "anthropic"}


def test_endpoint_binding_creation_full():
    binding = GatewayEndpointBinding(
        endpoint_id="endpoint-456",
        resource_type=GatewayResourceType.SCORER_JOB,
        resource_id="job-789",
        created_at=1234567890000,
        last_updated_at=1234567890000,
        created_by="test_user",
        last_updated_by="test_user",
    )

    assert binding.endpoint_id == "endpoint-456"
    assert binding.resource_type == GatewayResourceType.SCORER_JOB
    assert binding.resource_id == "job-789"
    assert binding.created_at == 1234567890000
    assert binding.last_updated_at == 1234567890000
    assert binding.created_by == "test_user"
    assert binding.last_updated_by == "test_user"


def test_endpoint_binding_creation_minimal():
    binding = GatewayEndpointBinding(
        endpoint_id="endpoint-minimal",
        resource_type=GatewayResourceType.SCORER_JOB,
        resource_id="job-minimal",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    assert binding.created_by is None
    assert binding.last_updated_by is None


def test_endpoint_binding_resource_type_enum():
    binding = GatewayEndpointBinding(
        endpoint_id="endpoint-1",
        resource_type=GatewayResourceType.SCORER_JOB,
        resource_id="job-enum",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    assert binding.resource_type == GatewayResourceType.SCORER_JOB
    assert binding.resource_type.value == "scorer_job"
    assert isinstance(binding.resource_type, GatewayResourceType)


def test_resource_type_enum():
    assert GatewayResourceType.SCORER_JOB == "scorer_job"
    assert GatewayResourceType.SCORER_JOB.value == "scorer_job"
    assert isinstance(GatewayResourceType.SCORER_JOB, str)


def test_resource_type_enum_usage():
    rt = GatewayResourceType.SCORER_JOB
    assert rt == "scorer_job"
    assert rt.value == "scorer_job"
    assert isinstance(rt, str)


def test_model_config_creation():
    model_config = GatewayModelConfig(
        model_definition_id="model-def-123",
        provider="openai",
        model_name="gpt-4o",
        secret_value="sk-proj-abc123xyz",
        credential_name="OPENAI_API_KEY",
    )

    assert model_config.model_definition_id == "model-def-123"
    assert model_config.provider == "openai"
    assert model_config.model_name == "gpt-4o"
    assert model_config.secret_value == "sk-proj-abc123xyz"
    assert model_config.credential_name == "OPENAI_API_KEY"


def test_model_config_without_credential_name():
    model_config = GatewayModelConfig(
        model_definition_id="model-def-456",
        provider="anthropic",
        model_name="claude-3-5-sonnet-20241022",
        secret_value="sk-ant-api-xyz789",
    )

    assert model_config.model_definition_id == "model-def-456"
    assert model_config.provider == "anthropic"
    assert model_config.secret_value == "sk-ant-api-xyz789"
    assert model_config.credential_name is None
    assert model_config.auth_config is None


def test_model_config_with_auth_config():
    auth_config = {
        "project_id": "my-gcp-project",
        "region": "us-central1",
        "credentials_path": "/path/to/creds.json",
    }

    model_config = GatewayModelConfig(
        model_definition_id="model-def-789",
        provider="vertexai",
        model_name="gemini-pro",
        secret_value='{"type": "service_account", "project_id": "..."}',
        credential_name="VERTEX_AI_CREDENTIALS",
        auth_config=auth_config,
    )

    assert model_config.model_definition_id == "model-def-789"
    assert model_config.provider == "vertexai"
    assert model_config.auth_config == auth_config
    assert model_config.auth_config["project_id"] == "my-gcp-project"
    assert model_config.auth_config["region"] == "us-central1"


def test_endpoint_config_creation():
    model1 = GatewayModelConfig(
        model_definition_id="model-def-1",
        provider="openai",
        model_name="gpt-4o",
        secret_value="sk-proj-abc123",
        credential_name="OPENAI_API_KEY",
    )

    model2 = GatewayModelConfig(
        model_definition_id="model-def-2",
        provider="vertexai",
        model_name="gemini-pro",
        secret_value='{"type": "service_account"}',
        credential_name="VERTEX_AI_CREDENTIALS",
        auth_config={"project_id": "my-project", "region": "us-central1"},
    )

    endpoint_config = GatewayEndpointConfig(
        endpoint_id="endpoint-123",
        endpoint_name="Production LLM Endpoint",
        models=[model1, model2],
    )

    assert endpoint_config.endpoint_id == "endpoint-123"
    assert endpoint_config.endpoint_name == "Production LLM Endpoint"
    assert len(endpoint_config.models) == 2
    assert endpoint_config.models[0].provider == "openai"
    assert endpoint_config.models[0].secret_value == "sk-proj-abc123"
    assert endpoint_config.models[0].auth_config is None
    assert endpoint_config.models[1].provider == "vertexai"
    assert endpoint_config.models[1].secret_value == '{"type": "service_account"}'
    assert endpoint_config.models[1].auth_config is not None
    assert endpoint_config.models[1].auth_config["project_id"] == "my-project"


def test_endpoint_config_minimal():
    endpoint_config = GatewayEndpointConfig(
        endpoint_id="endpoint-minimal",
        endpoint_name="Minimal Endpoint",
    )

    assert endpoint_config.endpoint_id == "endpoint-minimal"
    assert endpoint_config.endpoint_name == "Minimal Endpoint"
    assert endpoint_config.models == []


def test_endpoint_config_decrypted_secrets():
    model1 = GatewayModelConfig(
        model_definition_id="model-def-1",
        provider="openai",
        model_name="gpt-4o",
        secret_value="sk-proj-decrypted-value-123",
    )

    model2 = GatewayModelConfig(
        model_definition_id="model-def-2",
        provider="cohere",
        model_name="command-r-plus",
        secret_value="co-decrypted-value-456",
    )

    endpoint_config = GatewayEndpointConfig(
        endpoint_id="endpoint-secure",
        endpoint_name="Secure Endpoint",
        models=[model1, model2],
    )

    assert all(hasattr(m, "secret_value") for m in endpoint_config.models)
    assert endpoint_config.models[0].secret_value == "sk-proj-decrypted-value-123"
    assert endpoint_config.models[1].secret_value == "co-decrypted-value-456"
