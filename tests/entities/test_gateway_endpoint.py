from mlflow.entities import (
    GatewayEndpoint,
    GatewayEndpointBinding,
    GatewayEndpointModelMapping,
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


def test_model_definition_proto_round_trip():
    model_def = GatewayModelDefinition(
        model_definition_id="model-def-proto",
        name="Proto Test Model",
        secret_id="secret-proto",
        secret_name="proto_key",
        provider="openai",
        model_name="gpt-4o",
        created_at=1234567890000,
        last_updated_at=1234567891000,
        created_by="proto_user",
        last_updated_by="proto_user_2",
    )

    proto = model_def.to_proto()
    restored = GatewayModelDefinition.from_proto(proto)

    assert restored.model_definition_id == model_def.model_definition_id
    assert restored.name == model_def.name
    assert restored.secret_id == model_def.secret_id
    assert restored.secret_name == model_def.secret_name
    assert restored.provider == model_def.provider
    assert restored.model_name == model_def.model_name
    assert restored.created_at == model_def.created_at
    assert restored.last_updated_at == model_def.last_updated_at
    assert restored.created_by == model_def.created_by
    assert restored.last_updated_by == model_def.last_updated_by


def test_endpoint_model_mapping_proto_round_trip():
    mapping = GatewayEndpointModelMapping(
        mapping_id="mapping-proto",
        endpoint_id="endpoint-proto",
        model_definition_id="model-def-proto",
        model_definition=None,
        weight=2,
        created_at=1234567890000,
        created_by="mapping_user",
    )

    proto = mapping.to_proto()
    restored = GatewayEndpointModelMapping.from_proto(proto)

    assert restored.mapping_id == mapping.mapping_id
    assert restored.endpoint_id == mapping.endpoint_id
    assert restored.model_definition_id == mapping.model_definition_id
    assert restored.weight == mapping.weight
    assert restored.created_at == mapping.created_at
    assert restored.created_by == mapping.created_by


def test_endpoint_proto_round_trip():
    endpoint = GatewayEndpoint(
        endpoint_id="endpoint-proto",
        name="Proto Test Endpoint",
        created_at=1234567890000,
        last_updated_at=1234567891000,
        model_mappings=[],
        created_by="endpoint_user",
        last_updated_by="endpoint_user_2",
    )

    proto = endpoint.to_proto()
    restored = GatewayEndpoint.from_proto(proto)

    assert restored.endpoint_id == endpoint.endpoint_id
    assert restored.name == endpoint.name
    assert restored.created_at == endpoint.created_at
    assert restored.last_updated_at == endpoint.last_updated_at
    assert restored.created_by == endpoint.created_by
    assert restored.last_updated_by == endpoint.last_updated_by
    assert len(restored.model_mappings) == 0


def test_endpoint_binding_proto_round_trip():
    binding = GatewayEndpointBinding(
        endpoint_id="endpoint-proto",
        resource_type=GatewayResourceType.SCORER_JOB,
        resource_id="job-proto",
        created_at=1234567890000,
        last_updated_at=1234567891000,
        created_by="binding_user",
        last_updated_by="binding_user_2",
    )

    proto = binding.to_proto()
    restored = GatewayEndpointBinding.from_proto(proto)

    assert restored.endpoint_id == binding.endpoint_id
    assert restored.resource_type == binding.resource_type
    assert restored.resource_id == binding.resource_id
    assert restored.created_at == binding.created_at
    assert restored.last_updated_at == binding.last_updated_at
    assert restored.created_by == binding.created_by
    assert restored.last_updated_by == binding.last_updated_by
