from mlflow.entities import SecretBinding, SecretBindingListItem, SecretResourceType


def test_secret_binding_creation():
    binding = SecretBinding(
        binding_id="test-binding-id",
        endpoint_id="test-endpoint-id",
        secret_id="test-secret-id",
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
    assert binding.secret_id == "test-secret-id"
    assert binding.resource_type == "GLOBAL"
    assert binding.resource_id == "workspace"
    assert binding.field_name == "OPENAI_API_KEY"
    assert binding.created_at == 1234567890000
    assert binding.last_updated_at == 1234567890000
    assert binding.created_by == "test_user"
    assert binding.last_updated_by == "test_user"


def test_secret_binding_minimal():
    binding = SecretBinding(
        binding_id="test-binding-id",
        endpoint_id="test-endpoint-id",
        secret_id="test-secret-id",
        resource_type="SCORER_JOB",
        resource_id="job_123",
        field_name="llm_api_key",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    assert binding.created_by is None
    assert binding.last_updated_by is None


def test_secret_binding_list_item():
    list_item = SecretBindingListItem(
        binding_id="test-binding-id",
        endpoint_id="test-endpoint-id",
        secret_id="test-secret-id",
        resource_type="SCORER_JOB",
        resource_id="job_123",
        field_name="llm_api_key",
        created_at=1234567890000,
        last_updated_at=1234567890000,
        secret_name="my_openai_key",
        route_name="GPT-4 Turbo",
        provider="openai",
    )

    assert list_item.binding_id == "test-binding-id"
    assert list_item.secret_name == "my_openai_key"
    assert list_item.route_name == "GPT-4 Turbo"
    assert list_item.provider == "openai"
    assert isinstance(list_item, SecretBinding)


def test_secret_resource_type_enum():
    assert SecretResourceType.GLOBAL == "GLOBAL"
    assert SecretResourceType.SCORER_JOB == "SCORER_JOB"


def test_secret_binding_with_different_resource_types():
    binding1 = SecretBinding(
        binding_id="binding-1",
        endpoint_id="endpoint-1",
        secret_id="secret-1",
        resource_type=SecretResourceType.GLOBAL,
        resource_id="workspace",
        field_name="API_KEY",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    binding2 = SecretBinding(
        binding_id="binding-2",
        endpoint_id="endpoint-1",
        secret_id="secret-1",
        resource_type=SecretResourceType.SCORER_JOB,
        resource_id="job_456",
        field_name="llm_key",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    assert binding1.resource_type == "GLOBAL"
    assert binding2.resource_type == "SCORER_JOB"
    assert binding1.endpoint_id == binding2.endpoint_id
    assert binding1.secret_id == binding2.secret_id
