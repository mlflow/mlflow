from mlflow.entities import Endpoint, EndpointListItem, EndpointTag


def test_endpoint_creation():
    endpoint = Endpoint(
        endpoint_id="test-endpoint-id",
        secret_id="test-secret-id",
        model_name="gpt-4-turbo",
        created_at=1234567890000,
        last_updated_at=1234567890000,
        name="Test Endpoint",
        description="A test endpoint",
        created_by="test_user",
        last_updated_by="test_user",
        tags=[],
    )

    assert endpoint.endpoint_id == "test-endpoint-id"
    assert endpoint.secret_id == "test-secret-id"
    assert endpoint.model_name == "gpt-4-turbo"
    assert endpoint.created_at == 1234567890000
    assert endpoint.last_updated_at == 1234567890000
    assert endpoint.name == "Test Endpoint"
    assert endpoint.description == "A test endpoint"
    assert endpoint.created_by == "test_user"
    assert endpoint.last_updated_by == "test_user"
    assert endpoint.tags == []


def test_endpoint_with_tags():
    tag1 = EndpointTag(key="env", value="prod")
    tag2 = EndpointTag(key="team", value="ml-platform")

    endpoint = Endpoint(
        endpoint_id="test-endpoint-id",
        secret_id="test-secret-id",
        model_name="claude-3-5-sonnet-20241022",
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
        secret_id="test-secret-id",
        model_name="gpt-4",
        created_at=1234567890000,
        last_updated_at=1234567890000,
    )

    assert endpoint.endpoint_id == "test-endpoint-id"
    assert endpoint.name is None
    assert endpoint.description is None
    assert endpoint.created_by is None
    assert endpoint.last_updated_by is None
    assert endpoint.tags == []


def test_endpoint_list_item():
    list_item = EndpointListItem(
        endpoint_id="test-endpoint-id",
        secret_id="test-secret-id",
        model_name="gpt-4-turbo",
        created_at=1234567890000,
        last_updated_at=1234567890000,
        name="Test Endpoint",
        description="A test endpoint",
        secret_name="my_openai_key",
        provider="openai",
    )

    assert list_item.endpoint_id == "test-endpoint-id"
    assert list_item.secret_id == "test-secret-id"
    assert list_item.model_name == "gpt-4-turbo"
    assert list_item.secret_name == "my_openai_key"
    assert list_item.provider == "openai"
    assert isinstance(list_item, Endpoint)


def test_endpoint_tag():
    tag = EndpointTag(key="environment", value="production")

    assert tag.key == "environment"
    assert tag.value == "production"


def test_endpoint_tag_with_none_value():
    tag = EndpointTag(key="optional_key", value=None)

    assert tag.key == "optional_key"
    assert tag.value is None
