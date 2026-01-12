"""Tests for Gateway Rate Limit storage operations."""

import pytest

from mlflow.entities import GatewayEndpointModelConfig, GatewayModelLinkageType
from mlflow.entities.gateway_rate_limit import GatewayRateLimitConfig
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore


@pytest.fixture
def store(tmp_path):
    """Create a temporary SQLAlchemy store for testing."""
    db_path = tmp_path / "test.db"
    store = SqlAlchemyStore(f"sqlite:///{db_path}", str(tmp_path / "artifacts"))
    return store


def _create_test_endpoint(store, name="test-endpoint"):
    """Helper to create a test endpoint with prerequisite objects."""
    secret = store.create_gateway_secret(
        secret_name=f"{name}-key",
        secret_value={"api_key": "test-value"},
    )
    model_def = store.create_gateway_model_definition(
        name=f"{name}-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )
    endpoint = store.create_gateway_endpoint(
        name=name,
        model_configs=[
            GatewayEndpointModelConfig(
                model_definition_id=model_def.model_definition_id,
                linkage_type=GatewayModelLinkageType.PRIMARY,
                weight=1.0,
            ),
        ],
    )
    return endpoint.endpoint_id


@pytest.fixture
def endpoint_id(store):
    """Create a test endpoint and return its ID."""
    return _create_test_endpoint(store)


class TestGatewayRateLimitStorage:
    """Tests for gateway rate limit storage operations."""

    def test_create_gateway_rate_limit_default(self, store, endpoint_id):
        """Test creating a default rate limit for an endpoint."""
        rate_limit = store.create_gateway_rate_limit(
            endpoint_id=endpoint_id,
            queries_per_minute=100,
            username=None,
            created_by="admin",
        )

        assert rate_limit.rate_limit_id.startswith("rl-")
        assert rate_limit.endpoint_id == endpoint_id
        assert rate_limit.queries_per_minute == 100
        assert rate_limit.username is None
        assert rate_limit.is_default is True
        assert rate_limit.created_by == "admin"

    def test_create_gateway_rate_limit_per_user(self, store, endpoint_id):
        """Test creating a per-user rate limit."""
        rate_limit = store.create_gateway_rate_limit(
            endpoint_id=endpoint_id,
            queries_per_minute=50,
            username="testuser",
            created_by="admin",
        )

        assert rate_limit.username == "testuser"
        assert rate_limit.is_default is False
        assert rate_limit.queries_per_minute == 50

    def test_create_gateway_rate_limit_invalid_qpm(self, store, endpoint_id):
        """Test that creating a rate limit with invalid QPM fails."""
        with pytest.raises(MlflowException, match="positive integer"):
            store.create_gateway_rate_limit(
                endpoint_id=endpoint_id,
                queries_per_minute=0,
            )

        with pytest.raises(MlflowException, match="positive integer"):
            store.create_gateway_rate_limit(
                endpoint_id=endpoint_id,
                queries_per_minute=-10,
            )

    def test_create_gateway_rate_limit_duplicate(self, store, endpoint_id):
        """Test that creating a duplicate rate limit fails."""
        store.create_gateway_rate_limit(
            endpoint_id=endpoint_id,
            queries_per_minute=100,
            username=None,
        )

        with pytest.raises(MlflowException, match="already exists"):
            store.create_gateway_rate_limit(
                endpoint_id=endpoint_id,
                queries_per_minute=200,
                username=None,
            )

    def test_get_gateway_rate_limit(self, store, endpoint_id):
        """Test retrieving a rate limit by ID."""
        created = store.create_gateway_rate_limit(
            endpoint_id=endpoint_id,
            queries_per_minute=100,
        )

        retrieved = store.get_gateway_rate_limit(created.rate_limit_id)

        assert retrieved is not None
        assert retrieved.rate_limit_id == created.rate_limit_id
        assert retrieved.queries_per_minute == 100

    def test_get_gateway_rate_limit_not_found(self, store):
        """Test retrieving a non-existent rate limit."""
        retrieved = store.get_gateway_rate_limit("non-existent-id")
        assert retrieved is None

    def test_get_gateway_rate_limit_for_user_specific(self, store, endpoint_id):
        """Test getting rate limit for a specific user."""
        # Create default limit
        store.create_gateway_rate_limit(
            endpoint_id=endpoint_id,
            queries_per_minute=100,
            username=None,
        )
        # Create user-specific limit
        store.create_gateway_rate_limit(
            endpoint_id=endpoint_id,
            queries_per_minute=50,
            username="testuser",
        )

        # User should get their specific limit
        user_limit = store.get_gateway_rate_limit_for_user(endpoint_id, "testuser")
        assert user_limit.queries_per_minute == 50
        assert user_limit.username == "testuser"

    def test_get_gateway_rate_limit_for_user_fallback(self, store, endpoint_id):
        """Test getting rate limit falls back to default for users without specific limit."""
        # Create only default limit
        store.create_gateway_rate_limit(
            endpoint_id=endpoint_id,
            queries_per_minute=100,
            username=None,
        )

        # User should get default limit
        user_limit = store.get_gateway_rate_limit_for_user(endpoint_id, "otheruser")
        assert user_limit.queries_per_minute == 100
        assert user_limit.username is None

    def test_get_gateway_rate_limit_for_user_none(self, store, endpoint_id):
        """Test getting rate limit when no limits are configured."""
        limit = store.get_gateway_rate_limit_for_user(endpoint_id, "anyuser")
        assert limit is None

    def test_list_gateway_rate_limits(self, store, endpoint_id):
        """Test listing all rate limits."""
        store.create_gateway_rate_limit(endpoint_id=endpoint_id, queries_per_minute=100)
        store.create_gateway_rate_limit(
            endpoint_id=endpoint_id, queries_per_minute=50, username="user1"
        )
        store.create_gateway_rate_limit(
            endpoint_id=endpoint_id, queries_per_minute=25, username="user2"
        )

        rate_limits = store.list_gateway_rate_limits()
        assert len(rate_limits) == 3

    def test_list_gateway_rate_limits_by_endpoint(self, store, endpoint_id):
        """Test filtering rate limits by endpoint."""
        # Create another endpoint
        endpoint2_id = _create_test_endpoint(store, name="test-endpoint-2")

        store.create_gateway_rate_limit(endpoint_id=endpoint_id, queries_per_minute=100)
        store.create_gateway_rate_limit(endpoint_id=endpoint2_id, queries_per_minute=200)

        rate_limits = store.list_gateway_rate_limits(endpoint_id=endpoint_id)
        assert len(rate_limits) == 1
        assert rate_limits[0].endpoint_id == endpoint_id

    def test_list_gateway_rate_limits_defaults_only(self, store, endpoint_id):
        """Test filtering for only default rate limits."""
        store.create_gateway_rate_limit(endpoint_id=endpoint_id, queries_per_minute=100)
        store.create_gateway_rate_limit(
            endpoint_id=endpoint_id, queries_per_minute=50, username="user1"
        )

        rate_limits = store.list_gateway_rate_limits(include_defaults_only=True)
        assert len(rate_limits) == 1
        assert rate_limits[0].username is None

    def test_update_gateway_rate_limit(self, store, endpoint_id):
        """Test updating a rate limit."""
        created = store.create_gateway_rate_limit(
            endpoint_id=endpoint_id,
            queries_per_minute=100,
            created_by="admin",
        )

        updated = store.update_gateway_rate_limit(
            rate_limit_id=created.rate_limit_id,
            queries_per_minute=200,
            updated_by="admin2",
        )

        assert updated.queries_per_minute == 200
        assert updated.updated_by == "admin2"
        assert updated.updated_at >= created.updated_at

    def test_update_gateway_rate_limit_invalid_qpm(self, store, endpoint_id):
        """Test that updating with invalid QPM fails."""
        created = store.create_gateway_rate_limit(
            endpoint_id=endpoint_id,
            queries_per_minute=100,
        )

        with pytest.raises(MlflowException, match="positive integer"):
            store.update_gateway_rate_limit(
                rate_limit_id=created.rate_limit_id,
                queries_per_minute=0,
            )

    def test_update_gateway_rate_limit_not_found(self, store):
        """Test updating a non-existent rate limit."""
        with pytest.raises(MlflowException, match="not found"):
            store.update_gateway_rate_limit(
                rate_limit_id="non-existent-id",
                queries_per_minute=100,
            )

    def test_delete_gateway_rate_limit(self, store, endpoint_id):
        """Test deleting a rate limit."""
        created = store.create_gateway_rate_limit(
            endpoint_id=endpoint_id,
            queries_per_minute=100,
        )

        store.delete_gateway_rate_limit(created.rate_limit_id)

        retrieved = store.get_gateway_rate_limit(created.rate_limit_id)
        assert retrieved is None

    def test_delete_gateway_rate_limit_not_found(self, store):
        """Test deleting a non-existent rate limit."""
        with pytest.raises(MlflowException, match="not found"):
            store.delete_gateway_rate_limit("non-existent-id")

    def test_rate_limits_cascade_on_endpoint_delete(self, store, endpoint_id):
        """Test that rate limits are deleted when endpoint is deleted."""
        rate_limit = store.create_gateway_rate_limit(
            endpoint_id=endpoint_id,
            queries_per_minute=100,
        )

        # Delete the endpoint
        store.delete_gateway_endpoint(endpoint_id)

        # Rate limit should be gone
        retrieved = store.get_gateway_rate_limit(rate_limit.rate_limit_id)
        assert retrieved is None
