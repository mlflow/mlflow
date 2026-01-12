"""Tests for Gateway Rate Limit entity classes."""

import pytest

from mlflow.entities.gateway_rate_limit import (
    GatewayRateLimitConfig,
    GatewayRateLimitInput,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore


@pytest.fixture
def store(tmp_path):
    """Create a temporary SQLAlchemy store for testing."""
    db_path = tmp_path / "test.db"
    store = SqlAlchemyStore(f"sqlite:///{db_path}", str(tmp_path / "artifacts"))
    return store


class TestGatewayRateLimitEntities:
    """Tests for gateway rate limit entity classes."""

    def test_gateway_rate_limit_config_entity(self):
        config = GatewayRateLimitConfig(
            rate_limit_id="rl-123",
            endpoint_id="ep-456",
            queries_per_minute=100,
            username=None,
            created_at=1234567890000,
            updated_at=1234567890000,
            created_by="admin",
            updated_by="admin",
        )

        assert config.rate_limit_id == "rl-123"
        assert config.endpoint_id == "ep-456"
        assert config.queries_per_minute == 100
        assert config.username is None
        assert config.is_default is True
        assert config.created_by == "admin"

    def test_gateway_rate_limit_config_per_user(self):
        config = GatewayRateLimitConfig(
            rate_limit_id="rl-789",
            endpoint_id="ep-456",
            queries_per_minute=50,
            username="testuser",
            created_at=1234567890000,
            updated_at=1234567890000,
        )

        assert config.username == "testuser"
        assert config.is_default is False

    def test_gateway_rate_limit_input_entity(self):
        input_data = GatewayRateLimitInput(
            endpoint_id="ep-456",
            queries_per_minute=100,
            username=None,
        )

        assert input_data.endpoint_id == "ep-456"
        assert input_data.queries_per_minute == 100
        assert input_data.username is None

    def test_gateway_rate_limit_input_per_user(self):
        input_data = GatewayRateLimitInput(
            endpoint_id="ep-456",
            queries_per_minute=50,
            username="testuser",
        )

        assert input_data.username == "testuser"
