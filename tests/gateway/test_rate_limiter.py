"""Tests for Gateway Rate Limiter."""

import time
from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities.gateway_rate_limit import GatewayRateLimitConfig
from mlflow.gateway.rate_limiter import GatewayRateLimiter, get_rate_limiter


@pytest.fixture
def rate_limiter():
    """Create a fresh rate limiter instance for each test."""
    limiter = GatewayRateLimiter()
    yield limiter
    limiter.clear()


@pytest.fixture
def mock_store():
    """Create a mock store for testing."""
    return MagicMock()


class TestGatewayRateLimiter:
    """Tests for GatewayRateLimiter class."""

    def test_no_rate_limit_configured(self, rate_limiter, mock_store):
        """Test that requests are allowed when no rate limit is configured."""
        mock_store.get_gateway_rate_limit_for_user.return_value = None

        allowed, limit, remaining = rate_limiter.check_rate_limit(
            mock_store, "endpoint-1", "user-1"
        )

        assert allowed is True
        assert limit is None
        assert remaining is None

    def test_rate_limit_allows_requests_under_limit(self, rate_limiter, mock_store):
        """Test that requests under the limit are allowed."""
        mock_store.get_gateway_rate_limit_for_user.return_value = GatewayRateLimitConfig(
            rate_limit_id="rl-1",
            endpoint_id="endpoint-1",
            queries_per_minute=10,
            username=None,
            created_at=0,
            updated_at=0,
        )

        # First request should be allowed
        allowed, limit, remaining = rate_limiter.check_rate_limit(
            mock_store, "endpoint-1", "user-1"
        )

        assert allowed is True
        assert limit == 10
        assert remaining == 9

    def test_rate_limit_blocks_requests_over_limit(self, rate_limiter, mock_store):
        """Test that requests over the limit are blocked."""
        mock_store.get_gateway_rate_limit_for_user.return_value = GatewayRateLimitConfig(
            rate_limit_id="rl-1",
            endpoint_id="endpoint-1",
            queries_per_minute=3,
            username=None,
            created_at=0,
            updated_at=0,
        )

        # Make requests up to and over the limit
        for i in range(3):
            allowed, _, _ = rate_limiter.check_rate_limit(
                mock_store, "endpoint-1", "user-1"
            )
            assert allowed is True

        # Fourth request should be blocked
        allowed, limit, remaining = rate_limiter.check_rate_limit(
            mock_store, "endpoint-1", "user-1"
        )

        assert allowed is False
        assert limit == 3
        assert remaining == 0

    def test_rate_limit_per_user_isolation(self, rate_limiter, mock_store):
        """Test that per-user rate limits are isolated."""
        # User-specific rate limit
        mock_store.get_gateway_rate_limit_for_user.return_value = GatewayRateLimitConfig(
            rate_limit_id="rl-1",
            endpoint_id="endpoint-1",
            queries_per_minute=2,
            username="user-1",
            created_at=0,
            updated_at=0,
        )

        # User 1 makes 2 requests (at limit)
        rate_limiter.check_rate_limit(mock_store, "endpoint-1", "user-1")
        rate_limiter.check_rate_limit(mock_store, "endpoint-1", "user-1")

        # User 1's third request should be blocked
        allowed, _, _ = rate_limiter.check_rate_limit(mock_store, "endpoint-1", "user-1")
        assert allowed is False

        # User 2 with different limit should still be allowed
        mock_store.get_gateway_rate_limit_for_user.return_value = GatewayRateLimitConfig(
            rate_limit_id="rl-2",
            endpoint_id="endpoint-1",
            queries_per_minute=5,
            username="user-2",
            created_at=0,
            updated_at=0,
        )

        allowed, _, _ = rate_limiter.check_rate_limit(mock_store, "endpoint-1", "user-2")
        assert allowed is True

    def test_rate_limit_window_expires(self, rate_limiter, mock_store):
        """Test that rate limit window expires after 60 seconds."""
        mock_store.get_gateway_rate_limit_for_user.return_value = GatewayRateLimitConfig(
            rate_limit_id="rl-1",
            endpoint_id="endpoint-1",
            queries_per_minute=2,
            username=None,
            created_at=0,
            updated_at=0,
        )

        # Make requests up to the limit
        rate_limiter.check_rate_limit(mock_store, "endpoint-1", None)
        rate_limiter.check_rate_limit(mock_store, "endpoint-1", None)

        # Should be blocked now
        allowed, _, _ = rate_limiter.check_rate_limit(mock_store, "endpoint-1", None)
        assert allowed is False

        # Mock time to advance past the window
        with patch("mlflow.gateway.rate_limiter.time") as mock_time:
            # Set current time to 61 seconds in the future
            mock_time.time.return_value = time.time() + 61

            # Request should now be allowed as old requests expired
            allowed, _, remaining = rate_limiter.check_rate_limit(
                mock_store, "endpoint-1", None
            )
            assert allowed is True
            assert remaining == 1  # 2 - 1 (the new request)

    def test_rate_limit_endpoint_isolation(self, rate_limiter, mock_store):
        """Test that rate limits are isolated per endpoint."""
        mock_store.get_gateway_rate_limit_for_user.return_value = GatewayRateLimitConfig(
            rate_limit_id="rl-1",
            endpoint_id="endpoint-1",
            queries_per_minute=1,
            username=None,
            created_at=0,
            updated_at=0,
        )

        # Endpoint 1 reaches limit
        rate_limiter.check_rate_limit(mock_store, "endpoint-1", None)
        allowed, _, _ = rate_limiter.check_rate_limit(mock_store, "endpoint-1", None)
        assert allowed is False

        # Endpoint 2 should not be affected
        mock_store.get_gateway_rate_limit_for_user.return_value = GatewayRateLimitConfig(
            rate_limit_id="rl-2",
            endpoint_id="endpoint-2",
            queries_per_minute=1,
            username=None,
            created_at=0,
            updated_at=0,
        )

        allowed, _, _ = rate_limiter.check_rate_limit(mock_store, "endpoint-2", None)
        assert allowed is True

    def test_clear_rate_limits(self, rate_limiter, mock_store):
        """Test that clear() removes all rate limit windows."""
        mock_store.get_gateway_rate_limit_for_user.return_value = GatewayRateLimitConfig(
            rate_limit_id="rl-1",
            endpoint_id="endpoint-1",
            queries_per_minute=1,
            username=None,
            created_at=0,
            updated_at=0,
        )

        # Reach the limit
        rate_limiter.check_rate_limit(mock_store, "endpoint-1", None)
        allowed, _, _ = rate_limiter.check_rate_limit(mock_store, "endpoint-1", None)
        assert allowed is False

        # Clear the limiter
        rate_limiter.clear()

        # Should be allowed again
        allowed, _, _ = rate_limiter.check_rate_limit(mock_store, "endpoint-1", None)
        assert allowed is True


class TestGetRateLimiter:
    """Tests for the global rate limiter getter."""

    def test_get_rate_limiter_returns_singleton(self):
        """Test that get_rate_limiter returns the same instance."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()

        assert limiter1 is limiter2

    def test_get_rate_limiter_creates_instance(self):
        """Test that get_rate_limiter creates an instance when none exists."""
        limiter = get_rate_limiter()

        assert limiter is not None
        assert isinstance(limiter, GatewayRateLimiter)
