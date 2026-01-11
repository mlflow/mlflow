"""Tests for Gateway Usage Tracking database models and storage layer."""

import pytest

from mlflow.entities.gateway_usage import (
    GatewayInvocation,
    GatewayProviderCall,
    GatewayUsageMetrics,
    InvocationStatus,
    ProviderCallInput,
    ProviderCallStatus,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.time import get_current_time_millis


@pytest.fixture
def store(tmp_path):
    """Create a temporary SQLAlchemy store for testing."""
    db_path = tmp_path / "test.db"
    store = SqlAlchemyStore(f"sqlite:///{db_path}", str(tmp_path / "artifacts"))
    return store


class TestGatewayUsageEntities:
    """Tests for gateway usage entity classes."""

    def test_invocation_status_enum(self):
        assert InvocationStatus.SUCCESS.value == "SUCCESS"
        assert InvocationStatus.ERROR.value == "ERROR"
        assert InvocationStatus.PARTIAL.value == "PARTIAL"

    def test_provider_call_status_enum(self):
        assert ProviderCallStatus.SUCCESS.value == "SUCCESS"
        assert ProviderCallStatus.ERROR.value == "ERROR"

    def test_gateway_provider_call_entity(self):
        provider_call = GatewayProviderCall(
            provider_call_id="pc-123",
            invocation_id="i-456",
            provider="openai",
            model_name="gpt-4o",
            attempt_number=1,
            status=ProviderCallStatus.SUCCESS,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_cost=0.001,
            completion_cost=0.002,
            total_cost=0.003,
            latency_ms=1500,
            created_at=1234567890000,
        )

        assert provider_call.provider_call_id == "pc-123"
        assert provider_call.invocation_id == "i-456"
        assert provider_call.provider == "openai"
        assert provider_call.model_name == "gpt-4o"
        assert provider_call.status == ProviderCallStatus.SUCCESS
        assert provider_call.total_tokens == 150
        assert provider_call.total_cost == 0.003

    def test_gateway_invocation_entity(self):
        invocation = GatewayInvocation(
            invocation_id="i-123",
            endpoint_id="chat-endpoint",
            endpoint_type="llm/v1/chat",
            status=InvocationStatus.SUCCESS,
            total_prompt_tokens=100,
            total_completion_tokens=50,
            total_tokens=150,
            total_cost=0.003,
            total_latency_ms=1500,
            created_at=1234567890000,
            username="testuser",
        )

        assert invocation.invocation_id == "i-123"
        assert invocation.endpoint_id == "chat-endpoint"
        assert invocation.endpoint_type == "llm/v1/chat"
        assert invocation.status == InvocationStatus.SUCCESS
        assert invocation.total_tokens == 150
        assert invocation.username == "testuser"

    def test_gateway_usage_metrics_entity(self):
        metrics = GatewayUsageMetrics(
            endpoint_id="chat-endpoint",
            time_bucket=1234567890000,
            bucket_size=86400,  # 1 day in seconds
            total_invocations=100,
            successful_invocations=95,
            failed_invocations=5,
            total_prompt_tokens=10000,
            total_completion_tokens=5000,
            total_tokens=15000,
            total_cost=0.5,
            avg_latency_ms=1000.0,
        )

        assert metrics.endpoint_id == "chat-endpoint"
        assert metrics.total_invocations == 100
        assert metrics.successful_invocations == 95
        assert metrics.failed_invocations == 5
        assert metrics.total_tokens == 15000


class TestGatewayUsageStorage:
    """Tests for gateway usage storage operations."""

    def test_log_gateway_invocation_basic(self, store):
        """Test logging a basic gateway invocation."""
        invocation = store.log_gateway_invocation(
            endpoint_id="test-endpoint",
            endpoint_type="llm/v1/chat",
            status=InvocationStatus.SUCCESS,
            total_latency_ms=1000,
            username="testuser",
        )

        assert invocation.invocation_id.startswith("i-")
        assert invocation.endpoint_id == "test-endpoint"
        assert invocation.endpoint_type == "llm/v1/chat"
        assert invocation.status == InvocationStatus.SUCCESS
        assert invocation.username == "testuser"
        assert invocation.total_latency_ms == 1000

    def test_log_gateway_invocation_with_provider_calls(self, store):
        """Test logging a gateway invocation with provider calls."""
        provider_calls = [
            ProviderCallInput(
                provider="openai",
                model_name="gpt-4o",
                attempt_number=1,
                status=ProviderCallStatus.SUCCESS,
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                prompt_cost=0.001,
                completion_cost=0.002,
                total_cost=0.003,
                latency_ms=1500,
            )
        ]

        invocation = store.log_gateway_invocation(
            endpoint_id="test-endpoint",
            endpoint_type="llm/v1/chat",
            status=InvocationStatus.SUCCESS,
            provider_calls=provider_calls,
            total_latency_ms=1500,
        )

        assert invocation.total_prompt_tokens == 100
        assert invocation.total_completion_tokens == 50
        assert invocation.total_tokens == 150
        assert invocation.total_cost == 0.003
        assert len(invocation.provider_calls) == 1
        assert invocation.provider_calls[0].provider == "openai"
        assert invocation.provider_calls[0].model_name == "gpt-4o"

    def test_log_gateway_invocation_with_fallback(self, store):
        """Test logging a gateway invocation with fallback provider calls."""
        provider_calls = [
            ProviderCallInput(
                provider="anthropic",
                model_name="claude-3",
                attempt_number=1,
                status=ProviderCallStatus.ERROR,
                error_message="Rate limited",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                total_cost=0.0,
                latency_ms=500,
            ),
            ProviderCallInput(
                provider="openai",
                model_name="gpt-4o",
                attempt_number=2,
                status=ProviderCallStatus.SUCCESS,
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                prompt_cost=0.001,
                completion_cost=0.002,
                total_cost=0.003,
                latency_ms=1000,
            ),
        ]

        invocation = store.log_gateway_invocation(
            endpoint_id="fallback-endpoint",
            endpoint_type="llm/v1/chat",
            status=InvocationStatus.SUCCESS,
            provider_calls=provider_calls,
            total_latency_ms=1500,
        )

        assert len(invocation.provider_calls) == 2
        assert invocation.provider_calls[0].status == ProviderCallStatus.ERROR
        assert invocation.provider_calls[0].error_message == "Rate limited"
        assert invocation.provider_calls[1].status == ProviderCallStatus.SUCCESS
        assert invocation.total_tokens == 150  # Only successful call tokens

    def test_log_gateway_invocation_with_error(self, store):
        """Test logging a failed gateway invocation."""
        invocation = store.log_gateway_invocation(
            endpoint_id="test-endpoint",
            endpoint_type="llm/v1/chat",
            status=InvocationStatus.ERROR,
            error_message="All providers failed",
            total_latency_ms=500,
        )

        assert invocation.status == InvocationStatus.ERROR
        assert invocation.error_message == "All providers failed"

    def test_get_gateway_invocation(self, store):
        """Test retrieving a gateway invocation by ID."""
        created = store.log_gateway_invocation(
            endpoint_id="test-endpoint",
            endpoint_type="llm/v1/chat",
            status=InvocationStatus.SUCCESS,
        )

        retrieved = store.get_gateway_invocation(created.invocation_id)

        assert retrieved is not None
        assert retrieved.invocation_id == created.invocation_id
        assert retrieved.endpoint_id == "test-endpoint"

    def test_get_gateway_invocation_not_found(self, store):
        """Test retrieving a non-existent gateway invocation."""
        retrieved = store.get_gateway_invocation("non-existent-id")
        assert retrieved is None

    def test_list_gateway_invocations_basic(self, store):
        """Test listing gateway invocations."""
        for i in range(5):
            store.log_gateway_invocation(
                endpoint_id=f"endpoint-{i % 2}",
                endpoint_type="llm/v1/chat",
                status=InvocationStatus.SUCCESS,
            )

        invocations, next_token = store.list_gateway_invocations()

        assert len(invocations) == 5
        assert next_token is None

    def test_list_gateway_invocations_filter_by_endpoint(self, store):
        """Test filtering invocations by endpoint ID."""
        for i in range(5):
            store.log_gateway_invocation(
                endpoint_id=f"endpoint-{i % 2}",
                endpoint_type="llm/v1/chat",
                status=InvocationStatus.SUCCESS,
            )

        invocations, _ = store.list_gateway_invocations(endpoint_id="endpoint-0")

        assert len(invocations) == 3
        assert all(inv.endpoint_id == "endpoint-0" for inv in invocations)

    def test_list_gateway_invocations_filter_by_status(self, store):
        """Test filtering invocations by status."""
        store.log_gateway_invocation(
            endpoint_id="test",
            endpoint_type="llm/v1/chat",
            status=InvocationStatus.SUCCESS,
        )
        store.log_gateway_invocation(
            endpoint_id="test",
            endpoint_type="llm/v1/chat",
            status=InvocationStatus.ERROR,
        )

        invocations, _ = store.list_gateway_invocations(status=InvocationStatus.ERROR)

        assert len(invocations) == 1
        assert invocations[0].status == InvocationStatus.ERROR

    def test_list_gateway_invocations_pagination(self, store):
        """Test pagination of gateway invocations."""
        for i in range(10):
            store.log_gateway_invocation(
                endpoint_id="test",
                endpoint_type="llm/v1/chat",
                status=InvocationStatus.SUCCESS,
            )

        first_page, next_token = store.list_gateway_invocations(max_results=5)

        assert len(first_page) == 5
        assert next_token is not None

        second_page, final_token = store.list_gateway_invocations(
            max_results=5, page_token=next_token
        )

        assert len(second_page) == 5
        assert final_token is None

    def test_get_gateway_token_usage_by_endpoint(self, store):
        """Test getting token usage aggregated by endpoint."""
        provider_calls = [
            ProviderCallInput(
                provider="openai",
                model_name="gpt-4o",
                attempt_number=1,
                status=ProviderCallStatus.SUCCESS,
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                total_cost=0.003,
                latency_ms=1000,
            )
        ]

        for i in range(3):
            store.log_gateway_invocation(
                endpoint_id="endpoint-a",
                endpoint_type="llm/v1/chat",
                status=InvocationStatus.SUCCESS,
                provider_calls=provider_calls,
            )

        for i in range(2):
            store.log_gateway_invocation(
                endpoint_id="endpoint-b",
                endpoint_type="llm/v1/chat",
                status=InvocationStatus.SUCCESS,
                provider_calls=provider_calls,
            )

        usage = store.get_gateway_token_usage_by_endpoint()

        assert len(usage) == 2
        endpoint_a = next(u for u in usage if u["endpoint_id"] == "endpoint-a")
        assert endpoint_a["total_tokens"] == 450  # 150 * 3
        assert endpoint_a["invocation_count"] == 3

    def test_get_gateway_error_rate(self, store):
        """Test getting error rates by endpoint."""
        for i in range(8):
            store.log_gateway_invocation(
                endpoint_id="test",
                endpoint_type="llm/v1/chat",
                status=InvocationStatus.SUCCESS,
            )

        for i in range(2):
            store.log_gateway_invocation(
                endpoint_id="test",
                endpoint_type="llm/v1/chat",
                status=InvocationStatus.ERROR,
            )

        error_rates = store.get_gateway_error_rate(endpoint_id="test")

        # Aggregate across all time buckets to account for slight timestamp differences
        total_invocations = sum(r["total_invocations"] for r in error_rates)
        total_successful = sum(r["successful_invocations"] for r in error_rates)
        total_failed = sum(r["failed_invocations"] for r in error_rates)

        assert total_invocations == 10
        assert total_successful == 8
        assert total_failed == 2


class TestSqlGatewayInvocationModel:
    """Tests for SqlGatewayInvocation database model."""

    def test_to_mlflow_entity(self, store):
        """Test conversion to MLflow entity."""
        invocation = store.log_gateway_invocation(
            endpoint_id="test",
            endpoint_type="llm/v1/chat",
            status=InvocationStatus.SUCCESS,
            provider_calls=[
                ProviderCallInput(
                    provider="openai",
                    model_name="gpt-4o",
                    attempt_number=1,
                    status=ProviderCallStatus.SUCCESS,
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    total_cost=0.003,
                    latency_ms=1000,
                )
            ],
        )

        assert isinstance(invocation, GatewayInvocation)
        assert isinstance(invocation.status, InvocationStatus)
        assert len(invocation.provider_calls) == 1
        assert isinstance(invocation.provider_calls[0], GatewayProviderCall)
        assert isinstance(invocation.provider_calls[0].status, ProviderCallStatus)


class TestGatewayUsageMetrics:
    """Tests for gateway usage metrics queries."""

    def test_get_gateway_usage_metrics_by_day(self, store):
        """Test getting usage metrics bucketed by day."""
        provider_calls = [
            ProviderCallInput(
                provider="openai",
                model_name="gpt-4o",
                attempt_number=1,
                status=ProviderCallStatus.SUCCESS,
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                total_cost=0.003,
                latency_ms=1000,
            )
        ]

        for i in range(5):
            store.log_gateway_invocation(
                endpoint_id="test",
                endpoint_type="llm/v1/chat",
                status=InvocationStatus.SUCCESS,
                provider_calls=provider_calls,
            )

        metrics = store.get_gateway_usage_metrics(
            endpoint_id="test",
            bucket_size=86400,  # 1 day in seconds
        )

        assert len(metrics) >= 1
        total_invocations = sum(m.total_invocations for m in metrics)
        assert total_invocations == 5

    def test_get_gateway_usage_metrics_time_filter(self, store):
        """Test filtering metrics by time range."""
        current_time = get_current_time_millis()

        for i in range(3):
            store.log_gateway_invocation(
                endpoint_id="test",
                endpoint_type="llm/v1/chat",
                status=InvocationStatus.SUCCESS,
            )

        metrics = store.get_gateway_usage_metrics(
            start_time=current_time - 86400000,
            end_time=current_time + 86400000,
        )

        total = sum(m.total_invocations for m in metrics)
        assert total == 3
