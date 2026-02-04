import pytest

from mlflow.gateway.tracing_utils import traced_gateway_call
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig
from mlflow.tracing.client import TracingClient
from mlflow.tracking.fluent import _get_experiment_id


def get_traces():
    return TracingClient().search_traces(locations=[_get_experiment_id()])


@pytest.fixture
def endpoint_config():
    return GatewayEndpointConfig(
        endpoint_id="test-endpoint-id",
        endpoint_name="test-endpoint",
        experiment_id=_get_experiment_id(),
        models=[],
    )


@pytest.fixture
def endpoint_config_no_experiment():
    return GatewayEndpointConfig(
        endpoint_id="test-endpoint-id",
        endpoint_name="test-endpoint",
        experiment_id=None,
        models=[],
    )


async def mock_async_func(payload):
    return {"result": "success", "payload": payload}


@pytest.mark.asyncio
async def test_traced_gateway_call_basic(endpoint_config):
    traced_func = traced_gateway_call(mock_async_func, endpoint_config)
    result = await traced_func({"input": "test"})

    assert result == {"result": "success", "payload": {"input": "test"}}

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]

    # Find the gateway span
    span_name_to_span = {span.name: span for span in trace.data.spans}
    assert f"gateway/{endpoint_config.endpoint_name}" in span_name_to_span

    gateway_span = span_name_to_span[f"gateway/{endpoint_config.endpoint_name}"]
    assert gateway_span.attributes.get("endpoint_id") == "test-endpoint-id"
    assert gateway_span.attributes.get("endpoint_name") == "test-endpoint"
    # No user attributes should be present
    assert gateway_span.attributes.get("username") is None
    assert gateway_span.attributes.get("user_id") is None


@pytest.mark.asyncio
async def test_traced_gateway_call_with_user_attributes(endpoint_config):
    traced_func = traced_gateway_call(
        mock_async_func,
        endpoint_config,
        attributes={"username": "alice", "user_id": 123},
    )
    result = await traced_func({"input": "test"})

    assert result == {"result": "success", "payload": {"input": "test"}}

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]

    span_name_to_span = {span.name: span for span in trace.data.spans}
    gateway_span = span_name_to_span[f"gateway/{endpoint_config.endpoint_name}"]

    assert gateway_span.attributes.get("endpoint_id") == "test-endpoint-id"
    assert gateway_span.attributes.get("endpoint_name") == "test-endpoint"
    assert gateway_span.attributes.get("username") == "alice"
    assert gateway_span.attributes.get("user_id") == 123


@pytest.mark.asyncio
async def test_traced_gateway_call_without_experiment_id(endpoint_config_no_experiment):
    traced_func = traced_gateway_call(
        mock_async_func,
        endpoint_config_no_experiment,
        attributes={"username": "alice", "user_id": 123},
    )

    # When experiment_id is None, traced_gateway_call returns the original function
    assert traced_func is mock_async_func

    result = await traced_func({"input": "test"})
    assert result == {"result": "success", "payload": {"input": "test"}}

    # No traces should be created
    traces = get_traces()
    assert len(traces) == 0
