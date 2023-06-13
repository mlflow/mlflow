import pytest
from requests.exceptions import HTTPError
from unittest import mock

from mlflow.gateway.envs import MLFLOW_GATEWAY_URI  # TODO: change to environment_variables import
from mlflow.exceptions import MlflowException, InvalidUrlException
import mlflow.gateway.utils
from mlflow.gateway import set_gateway_uri, MlflowGatewayClient
from mlflow.gateway.config import Route
from mlflow.utils.databricks_utils import MlflowHostCreds
from tests.gateway.tools import Gateway, store_conf


@pytest.fixture
def basic_config_dict():
    return {
        "routes": [
            {
                "name": "completions-gpt4",
                "type": "llm/v1/completions",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                    "config": {
                        "openai_api_key": "mykey",
                        "openai_api_base": "https://api.openai.com/v1",
                        "openai_api_version": "2023-05-10",
                        "openai_api_type": "openai/v1/chat/completions",
                    },
                },
            },
            {
                "name": "chat-gpt4",
                "type": "llm/v1/chat",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                    "config": {"openai_api_key": "MY_API_KEY"},
                },
            },
            {
                "name": "claude-chat",
                "type": "llm/v1/chat",
                "model": {
                    "name": "claude-v1",
                    "provider": "anthropic",
                    "config": {
                        "anthropic_api_key": "api_key",
                    },
                },
            },
        ]
    }


@pytest.fixture(autouse=True)
def clear_uri():
    mlflow.gateway.utils._gateway_uri = None


@pytest.fixture
def gateway(basic_config_dict, tmp_path):
    conf = tmp_path / "config.yaml"
    store_conf(conf, basic_config_dict)
    with Gateway(conf) as g:
        yield g


@pytest.mark.parametrize(
    "uri",
    [
        "''",  # empty string
        "http://",  # missing netloc
        "gateway.org:8000",  # missing scheme
        "gateway.org",  # missing scheme
        "ftp://",  # missing netloc, wrong scheme
        "www.gateway.org",  # missing scheme
        "http:://gateway.org",  # double colon typo
        "http:/gateway.com",  # single slash typo
        "http:gateway.org",  # missing slashes
    ],
)
def test_invalid_uri_on_utils_raises(uri):
    with pytest.raises(MlflowException, match="The gateway uri provided is missing required"):
        set_gateway_uri(uri)


@pytest.mark.parametrize(
    "uri, base_start",
    [
        ("http://local:6000", "/gateway"),
        ("databricks", "/ml/gateway"),
        ("databricks://my.shard", "/ml/gateway"),
    ],
)
def test_databricks_base_route_modification(uri, base_start):
    mock_host_creds = MlflowHostCreds("mock-host")

    with mock.patch(
        "mlflow.gateway.client.get_databricks_host_creds", return_value=mock_host_creds
    ):
        client = MlflowGatewayClient(gateway_uri=uri)

        assert client._route_base.startswith(base_start)


def test_non_running_server_raises_when_called():
    set_gateway_uri("http://invalid.server:6000")
    client = MlflowGatewayClient()
    with pytest.raises(
        MlflowException,
        match="API request to http://invalid.server:6000/health failed with exception",
    ):
        client.get_gateway_health()


def test_create_gateway_client_with_declared_url(gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)
    assert gateway_client.gateway_uri == gateway.url
    assert gateway_client.get_gateway_health()


def test_set_gateway_uri_from_utils(gateway):
    set_gateway_uri(gateway_uri=gateway.url)

    gateway_client = MlflowGatewayClient()
    assert gateway_client.gateway_uri == gateway.url
    assert gateway_client.get_gateway_health()


def test_create_gateway_client_with_environment_variable(gateway, monkeypatch):
    monkeypatch.setenv(MLFLOW_GATEWAY_URI.name, gateway.url)

    gateway_client = MlflowGatewayClient()
    assert gateway_client.gateway_uri == gateway.url
    assert gateway_client.get_gateway_health()


def test_create_gateway_client_with_overriden_env_variable(gateway, monkeypatch):
    monkeypatch.setenv(MLFLOW_GATEWAY_URI.name, "http://localhost:99999")

    # Pass a bad env variable config in
    with pytest.raises(InvalidUrlException, match="Invalid url: http://localhost:99999/health"):
        MlflowGatewayClient().get_gateway_health()

    # Ensure that the global variable override preempts trying the environment variable value
    set_gateway_uri(gateway_uri=gateway.url)
    gateway_client = MlflowGatewayClient()

    assert gateway_client.gateway_uri == gateway.url
    assert gateway_client.get_gateway_health()


def test_query_individual_route(gateway, monkeypatch):
    monkeypatch.setenv(MLFLOW_GATEWAY_URI.name, gateway.url)

    gateway_client = MlflowGatewayClient()

    route1 = gateway_client.get_route(name="completions-gpt4")
    assert isinstance(route1, Route)
    assert route1.dict() == {
        "model": {"name": "gpt-4", "provider": "openai"},
        "name": "completions-gpt4",
        "type": "llm/v1/completions",
    }

    route2 = gateway_client.get_route(name="chat-gpt4")
    assert isinstance(route2, Route)
    assert route2.dict() == {
        "model": {"name": "gpt-4", "provider": "openai"},
        "name": "chat-gpt4",
        "type": "llm/v1/chat",
    }


def test_query_invalid_route(gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)

    with pytest.raises(HTTPError, match="404 Client Error: Not Found"):
        gateway_client.get_route("invalid-route")


def test_list_all_configured_routes(gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)

    # This is a non-functional filter applied only to ensure that print a warning
    with pytest.raises(MlflowException, match="Search functionality is not implemented"):
        gateway_client.search_routes(search_filter="where 'myroute' contains 'gpt'")

    routes = gateway_client.search_routes()
    assert all(isinstance(x, Route) for x in routes)
    assert routes[0].dict() == {
        "model": {"name": "gpt-4", "provider": "openai"},
        "name": "completions-gpt4",
        "type": "llm/v1/completions",
    }
    assert routes[1].dict() == {
        "model": {"name": "gpt-4", "provider": "openai"},
        "name": "chat-gpt4",
        "type": "llm/v1/chat",
    }
