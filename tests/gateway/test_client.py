import pytest
from requests.exceptions import HTTPError
from unittest import mock

from mlflow.gateway.envs import MLFLOW_GATEWAY_URI  # TODO: change to environment_variables import
from mlflow.exceptions import MlflowException, InvalidUrlException
import mlflow.gateway.utils
from mlflow.gateway import set_gateway_uri, MlflowGatewayClient
from mlflow.gateway.config import Route
from tests.gateway.tools import Gateway, save_yaml


@pytest.fixture
def basic_config_dict():
    return {
        "routes": [
            {
                "name": "completions",
                "route_type": "llm/v1/completions",
                "model": {
                    "name": "text-davinci-003",
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
                "name": "chat",
                "route_type": "llm/v1/chat",
                "model": {
                    "name": "gpt-3.5-turbo",
                    "provider": "openai",
                    "config": {"openai_api_key": "mykey"},
                },
            },
            {
                "name": "embeddings",
                "route_type": "llm/v1/embeddings",
                "model": {
                    "provider": "openai",
                    "name": "text-embedding-ada-002",
                    "config": {
                        "openai_api_base": "https://api.openai.com/v1",
                        "openai_api_key": "mykey",
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
    save_yaml(conf, basic_config_dict)
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


def test_non_running_server_raises_when_called(monkeypatch):
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    set_gateway_uri("http://invalid.server:6000")
    client = MlflowGatewayClient()
    with pytest.raises(
        MlflowException,
        match="API request to http://invalid.server:6000/api/2.0/gateway/routes/ failed ",
    ):
        client.search_routes()


def test_create_gateway_client_with_declared_url(gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)
    assert gateway_client.gateway_uri == gateway.url
    assert isinstance(gateway_client.get_route("chat"), Route)


def test_set_gateway_uri_from_utils(gateway):
    set_gateway_uri(gateway_uri=gateway.url)

    gateway_client = MlflowGatewayClient()
    assert gateway_client.gateway_uri == gateway.url
    assert isinstance(gateway_client.get_route("completions"), Route)


def test_create_gateway_client_with_environment_variable(gateway, monkeypatch):
    monkeypatch.setenv(MLFLOW_GATEWAY_URI.name, gateway.url)

    gateway_client = MlflowGatewayClient()
    assert gateway_client.gateway_uri == gateway.url
    assert isinstance(gateway_client.get_route("completions"), Route)


def test_create_gateway_client_with_overriden_env_variable(gateway, monkeypatch):
    monkeypatch.setenv(MLFLOW_GATEWAY_URI.name, "http://localhost:99999")

    # Pass a bad env variable config in
    with pytest.raises(
        InvalidUrlException, match="Invalid url: http://localhost:99999/api/2.0/gateway/routes"
    ):
        MlflowGatewayClient().search_routes()

    # Ensure that the global variable override preempts trying the environment variable value
    set_gateway_uri(gateway_uri=gateway.url)
    gateway_client = MlflowGatewayClient()

    assert gateway_client.gateway_uri == gateway.url
    assert gateway_client.get_route("chat").route_type == "llm/v1/chat"


def test_query_individual_route(gateway, monkeypatch):
    monkeypatch.setenv(MLFLOW_GATEWAY_URI.name, gateway.url)

    gateway_client = MlflowGatewayClient()

    route1 = gateway_client.get_route(name="completions")
    assert isinstance(route1, Route)
    assert route1.dict() == {
        "model": {"name": "text-davinci-003", "provider": "openai"},
        "name": "completions",
        "route_type": "llm/v1/completions",
    }

    route2 = gateway_client.get_route(name="chat")
    assert isinstance(route2, Route)
    assert route2.dict() == {
        "model": {"name": "gpt-3.5-turbo", "provider": "openai"},
        "name": "chat",
        "route_type": "llm/v1/chat",
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
        "model": {"name": "text-davinci-003", "provider": "openai"},
        "name": "completions",
        "route_type": "llm/v1/completions",
    }
    assert routes[1].dict() == {
        "model": {"name": "gpt-3.5-turbo", "provider": "openai"},
        "name": "chat",
        "route_type": "llm/v1/chat",
    }


def test_client_query_chat(gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)

    routes = gateway_client.search_routes()

    data = {"messages": [{"role": "user", "content": "How hot is the core of the sun?"}]}

    expected_output = {
        "candidates": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The core of the sun is estimated to have a temperature of about "
                    "15 million degrees Celsius (27 million degrees Fahrenheit).",
                },
                "metadata": {"finish_reason": "stop"},
            }
        ],
        "metadata": {
            "input_tokens": 17,
            "output_tokens": 24,
            "total_tokens": 41,
            "model": "gpt-3.5-turbo-0301",
            "route_type": "llm/v1/chat",
        },
    }
    mock_response = mock.Mock()
    mock_response.json.return_value = expected_output

    with mock.patch.object(gateway_client, "_call_endpoint", return_value=mock_response):
        response = gateway_client.query(route=routes[1].name, data=data)

        assert response == expected_output


def test_client_query_completions(gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)

    routes = gateway_client.search_routes()

    expected_output = {
        "candidates": [
            {
                "text": " car\n\nDriving fast can be dangerous and is not recommended. It is",
                "metadata": {"finish_reason": "length"},
            }
        ],
        "metadata": {
            "input_tokens": 7,
            "output_tokens": 16,
            "total_tokens": 23,
            "model": "text-davinci-003",
            "route_type": "llm/v1/completions",
        },
    }

    data = {"prompt": "I like to drive fast in my"}

    mock_response = mock.Mock()
    mock_response.json.return_value = expected_output

    with mock.patch.object(gateway_client, "_call_endpoint", return_value=mock_response):
        response = gateway_client.query(route=routes[0].name, data=data)
        assert response == expected_output


def test_client_query_embeddings(gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)

    routes = gateway_client.search_routes()

    expected_output = {
        "embeddings": [
            [
                0.1,
                0.2,
                0.3,
            ],
            [
                0.4,
                0.5,
                0.6,
            ],
        ],
        "metadata": {
            "input_tokens": 8,
            "output_tokens": 0,
            "total_tokens": 8,
            "model": "text-embedding-ada-002",
            "route_type": "llm/v1/embeddings",
        },
    }
    data = {"text": ["Jenny", "What's her number?"]}

    mock_response = mock.Mock()
    mock_response.json.return_value = expected_output

    with mock.patch.object(gateway_client, "_call_endpoint", return_value=mock_response):
        response = gateway_client.query(route=routes[2].name, data=data)
        assert response == expected_output
