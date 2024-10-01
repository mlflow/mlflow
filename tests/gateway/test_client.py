from unittest import mock

import pytest
from requests.exceptions import HTTPError, Timeout

import mlflow.gateway.utils
from mlflow.environment_variables import MLFLOW_GATEWAY_URI
from mlflow.exceptions import InvalidUrlException, MlflowException
from mlflow.gateway import MlflowGatewayClient, set_gateway_uri
from mlflow.gateway.config import Route
from mlflow.gateway.constants import MLFLOW_GATEWAY_SEARCH_ROUTES_PAGE_SIZE
from mlflow.gateway.utils import resolve_route_url

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
                        "openai_api_type": "openai",
                    },
                },
            },
            {
                "name": "chat",
                "route_type": "llm/v1/chat",
                "model": {
                    "name": "gpt-4o-mini",
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


@pytest.fixture
def mixed_config_dict():
    return {
        "routes": [
            {
                "name": "chat",
                "route_type": "llm/v1/chat",
                "model": {
                    "name": "gpt-4o-mini",
                    "provider": "openai",
                    "config": {"openai_api_key": "mykey"},
                },
            },
            {
                "name": "completions",
                "route_type": "llm/v1/completions",
                "model": {
                    "provider": "anthropic",
                    "name": "claude-instant-1",
                    "config": {
                        "anthropic_api_key": "key",
                    },
                },
            },
        ]
    }


@pytest.fixture
def mlflow_mixed_config_dict():
    return {
        "routes": [
            {
                "name": "chat-oss",
                "route_type": "llm/v1/chat",
                "model": {
                    "provider": "mlflow-model-serving",
                    "name": "mpt-chatbot",
                    "config": {"model_server_url": "http://127.0.0.1:5000"},
                },
            },
            {
                "name": "completions-oss",
                "route_type": "llm/v1/completions",
                "model": {
                    "provider": "mlflow-model-serving",
                    "name": "mpt-completion-model",
                    "config": {"model_server_url": "http://127.0.0.1:5001"},
                },
            },
            {
                "name": "embeddings-oss",
                "route_type": "llm/v1/embeddings",
                "model": {
                    "provider": "mlflow-model-serving",
                    "name": "sentence-transformers",
                    "config": {"model_server_url": "http://127.0.0.1:5002"},
                },
            },
            {
                "name": "completions",
                "route_type": "llm/v1/completions",
                "model": {
                    "provider": "anthropic",
                    "name": "claude-instant-1",
                    "config": {
                        "anthropic_api_key": "key",
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


@pytest.fixture
def mixed_gateway(mixed_config_dict, tmp_path):
    conf = tmp_path / "config.yaml"
    save_yaml(conf, mixed_config_dict)
    with Gateway(conf) as g:
        yield g


@pytest.fixture
def oss_gateway(mlflow_mixed_config_dict, tmp_path):
    conf = tmp_path / "config.yaml"
    save_yaml(conf, mlflow_mixed_config_dict)
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


def test_get_individual_routes(gateway, monkeypatch):
    monkeypatch.setenv(MLFLOW_GATEWAY_URI.name, gateway.url)

    gateway_client = MlflowGatewayClient()

    route1 = gateway_client.get_route(name="completions")
    assert isinstance(route1, Route)
    assert route1.dict() == {
        "model": {"name": "text-davinci-003", "provider": "openai"},
        "name": "completions",
        "route_type": "llm/v1/completions",
        "route_url": resolve_route_url(gateway.url, "gateway/completions/invocations"),
        "limit": None,
    }

    route2 = gateway_client.get_route(name="chat")
    assert isinstance(route2, Route)
    assert route2.dict() == {
        "model": {"name": "gpt-4o-mini", "provider": "openai"},
        "name": "chat",
        "route_type": "llm/v1/chat",
        "route_url": resolve_route_url(gateway.url, "gateway/chat/invocations"),
        "limit": None,
    }


def test_get_mixed_routes(mixed_gateway, monkeypatch):
    monkeypatch.setenv(MLFLOW_GATEWAY_URI.name, mixed_gateway.url)

    gateway_client = MlflowGatewayClient()
    chat_route = gateway_client.get_route(name="chat")
    assert chat_route.route_type == "llm/v1/chat"
    assert chat_route.name == "chat"
    assert chat_route.model.name == "gpt-4o-mini"
    assert chat_route.model.provider == "openai"
    assert chat_route.route_url == resolve_route_url(mixed_gateway.url, "gateway/chat/invocations")

    completions_route = gateway_client.get_route(name="completions")
    assert completions_route.route_type == "llm/v1/completions"
    assert completions_route.name == "completions"
    assert completions_route.model.name == "claude-instant-1"
    assert completions_route.model.provider == "anthropic"
    assert completions_route.route_url == resolve_route_url(
        mixed_gateway.url, "gateway/completions/invocations"
    )


def test_search_mixed_routes(mixed_gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=mixed_gateway.url)
    routes = gateway_client.search_routes()

    assert len(routes) == 2
    chat_route = routes[0]
    assert chat_route.route_type == "llm/v1/chat"
    assert chat_route.name == "chat"
    assert chat_route.model.provider == "openai"
    assert chat_route.route_url == resolve_route_url(mixed_gateway.url, "gateway/chat/invocations")

    completions_route = routes[1]
    assert completions_route.route_type == "llm/v1/completions"
    assert completions_route.name == "completions"
    assert completions_route.model.provider == "anthropic"
    assert completions_route.route_url == resolve_route_url(
        mixed_gateway.url, "gateway/completions/invocations"
    )


def test_search_routes_returns_expected_pages(tmp_path):
    conf = tmp_path / "config.yaml"
    base_route_config = {
        "route_type": "llm/v1/completions",
        "model": {
            "name": "text-davinci-003",
            "provider": "openai",
            "config": {
                "openai_api_key": "mykey",
                "openai_api_base": "https://api.openai.com/v1",
                "openai_api_version": "2023-05-10",
                "openai_api_type": "openai",
            },
        },
    }
    num_routes = MLFLOW_GATEWAY_SEARCH_ROUTES_PAGE_SIZE + 5
    gateway_route_names = [f"route_{i}" for i in range(num_routes)]
    gateway_config_dict = {
        "routes": [{"name": route_name, **base_route_config} for route_name in gateway_route_names]
    }
    save_yaml(conf, gateway_config_dict)
    with Gateway(conf) as gateway:
        gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)
        routes_page_1 = gateway_client.search_routes()
        assert [route.name for route in routes_page_1] == gateway_route_names[
            :MLFLOW_GATEWAY_SEARCH_ROUTES_PAGE_SIZE
        ]
        assert routes_page_1.token

        routes_page_2 = gateway_client.search_routes(page_token=routes_page_1.token)
        assert len(routes_page_2) == 5
        assert [route.name for route in routes_page_2] == gateway_route_names[
            MLFLOW_GATEWAY_SEARCH_ROUTES_PAGE_SIZE:
        ]
        assert routes_page_2.token is None


def test_query_invalid_route(gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)

    with pytest.raises(HTTPError, match="404 Client Error: Not Found"):
        gateway_client.get_route("invalid-route")


def test_list_all_configured_routes(gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)

    routes = gateway_client.search_routes()
    assert all(isinstance(x, Route) for x in routes)
    assert routes[0].dict() == {
        "model": {"name": "text-davinci-003", "provider": "openai"},
        "name": "completions",
        "route_type": "llm/v1/completions",
        "route_url": resolve_route_url(gateway.url, "gateway/completions/invocations"),
        "limit": None,
    }
    assert routes[1].dict() == {
        "model": {"name": "gpt-4o-mini", "provider": "openai"},
        "name": "chat",
        "route_type": "llm/v1/chat",
        "route_url": resolve_route_url(gateway.url, "gateway/chat/invocations"),
        "limit": None,
    }


def test_client_query_chat(gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)

    routes = gateway_client.search_routes()

    data = {"messages": [{"role": "user", "content": "How hot is the core of the sun?"}]}

    expected_output = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The core of the sun is estimated to have a temperature of about "
                    "15 million degrees Celsius (27 million degrees Fahrenheit).",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "usage": {
            "prompt_tokens": 17,
            "completion_tokens": 24,
            "total_tokens": 41,
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
        "id": "chatcmpl-abc123",
        "object": "text_completion",
        "created": 1677858242,
        "model": "text-davinci-003",
        "choices": [
            {
                "text": " car\n\nDriving fast can be dangerous and is not recommended. It is",
                "index": 0,
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 16, "total_tokens": 23},
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
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [
                    0.1,
                    0.2,
                    0.3,
                ],
                "index": 0,
            },
            {
                "object": "embedding",
                "embedding": [
                    0.4,
                    0.5,
                    0.6,
                ],
                "index": 1,
            },
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }
    data = {"text": ["Jenny", "What's her number?"]}

    mock_response = mock.Mock()
    mock_response.json.return_value = expected_output

    with mock.patch.object(gateway_client, "_call_endpoint", return_value=mock_response):
        response = gateway_client.query(route=routes[2].name, data=data)
        assert response == expected_output


def test_client_create_route_raises(gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)

    # This API is available only in Databricks for route creation
    with pytest.raises(MlflowException, match="The create_route API is only available when"):
        gateway_client.create_route(
            "some-route",
            "llm/v1/chat",
            {
                "name": "a-route",
                "provider": "openai",
                "config": {
                    "openai_api_key": "mykey",
                    "openai_api_type": "openai",
                },
            },
        )


def test_client_set_limits_raises(gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)

    with pytest.raises(HTTPError, match="The set_limits API is not available"):
        gateway_client.set_limits("some-route", [])


def test_client_get_limits_raises(gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)

    with pytest.raises(HTTPError, match="The get_limits API is not available"):
        gateway_client.get_limits("some-route")


def test_client_delete_route_raises(gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)

    # This API is available only in Databricks for route deletion
    with pytest.raises(MlflowException, match="The delete_route API is only available when"):
        gateway_client.delete_route("some-route")


def test_client_query_anthropic_completions(mixed_gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=mixed_gateway.url)

    route = gateway_client.get_route(name="completions")
    assert route.model.provider == "anthropic"

    expected_output = {
        "id": None,
        "object": "text_completion",
        "created": 1677858242,
        "model": "claude-instant-1.1",
        "choices": [
            {
                "text": "Here are the steps for making a peanut butter sandwich:\n\n1. Get bread. "
                "\n\n2. Spread peanut butter on bread.",
                "index": 0,
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }
    data = {"prompt": "Can you tell me how to make a peanut butter sandwich?", "max_tokens": 500}

    mock_response = mock.Mock()
    mock_response.json.return_value = expected_output

    with mock.patch.object(gateway_client, "_call_endpoint", return_value=mock_response):
        response = gateway_client.query(route=route.name, data=data)
        assert response == expected_output


def test_client_query_with_disallowed_param(mixed_gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=mixed_gateway.url)

    route = gateway_client.get_route(name="completions")

    data = {"prompt": "Testing", "temperature": 0.8, "model": "gpt-4"}

    with pytest.raises(HTTPError, match=".*The parameter 'model' is not permitted.*"):
        gateway_client.query(route=route.name, data=data)


def test_query_timeout_not_retried(mixed_gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=mixed_gateway.url)

    data = {"prompt": "Test", "temperature": 0.4}
    route = "completions"

    with mock.patch(
        "mlflow.gateway.constants.MLFLOW_GATEWAY_CLIENT_QUERY_TIMEOUT_SECONDS", new=1
    ), mock.patch("requests.Session.request", side_effect=Timeout) as mocked_request:
        with pytest.raises(MlflowException, match="The provider has timed out while generating"):
            gateway_client.query(route=route, data=data)

        mocked_request.assert_called_once()


def test_client_query_mlflow_chat_route(oss_gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=oss_gateway.url)

    route = gateway_client.get_route(name="chat-oss")
    assert route.model.provider == "mlflow-model-serving"
    assert route.route_url == f"{oss_gateway.url}/gateway/chat-oss/invocations"

    data = {"messages": [{"role": "user", "content": "Is this a test?"}]}

    expected_output = {
        "id": None,
        "created": 1700242674,
        "object": "chat.completion",
        "model": "mpt-chatbot",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "It is a test",
                },
                "finish_reason": None,
                "index": 0,
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    }
    mock_response = mock.Mock()
    mock_response.json.return_value = expected_output

    with mock.patch.object(gateway_client, "_call_endpoint", return_value=mock_response):
        response = gateway_client.query(route="chat-oss", data=data)
        assert response == expected_output


def test_client_query_mlflow_completions_route(oss_gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=oss_gateway.url)

    route = gateway_client.get_route(name="completions-oss")
    assert route.model.provider == "mlflow-model-serving"
    assert route.route_url == f"{oss_gateway.url}/gateway/completions-oss/invocations"

    data = {"prompt": "Tell me what this is"}

    expected_output = {
        "id": None,
        "object": "text_completion",
        "created": 1677858242,
        "model": "mpt-completion-model",
        "choices": [
            {
                "text": "a test",
                "index": 0,
                "finish_reason": None,
            }
        ],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }
    mock_response = mock.Mock()
    mock_response.json.return_value = expected_output

    with mock.patch.object(gateway_client, "_call_endpoint", return_value=mock_response):
        response = gateway_client.query(route="completions-oss", data=data)
        assert response == expected_output


def test_client_query_mlflow_embeddings_route(oss_gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=oss_gateway.url)

    route = gateway_client.get_route(name="embeddings-oss")
    assert route.model.provider == "mlflow-model-serving"
    assert route.route_url == f"{oss_gateway.url}/gateway/embeddings-oss/invocations"

    data = {"text": ["test1", "test2"]}

    expected_output = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [
                    0.1,
                    0.2,
                    0.3,
                ],
                "index": 0,
            },
            {
                "object": "embedding",
                "embedding": [
                    0.4,
                    0.5,
                    0.6,
                ],
                "index": 1,
            },
        ],
        "model": "sentence-transformers",
        "usage": {"prompt_tokens": None, "total_tokens": None},
    }

    mock_response = mock.Mock()
    mock_response.json.return_value = expected_output

    with mock.patch.object(gateway_client, "_call_endpoint", return_value=mock_response):
        response = gateway_client.query(route="embeddings-oss", data=data)
        assert response == expected_output


def test_search_routes_no_routes():
    gateway_client = MlflowGatewayClient(gateway_uri="http://localhost:5000")
    resp = mock.Mock(status_code=200)
    resp.json.return_value = {}
    with mock.patch("requests.Session.request", return_value=resp) as request_mock:
        routes = gateway_client.search_routes()
        request_mock.assert_called_once()
        assert routes == []
        assert routes.token is None
