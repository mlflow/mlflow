from unittest import mock

import pytest
from requests.exceptions import HTTPError

import mlflow.gateway.utils
from mlflow.environment_variables import MLFLOW_GATEWAY_URI
from mlflow.exceptions import MlflowException
from mlflow.gateway import (
    create_route,
    delete_route,
    get_gateway_uri,
    get_limits,
    get_route,
    query,
    search_routes,
    set_gateway_uri,
    set_limits,
)
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
                    "name": "gpt-3.5-turbo",
                    "provider": "openai",
                    "config": {"openai_api_key": "mykey"},
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


def test_fluent_apis_with_no_server_set():
    with pytest.raises(MlflowException, match="No Gateway server uri has been set. Please"):
        get_route("bogus")

    with pytest.raises(MlflowException, match="No Gateway server uri has been set. Please"):
        get_route("claude-chat")


def test_fluent_health_check_on_non_running_server(monkeypatch):
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    set_gateway_uri("http://not.real:1000")
    with pytest.raises(
        MlflowException,
        match="API request to http://not.real:1000/api/2.0/gateway/routes/not-a-route failed with",
    ):
        get_route("not-a-route")


def test_fluent_health_check_on_env_var_uri(gateway, monkeypatch):
    monkeypatch.setenv(MLFLOW_GATEWAY_URI.name, gateway.url)
    mlflow.gateway.utils._gateway_uri = None
    assert get_route("completions").model.name == "text-davinci-003"


def test_fluent_health_check_on_fluent_set(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    assert get_route("completions").model.provider == "openai"


def test_fluent_get_valid_route(gateway):
    set_gateway_uri(gateway_uri=gateway.url)

    route = get_route("completions")
    assert isinstance(route, Route)
    assert route.dict() == {
        "model": {"name": "text-davinci-003", "provider": "openai"},
        "name": "completions",
        "route_type": "llm/v1/completions",
        "route_url": resolve_route_url(gateway.url, "gateway/completions/invocations"),
    }


def test_fluent_get_invalid_route(gateway):
    set_gateway_uri(gateway_uri=gateway.url)

    with pytest.raises(HTTPError, match="404 Client Error: Not Found"):
        get_route("not-a-route")


def test_fluent_search_routes(gateway):
    set_gateway_uri(gateway_uri=gateway.url)

    routes = search_routes()
    assert all(isinstance(route, Route) for route in routes)
    assert routes[0].dict() == {
        "model": {"name": "text-davinci-003", "provider": "openai"},
        "name": "completions",
        "route_type": "llm/v1/completions",
        "route_url": resolve_route_url(gateway.url, "gateway/completions/invocations"),
    }
    assert routes[1].dict() == {
        "model": {"name": "gpt-3.5-turbo", "provider": "openai"},
        "name": "chat",
        "route_type": "llm/v1/chat",
        "route_url": resolve_route_url(gateway.url, "gateway/chat/invocations"),
    }


def test_fluent_search_routes_handles_pagination(tmp_path):
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
    num_routes = (MLFLOW_GATEWAY_SEARCH_ROUTES_PAGE_SIZE * 2) + 1
    gateway_route_names = [f"route_{i}" for i in range(num_routes)]
    gateway_config_dict = {
        "routes": [{"name": route_name, **base_route_config} for route_name in gateway_route_names]
    }
    save_yaml(conf, gateway_config_dict)
    with Gateway(conf) as gateway:
        set_gateway_uri(gateway_uri=gateway.url)
        assert [route.name for route in search_routes()] == gateway_route_names


def test_fluent_get_gateway_uri(gateway):
    set_gateway_uri(gateway_uri=gateway.url)

    assert get_gateway_uri() == gateway.url


def test_fluent_query_chat(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    routes = search_routes()
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

    data = {"messages": [{"role": "user", "content": "How hot is the core of the sun?"}]}

    with mock.patch(
        "mlflow.gateway.fluent.MlflowGatewayClient.query", return_value=expected_output
    ):
        response = query(route=routes[1].name, data=data)
        assert response == expected_output


def test_fluent_query_completions(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    routes = search_routes()
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

    with mock.patch(
        "mlflow.gateway.fluent.MlflowGatewayClient.query", return_value=expected_output
    ):
        response = query(route=routes[0].name, data=data)
        assert response == expected_output


def test_fluent_create_route_raises(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    # This API is only available in Databricks
    with pytest.raises(MlflowException, match="The create_route API is only available when"):
        create_route(
            "some-route", "llm/v1/completions", {"name": "some_name", "provider": "anthropic"}
        )


def test_fluent_delete_route_raises(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    # This API is only available in Databricks
    with pytest.raises(MlflowException, match="The delete_route API is only available when"):
        delete_route("some-route")


def test_fluent_set_limits_raises(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    # This API is only available in Databricks
    with pytest.raises(
        HTTPError, match=".*The set_limits API is not available in OSS MLflow AI Gateway."
    ):
        set_limits("some-route", [])


def test_fluent_get_limits_raises(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    # This API is only available in Databricks
    with pytest.raises(
        HTTPError, match=".*The get_limits API is not available in OSS MLflow AI Gateway."
    ):
        get_limits("some-route")


def test_fluent_query_with_disallowed_param(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    route = get_route("completions")

    data = {"prompt": "Test", "temperature": 0.4, "model": "gpt-4"}

    with pytest.raises(HTTPError, match=".*The parameter 'model' is not permitted.*"):
        query(route=route.name, data=data)


def test_get_route_accepts_unknown_provider():
    set_gateway_uri("http://localhost:5000")
    mock_resp = mock.Mock(status_code=200)
    mock_resp.json.return_value = {
        "name": "chat",
        "route_type": "llm/v1/chat",
        "model": {"name": "unknown-5", "provider": "unknown-ai"},
        "route_url": "http://localhost:5000/gateway/chat/invocations",
    }
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        route = get_route("chat")
        mock_request.assert_called_once()
        assert route.dict() == mock_resp.json()


def test_get_route_accepts_unknown_route_type():
    set_gateway_uri("http://localhost:5000")
    mock_resp = mock.Mock(status_code=200)
    mock_resp.json.return_value = {
        "name": "chat",
        "route_type": "llm/v1/unknown",
        "model": {"name": "gpt4", "provider": "openai"},
        "route_url": "http://localhost:5000/gateway/chat/invocations",
    }
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        route = get_route("chat")
        mock_request.assert_called_once()
        assert route.dict() == mock_resp.json()


def test_search_routes_accepts_unknown_provider():
    set_gateway_uri("http://localhost:5000")
    mock_resp = mock.Mock(status_code=200)
    mock_resp.json.return_value = {
        "routes": [
            {
                "name": "chat",
                "route_type": "llm/v1/chat",
                "model": {"name": "unknown-5", "provider": "unknown-ai"},
                "route_url": "http://localhost:5000/gateway/chat/invocations",
            },
        ],
        "next_page_token": None,
    }
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        routes = search_routes()
        mock_request.assert_called_once()
        assert [r.dict() for r in routes] == mock_resp.json()["routes"]


def test_search_routes_accepts_unknown_route_type():
    set_gateway_uri("http://localhost:5000")
    mock_resp = mock.Mock(status_code=200)
    mock_resp.json.return_value = {
        "routes": [
            {
                "name": "chat",
                "route_type": "llm/v1/unknown",
                "model": {"name": "gpt4", "provider": "openai"},
                "route_url": "http://localhost:5000/gateway/chat/invocations",
            },
        ],
        "next_page_token": None,
    }
    with mock.patch("requests.Session.request", return_value=mock_resp) as mock_request:
        routes = search_routes()
        mock_request.assert_called_once()
        assert [r.dict() for r in routes] == mock_resp.json()["routes"]
