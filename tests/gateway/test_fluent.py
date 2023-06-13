import pytest
from requests.exceptions import HTTPError

from mlflow.exceptions import MlflowException
from mlflow.gateway import (
    set_gateway_uri,
    get_gateway_uri,
    get_route,
    search_routes,
    get_gateway_health,
)
from mlflow.gateway.config import Route
from mlflow.gateway.envs import MLFLOW_GATEWAY_URI
import mlflow.gateway.utils
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


@pytest.fixture
def gateway(basic_config_dict, tmp_path):
    conf = tmp_path / "config.yaml"
    store_conf(conf, basic_config_dict)
    with Gateway(conf) as g:
        yield g
    mlflow.gateway.utils._gateway_uri = None


def test_fluent_apis_with_no_server_set():
    with pytest.raises(MlflowException, match="No Gateway server uri has been set. Please"):
        get_gateway_health()

    with pytest.raises(MlflowException, match="No Gateway server uri has been set. Please"):
        get_route("claude-chat")


def test_fluent_health_check_on_non_running_server():
    set_gateway_uri("http://not.real:1000")
    with pytest.raises(
        MlflowException, match="API request to http://not.real:1000/health failed with exception"
    ):
        get_gateway_health()


def test_fluent_health_check_on_env_var_uri(gateway, monkeypatch):
    monkeypatch.setenv(MLFLOW_GATEWAY_URI.name, gateway.url)
    mlflow.gateway.utils._gateway_uri = None
    assert get_gateway_health()


def test_fluent_health_check_on_fluent_set(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    assert get_gateway_health()


def test_fluent_get_valid_route(gateway):
    set_gateway_uri(gateway_uri=gateway.url)

    route = get_route("claude-chat")
    assert isinstance(route, Route)
    assert route.dict() == {
        "model": {"name": "claude-v1", "provider": "anthropic"},
        "name": "claude-chat",
        "type": "llm/v1/chat",
    }


def test_fluent_get_invalid_route(gateway):
    set_gateway_uri(gateway_uri=gateway.url)

    with pytest.raises(HTTPError, match="404 Client Error: Not Found"):
        get_route("not-a-route")


def test_fluent_search_routes(gateway):
    set_gateway_uri(gateway_uri=gateway.url)

    with pytest.raises(MlflowException, match="Search functionality is not implemented"):
        search_routes(search_filter="route like %anthrop")

    routes = search_routes()
    assert all(isinstance(route, Route) for route in routes)
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


def test_fluent_get_gateway_uri(gateway):
    set_gateway_uri(gateway_uri=gateway.url)

    assert get_gateway_uri() == gateway.url
