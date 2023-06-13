import pytest

from mlflow.environment_variables import MLFLOW_GATEWAY_URI
from mlflow.exceptions import MlflowException
from mlflow.gateway import set_gateway_uri, MlflowGatewayClient
from mlflow.gateway.config import Route
from tests.gateway.helper_functions import Gateway, store_conf


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
                        "openai_organization": "my_company",
                    },
                },
            },
            {
                "name": "chat-gpt4",
                "type": "llm/v1/chat",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                    "config": {"openai_api_key": "$MY_API_KEY"},
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


def test_non_running_server_raises():
    with pytest.raises(MlflowException, match="The gateway server cannot be verified at"):
        set_gateway_uri("http://invalid.server:6000")


def test_instantiating_client_with_no_server_uri_raises():
    with pytest.raises(MlflowException, match="No Gateway server uri has been set. Please either"):
        MlflowGatewayClient()


def test_create_gateway_client_with_declared_url(basic_config_dict, tmp_path):
    config = tmp_path / "config.yaml"
    store_conf(config, basic_config_dict)

    with Gateway(config) as gateway:
        gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)

        assert gateway_client.get_gateway_uri == gateway.url

        health = gateway_client.get_gateway_health()
        assert health == {"status": "OK"}


def test_set_gateway_uri_from_utils(basic_config_dict, tmp_path):
    config = tmp_path / "config.yaml"
    store_conf(config, basic_config_dict)

    with Gateway(config) as gateway:
        set_gateway_uri(gateway_uri=gateway.url)

        gateway_client = MlflowGatewayClient()

        assert gateway_client.get_gateway_uri == gateway.url

        health = gateway_client.get_gateway_health()
        assert health == {"status": "OK"}


def test_create_gateway_client_with_environment_variable(basic_config_dict, tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    store_conf(config, basic_config_dict)

    with Gateway(config) as gateway:
        monkeypatch.setenv(MLFLOW_GATEWAY_URI.name, gateway.url)

        gateway_client = MlflowGatewayClient()

        assert gateway_client.get_gateway_uri == gateway.url

        health = gateway_client.get_gateway_health()
        assert health == {"status": "OK"}


def test_query_individual_route(basic_config_dict, tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    store_conf(config, basic_config_dict)

    with Gateway(config) as gateway:
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
        assert route2.dict() == {
            "model": {"name": "gpt-4", "provider": "openai"},
            "name": "chat-gpt4",
            "type": "llm/v1/chat",
        }


def test_list_all_configured_routes(basic_config_dict, tmp_path, capsys):
    config = tmp_path / "config.yaml"
    store_conf(config, basic_config_dict)

    with Gateway(config) as gateway:
        gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)

        # This is a non-functional filter applied only to ensure that print a warning
        routes = gateway_client.search_routes(search_filter="where 'myroute' contains 'gpt'")
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

    captured = capsys.readouterr()
    assert "Search functionality is not implemented. This API will" in captured.err
