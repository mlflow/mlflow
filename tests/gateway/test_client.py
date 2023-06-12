import pytest
import yaml

from mlflow.environment_variables import MLFLOW_GATEWAY_URI
from mlflow.exceptions import MlflowException
import mlflow.gateway
from mlflow.gateway.client import set_mlflow_gateway_uri
from tests.gateway.helper_functions import Gateway


@pytest.fixture
def basic_config_dict():
    return [
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


@pytest.fixture(autouse=True)
def reset_mlflow_gateway_uri_env_var():
    # reset the gateway uri to None prior to each test
    MLFLOW_GATEWAY_URI.set("")
    yield


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
def test_invalid_uri_raises(uri):
    with pytest.raises(MlflowException, match="The gateway uri provided is missing required"):
        set_mlflow_gateway_uri(uri)


def test_non_running_server_raises():
    with pytest.raises(MlflowException, match="The gateway server cannot be verified at"):
        set_mlflow_gateway_uri("http://invalid.server:6000")


def test_running_server_validates_when_setting_env_var(basic_config_dict, tmp_path):
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(basic_config_dict))

    with Gateway(str(conf_path)) as gateway:
        uri = gateway.url

        mlflow.gateway.set_mlflow_gateway_uri(gateway_uri=uri)

        assert MLFLOW_GATEWAY_URI.get() == uri
        assert mlflow.gateway.get_mlflow_gateway_uri() == uri


def test_query_individual_route(basic_config_dict, tmp_path):
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(basic_config_dict))

    with Gateway(str(conf_path)) as gateway:
        assert MLFLOW_GATEWAY_URI.get() == ""

        with pytest.raises(MlflowException, match="The MLflow Gateway uri has not been set"):
            mlflow.gateway.get_mlflow_gateway_uri()

        mlflow.gateway.set_mlflow_gateway_uri(gateway_uri=gateway.url)

        route1 = mlflow.gateway.get_route(name="completions-gpt4")
        assert route1 == {
            "route": {
                "name": "completions-gpt4",
                "type": "llm/v1/completions",
                "model": {"name": "gpt-4", "provider": "openai"},
            }
        }
        route2 = mlflow.gateway.get_route(name="chat-gpt4")
        assert route2 == {
            "route": {
                "model": {"name": "gpt-4", "provider": "openai"},
                "name": "chat-gpt4",
                "type": "llm/v1/chat",
            }
        }


# def test_list_all_configured_routes()
