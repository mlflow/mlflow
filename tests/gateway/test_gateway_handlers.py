import pytest
import yaml

from mlflow.exceptions import MlflowException
from mlflow.gateway.handlers import (
    _load_gateway_config,
    _save_gateway_config,
    RouteConfig,
    _convert_route_config_to_route,
)


@pytest.fixture(autouse=True)
def basic_config_dict():
    return [
        {
            "name": "instruct-gpt4",
            "type": "llm/v1/instruct",
            "model": {
                "name": "gpt-4",
                "provider": "openai",
                "config": {
                    "openai_api_key": "mykey",
                    "openai_api_base": "https://api.openai.com/",
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
                "config": {"openai_api_key_env_var": "MY_API_KEY"},
            },
        },
        {
            "name": "claude-chat",
            "type": "llm/v1/chat",
            "model": {
                "name": "claude-v1",
                "provider": "anthropic",
                "config": {
                    "anthropic_api_key": "claudekey",
                },
            },
        },
    ]


def test_route_configuration_parsing(basic_config_dict, tmp_path):
    conf_path = tmp_path.joinpath("config.yaml")

    conf_path.write_text(yaml.safe_dump(basic_config_dict))

    loaded_config = _load_gateway_config(conf_path)

    save_path = tmp_path.joinpath("config2.yaml")
    _save_gateway_config(loaded_config, save_path)
    loaded_from_save = _load_gateway_config(save_path)
    assert loaded_config == loaded_from_save


def test_convert_route_config_to_routes_payload(basic_config_dict, tmp_path):
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(basic_config_dict))
    loaded = _load_gateway_config(conf_path)

    assert all(isinstance(route, RouteConfig) for route in loaded)

    routes = _convert_route_config_to_route(loaded)

    for config in loaded:
        route = [x for x in routes if x.name == config.name][0]
        assert route.type == config.type
        assert route.model.name == config.model.name
        assert route.model.provider == config.model.provider
        # Pydantic doesn't allow undefined elements to be a part of its serialized object.
        # This test is a guard for devs only in case we inadvertantly add sensitive keys to the
        # Route definition that would be returned via the GetRoute or SearchRoutes APIs
        assert not hasattr(route.model, "config")


def test_invalid_route_definition(tmp_path):
    invalid_partial_config = [
        {
            "name": "some_name",
            "type": "invalid",
            "model": {
                "name": "invalid",
                "provider": "openai",
                "config": {"openai_api_type": "chat"},
            },
        }
    ]

    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(invalid_partial_config))

    with pytest.raises(
        MlflowException,
        match="For the openai provider, the api key must either be specified within the ",
    ):
        _load_gateway_config(conf_path)

    invalid_no_config = [
        {
            "name": "some_name",
            "type": "invalid",
            "model": {
                "name": "invalid",
                "provider": "anthropic",
            },
        }
    ]
    conf_path = tmp_path.joinpath("config2.yaml")
    conf_path.write_text(yaml.safe_dump(invalid_no_config))

    with pytest.raises(
        MlflowException,
        match="A config must be supplied when setting a provider. The provider entry",
    ):
        _load_gateway_config(conf_path)


def test_custom_provider(tmp_path):
    basic_generic_provider = [
        {
            "name": "some_name",
            "type": "some/type",
            "model": {
                "name": "my_custom_provider",
                "provider": "my_provider",
                "config": {"api_key": "mykey", "api_base": "http://my.endpoint.com/"},
            },
        }
    ]
    conf_path = tmp_path.joinpath("config2.yaml")
    conf_path.write_text(yaml.safe_dump(basic_generic_provider))

    generic_conf = _load_gateway_config(conf_path)
    route = generic_conf[0]

    assert route.model.provider == "custom"
    assert route.name == "some_name"
    assert route.model.name == "my_custom_provider"
    assert route.model.config.get("api_key") == "mykey"
    assert route.model.config.get("api_key_env_var", None) is None
    assert route.model.config.get("api_version", None) is None


@pytest.mark.parametrize(
    "route_name", ["Space Name", "bang!name", "query?name", "redirect#name", "bracket[]name"]
)
def test_invalid_route_name(tmp_path, route_name):
    bad_name = [
        {
            "name": route_name,
            "type": "bad/naming",
            "model": {
                "name": "claude-v1",
                "provider": "anthropic",
                "config": {
                    "anthropic_api_key": "claudekey",
                },
            },
        }
    ]

    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(bad_name))

    with pytest.raises(
        MlflowException, match="The route name provided contains disallowed characters"
    ):
        _load_gateway_config(conf_path)


def test_custom_route(tmp_path):
    custom_routes = [
        {
            "name": "route1",
            "type": "document/classification",
            "model": {
                "name": "prod",
                "provider": "hosted",
                "config": {
                    "api_key_env_var": "MY_KEY",
                    "api_base": "http://myserver.endpoint.org/",
                },
            },
        },
        {
            "name": "route2",
            "type": "document/sentiment",
            "model": {
                "name": "staging",
                "provider": "hosted",
                "config": {
                    "api_key": "MY_KEY",
                    "api_base": "http://myserver.endpoint.org/",
                    "api_version": "3",
                },
            },
        },
    ]
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(custom_routes))
    loaded_conf = _load_gateway_config(conf_path)

    assert loaded_conf[0].name == "route1"
    assert loaded_conf[0].model.config.get("api_base") == "http://myserver.endpoint.org/"
    assert loaded_conf[0].model.config.get("api_version", None) is None
    assert loaded_conf[1].model.provider == "custom"
    assert loaded_conf[1].model.config.get("api_key") == "MY_KEY"


def test_default_base_api(tmp_path):
    route_no_base = [
        {
            "name": "chat-gpt4",
            "type": "llm/v1/chat",
            "model": {
                "name": "gpt-4",
                "provider": "openai",
                "config": {"openai_api_key_env_var": "MY_API_KEY"},
            },
        },
    ]
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(route_no_base))
    loaded_conf = _load_gateway_config(conf_path)

    assert loaded_conf[0].model.config.get("openai_api_base") == "https://api.openai.com/"


def test_databricks_route_config(tmp_path):
    databricks_route = [
        {
            "name": "classifier",
            "type": "llm/v1/classifier",
            "model": {
                "name": "serving-endpoints/document-classifier/Production/invocations",
                "provider": "databricks",
                "config": {
                    "databricks_api_token_env_var": "MY_TOKEN",
                    "databricks_api_base": "https://my-shard-001/",
                },
            },
        }
    ]
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(databricks_route))
    loaded_conf = _load_gateway_config(conf_path)
    route = loaded_conf[0]

    assert route.type == "custom"
    assert route.model.name == "serving-endpoints/document-classifier/Production/invocations"
    assert route.model.provider == "databricks"
    assert route.model.config.get("databricks_api_token", None) is None
    assert route.model.config.get("databricks_api_base") == "https://my-shard-001/"
