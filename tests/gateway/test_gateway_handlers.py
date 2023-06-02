import pytest
import yaml


from mlflow.exceptions import MlflowException
from mlflow.gateway.config import GatewayConfigSingleton
from mlflow.gateway.handlers import (
    _load_gateway_config,
    _save_gateway_config,
    RouteConfig,
    _convert_route_config_to_route,
)


@pytest.fixture(autouse=True)
def basic_config_dict():
    routes_data = [
        {
            "name": "instruct-gpt4",
            "type": "llm/v1/instruct",
            "model": {
                "name": "gpt-4",
                "provider": "openai",
                "config": {
                    "openai_api_key": "mykey",
                    "openai_api_base": "https://api.openai.com/",
                    "openai_api_version": "v1",
                    "openai_api_type": "completions",
                },
            },
        },
        {
            "name": "bard-chat",
            "type": "llm/v1/chat",
            "model": {
                "name": "bard-chat",
                "provider": "bard",
                "config": {"bard_api_key": "somekey"},
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

    return routes_data


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


def test_gateway_config_singleton(basic_config_dict, tmp_path):
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(basic_config_dict))
    loaded = _load_gateway_config(conf_path)

    GatewayConfigSingleton.getInstance().update_config(loaded)

    config = GatewayConfigSingleton.getInstance().gateway_config

    assert config == loaded

    with pytest.raises(
        MlflowException, match="The GatewayConfigSingleton can only be instantiated"
    ):
        GatewayConfigSingleton()

    updated_route_conf_dict = [
        {
            "name": "instruct-gpt3.5",
            "type": "llm/v1/instruct",
            "model": {
                "name": "gpt-3.5",
                "provider": "openai",
                "config": {
                    "openai_api_key": "adifferentkey",
                    "openai_api_base": "https://api.openai.com/",
                    "openai_api_version": "v1",
                    "openai_api_type": "completions",
                },
            },
        },
    ]
    updated_path = tmp_path.joinpath("updated_conf.yaml")
    updated_path.write_text(yaml.safe_dump(updated_route_conf_dict))

    updated_loaded = _load_gateway_config(updated_path)

    GatewayConfigSingleton.getInstance().update_config(updated_loaded)

    updated_config = GatewayConfigSingleton.getInstance().gateway_config
    assert updated_loaded == updated_config
    assert updated_config != config
