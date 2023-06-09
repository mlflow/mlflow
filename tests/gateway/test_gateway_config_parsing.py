import os
import pathlib
import pytest
import yaml

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import (
    _load_route_config,
    _save_route_config,
    RouteConfig,
    _route_configs_to_routes,
    _resolve_api_key_from_input,
)


@pytest.fixture
def basic_config_dict():
    return [
        {
            "name": "instruct-gpt4",
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
                    "anthropic_api_key": "/tmp/claudekey.conf",
                },
            },
        },
    ]


def test_api_key_parsing():
    os.environ["KEY_AS_ENV"] = "my_key"
    env_keys = ["KEY_AS_ENV", "$KEY_AS_ENV"]

    for key in env_keys:
        assert _resolve_api_key_from_input(key) == "my_key"

    string_key = "my_key_as_a_string"

    assert _resolve_api_key_from_input(string_key) == string_key

    path_for_file = "~/mlflow/gateway/mykey.conf"
    file_key = "Here is my key that sits safely in a file"

    file_dir = pathlib.Path(path_for_file)
    file_dir.parent.mkdir(parents=True, exist_ok=True)
    file_dir.write_text(file_key)

    assert _resolve_api_key_from_input(path_for_file) == file_key

    del os.environ["KEY_AS_ENV"]
    file_dir.unlink()


def test_route_configuration_parsing(basic_config_dict, tmp_path):
    conf_path = tmp_path.joinpath("config.yaml")

    conf_path.write_text(yaml.safe_dump(basic_config_dict))

    # Write a file in /tmp/claudekey that contains a string
    path_for_file = "/tmp/claudekey.conf"
    file_key = "Here is my key that sits safely in a file"

    file_dir = pathlib.Path(path_for_file)
    file_dir.parent.mkdir(parents=True, exist_ok=True)
    file_dir.write_text(file_key)

    # Set an environment variable
    os.environ["MY_API_KEY"] = "my_env_var_key"

    loaded_config = _load_route_config(conf_path)

    save_path = tmp_path.joinpath("config2.yaml")
    _save_route_config(loaded_config, save_path)
    loaded_from_save = _load_route_config(save_path)

    instruct_gpt4 = loaded_from_save[0]
    assert instruct_gpt4.name == "instruct-gpt4"
    assert instruct_gpt4.type == "llm/v1/completions"
    assert instruct_gpt4.model.name == "gpt-4"
    assert instruct_gpt4.model.provider == "openai"
    instruct_conf = instruct_gpt4.model.config
    assert instruct_conf["openai_api_key"] == "mykey"
    assert instruct_conf["openai_api_base"] == "https://api.openai.com/v1"
    assert instruct_conf["openai_api_version"] == "2023-05-10"
    assert instruct_conf["openai_api_type"] == "openai/v1/chat/completions"
    assert instruct_conf["openai_organization"] == "my_company"

    chat_gpt4 = loaded_from_save[1]

    assert chat_gpt4.name == "chat-gpt4"
    assert chat_gpt4.type == "llm/v1/chat"
    assert chat_gpt4.model.name == "gpt-4"
    assert chat_gpt4.model.provider == "openai"
    chat_conf = chat_gpt4.model.config
    assert chat_conf["openai_api_key"] == "my_env_var_key"
    assert chat_conf["openai_api_base"] == "https://api.openai.com/v1"
    assert chat_conf.get("openai_api_version", None) is None
    assert chat_conf.get("openai_api_type", None) is None
    assert chat_conf.get("openai_organization", None) is None

    claude = loaded_from_save[2]
    assert claude.name == "claude-chat"
    assert claude.type == "llm/v1/chat"
    assert claude.model.name == "claude-v1"
    assert claude.model.provider == "anthropic"
    claude_conf = claude.model.config
    assert claude_conf["anthropic_api_key"] == file_key
    assert claude_conf["anthropic_api_base"] == "https://api.anthropic.com/"

    # Delete the environment variable
    del os.environ["MY_API_KEY"]
    # Delete the file
    file_dir.unlink()


def test_convert_route_config_to_routes_payload(basic_config_dict, tmp_path):
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(basic_config_dict))
    loaded = _load_route_config(conf_path)

    assert all(isinstance(route, RouteConfig) for route in loaded)

    routes = _route_configs_to_routes(loaded)

    for config in loaded:
        route = [x for x in routes if x.name == config.name][0]
        assert route.type == config.type
        assert route.model.name == config.model.name
        assert route.model.provider == config.model.provider
        # Pydantic doesn't allow undefined elements to be a part of its serialized object.
        # This test is a guard for devs only in case we inadvertently add sensitive keys to the
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
        _load_route_config(conf_path)

    invalid_format_config_key_is_not_string = [
        {
            "name": "some_name",
            "type": "invalid",
            "model": {
                "name": "invalid",
                "provider": "openai",
                "config": {"openai_api_type": "chat", "openai_api_key": [42]},
            },
        }
    ]

    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(invalid_format_config_key_is_not_string))

    with pytest.raises(
        MlflowException,
        match="The gateway configuration is invalid",
    ):
        _load_route_config(conf_path)

    invalid_format_config_key_invalid_path = [
        {
            "name": "some_name",
            "type": "invalid",
            "model": {
                "name": "invalid",
                "provider": "openai",
                "config": {"openai_api_type": "chat", "openai_api_key": "/not/a/real/path"},
            },
        }
    ]

    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(invalid_format_config_key_invalid_path))

    assert _load_route_config(conf_path)[0].model.config["openai_api_key"] == "/not/a/real/path"

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
        _load_route_config(conf_path)


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

    generic_conf = _load_route_config(conf_path)
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
        _load_route_config(conf_path)


def test_custom_route(tmp_path):
    custom_routes = [
        {
            "name": "route1",
            "type": "document/classification",
            "model": {
                "name": "prod",
                "provider": "hosted",
                "config": {
                    "api_key": "MY_KEY",
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
    loaded_conf = _load_route_config(conf_path)

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
                "config": {"openai_api_key": "MY_API_KEY"},
            },
        },
    ]
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(route_no_base))
    loaded_conf = _load_route_config(conf_path)

    assert loaded_conf[0].model.config.get("openai_api_base") == "https://api.openai.com/v1"


def test_databricks_route_config(tmp_path):
    databricks_route = [
        {
            "name": "classifier",
            "type": "llm/v1/classifier",
            "model": {
                "name": "serving-endpoints/document-classifier/Production/invocations",
                "provider": "databricks_serving_endpoint",
                "config": {
                    "databricks_api_token": "MY_TOKEN",
                    "databricks_api_base": "https://my-shard-001/",
                },
            },
        }
    ]
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(databricks_route))
    loaded_conf = _load_route_config(conf_path)
    route = loaded_conf[0]

    assert route.type == "custom"
    assert route.model.name == "serving-endpoints/document-classifier/Production/invocations"
    assert route.model.provider == "databricks_serving_endpoint"
    assert route.model.config.get("databricks_api_token") == "MY_TOKEN"
    assert route.model.config.get("databricks_api_base") == "https://my-shard-001/"


def test_duplicate_routes_in_config(tmp_path):
    route = [
        {
            "name": "classifier",
            "type": "llm/v1/classifier",
            "model": {
                "name": "serving-endpoints/document-classifier/Production/invocations",
                "provider": "databricks_serving_endpoint",
                "config": {
                    "databricks_api_token": "MY_TOKEN",
                    "databricks_api_base": "https://my-shard-001/",
                },
            },
        },
        {
            "name": "classifier",
            "type": "llm/v1/classifier",
            "model": {
                "name": "serving-endpoints/document-classifier/Production/invocations",
                "provider": "databricks_serving_endpoint",
                "config": {
                    "databricks_api_token": "MY_TOKEN",
                    "databricks_api_base": "https://my-shard-001/",
                },
            },
        },
    ]
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(route))
    with pytest.raises(
        MlflowException, match="Duplicate names found in route configurations. Please"
    ):
        _load_route_config(conf_path)
