import re

import pytest
import yaml

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import (
    AnthropicConfig,
    OpenAIConfig,
    RouteConfig,
    _load_route_config,
    _resolve_api_key_from_input,
    _save_route_config,
)
from mlflow.gateway.utils import assemble_uri_path


@pytest.fixture
def basic_config_dict():
    return {
        "endpoints": [
            {
                "name": "completions-gpt4",
                "endpoint_type": "llm/v1/completions",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                    "config": {
                        "openai_api_key": "mykey",
                        "openai_api_base": "https://api.openai.com/v1",
                        "openai_api_version": "2023-05-10",
                        "openai_api_type": "openai",
                        "openai_organization": "my_company",
                    },
                },
            },
            {
                "name": "chat-gpt4",
                "endpoint_type": "llm/v1/chat",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                    "config": {"openai_api_key": "sk-openai"},
                },
            },
            {
                "name": "claude-chat",
                "endpoint_type": "llm/v1/chat",
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
    ("paths", "expected"),
    [
        (["gateway", "/routes/", "/chat"], "/gateway/routes/chat"),
        (["/gateway/", "/routes", "chat"], "/gateway/routes/chat"),
        (["gateway/routes/", "chat"], "/gateway/routes/chat"),
        (["gateway/", "routes/chat"], "/gateway/routes/chat"),
        (["/gateway/routes", "/chat/"], "/gateway/routes/chat"),
        (["/gateway", "/routes/", "chat/"], "/gateway/routes/chat"),
        (["/"], "/"),
        (["gateway", "", "/routes/", "", "/chat", ""], "/gateway/routes/chat"),
    ],
)
def test_assemble_uri_path(paths, expected):
    assert assemble_uri_path(paths) == expected


def test_api_key_parsing_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KEY_AS_ENV", "my_key")

    assert _resolve_api_key_from_input("$KEY_AS_ENV") == "my_key"
    monkeypatch.delenv("KEY_AS_ENV", raising=False)
    with pytest.raises(MlflowException, match="Environment variable 'KEY_AS_ENV' is not set"):
        _resolve_api_key_from_input("$KEY_AS_ENV")

    string_key = "my_key_as_a_string"

    assert _resolve_api_key_from_input(string_key) == string_key

    conf_path = tmp_path.joinpath("mykey.conf")
    file_key = "Here is my key that sits safely in a file"

    conf_path.write_text(file_key)

    assert _resolve_api_key_from_input(str(conf_path)) == file_key


def test_api_key_input_exceeding_maximum_filename_length():
    assert _resolve_api_key_from_input("a" * 256) == "a" * 256


def test_api_key_parsing_file(tmp_path):
    key_path = tmp_path.joinpath("api.key")
    config = {
        "endpoints": [
            {
                "name": "claude-chat",
                "endpoint_type": "llm/v1/chat",
                "model": {
                    "name": "claude-v1",
                    "provider": "anthropic",
                    "config": {
                        "anthropic_api_key": str(key_path),
                    },
                },
            },
        ]
    }

    key_path.write_text("abc")
    config_path = tmp_path.joinpath("config.yaml")
    config_path.write_text(yaml.safe_dump(config))
    loaded_config = _load_route_config(config_path)

    assert isinstance(loaded_config.endpoints[0].model.config, AnthropicConfig)
    assert loaded_config.endpoints[0].model.config.anthropic_api_key == "abc"


def test_route_configuration_parsing(basic_config_dict, tmp_path, monkeypatch):
    conf_path = tmp_path.joinpath("config.yaml")

    conf_path.write_text(yaml.safe_dump(basic_config_dict))

    loaded_config = _load_route_config(conf_path)

    save_path = tmp_path.joinpath("config2.yaml")
    _save_route_config(loaded_config, save_path)
    loaded_from_save = _load_route_config(save_path)

    completions_gpt4 = loaded_from_save.endpoints[0]
    assert completions_gpt4.name == "completions-gpt4"
    assert completions_gpt4.endpoint_type == "llm/v1/completions"
    assert completions_gpt4.model.name == "gpt-4"
    assert completions_gpt4.model.provider == "openai"
    completions_conf = completions_gpt4.model.config
    assert isinstance(completions_conf, OpenAIConfig)
    assert completions_conf.openai_api_key == "mykey"
    assert completions_conf.openai_api_base == "https://api.openai.com/v1"
    assert completions_conf.openai_api_version == "2023-05-10"
    assert completions_conf.openai_api_type == "openai"
    assert completions_conf.openai_organization == "my_company"

    chat_gpt4 = loaded_from_save.endpoints[1]
    assert chat_gpt4.name == "chat-gpt4"
    assert chat_gpt4.endpoint_type == "llm/v1/chat"
    assert chat_gpt4.model.name == "gpt-4"
    assert chat_gpt4.model.provider == "openai"
    chat_conf = chat_gpt4.model.config
    assert isinstance(chat_conf, OpenAIConfig)
    assert chat_conf.openai_api_key == "sk-openai"
    assert chat_conf.openai_api_base == "https://api.openai.com/v1"
    assert chat_conf.openai_api_type == "openai"
    assert chat_conf.openai_api_version is None
    assert chat_conf.openai_organization is None

    claude = loaded_from_save.endpoints[2]
    assert isinstance(claude.model.config, AnthropicConfig)
    assert claude.name == "claude-chat"
    assert claude.endpoint_type == "llm/v1/chat"
    assert claude.model.name == "claude-v1"
    assert claude.model.provider == "anthropic"
    claude_conf = claude.model.config
    assert claude_conf.anthropic_api_key == "api_key"


def test_convert_route_config_to_routes_payload(basic_config_dict, tmp_path):
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(basic_config_dict))
    loaded = _load_route_config(conf_path)

    assert all(isinstance(route, RouteConfig) for route in loaded.endpoints)

    routes = [r.to_route() for r in loaded.endpoints]

    for config in loaded.endpoints:
        route = [x for x in routes if x.name == config.name][0]
        assert route.route_type == config.endpoint_type
        assert route.model.name == config.model.name
        assert route.model.provider == config.model.provider
        # Pydantic doesn't allow undefined elements to be a part of its serialized object.
        # This test is a guard for devs only in case we inadvertently add sensitive keys to the
        # Route definition that would be returned via the GetRoute or SearchRoutes APIs
        assert not hasattr(route.model, "config")


def test_invalid_route_definition(tmp_path):
    invalid_conf = {
        "endpoints": [
            {
                "name": "invalid_route",
                "endpoint_type": "invalid/route",
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
            }
        ]
    }
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(invalid_conf))

    with pytest.raises(MlflowException, match=r"The route_type 'invalid/route' is not supported."):
        _load_route_config(conf_path)


def test_invalid_provider(tmp_path):
    invalid_conf = {
        "endpoints": [
            {
                "name": "invalid_route",
                "endpoint_type": "llm/v1/completions",
                "model": {
                    "name": "gpt-4",
                    "provider": "my_provider",
                    "config": {
                        "openai_api_key": "mykey",
                        "openai_api_base": "https://api.openai.com/v1",
                        "openai_api_version": "2023-05-10",
                        "openai_api_type": "openai/v1/chat/completions",
                        "openai_organization": "my_company",
                    },
                },
            }
        ]
    }
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(invalid_conf))

    with pytest.raises(MlflowException, match=r"The provider 'my_provider' is not supported."):
        _load_route_config(conf_path)


def test_invalid_model_definition(tmp_path):
    invalid_partial_config = {
        "endpoints": [
            {
                "name": "some_name",
                "endpoint_type": "llm/v1/completions",
                "model": {
                    "name": "invalid",
                    "provider": "openai",
                    "config": {"openai_api_type": "openai"},
                },
            }
        ]
    }

    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(invalid_partial_config))

    with pytest.raises(
        MlflowException, match=re.compile(r"validation error.+openai_api_key", re.DOTALL)
    ):
        _load_route_config(conf_path)

    invalid_format_config_key_is_not_string = {
        "endpoints": [
            {
                "name": "some_name",
                "endpoint_type": "llm/v1/chat",
                "model": {
                    "name": "invalid",
                    "provider": "openai",
                    "config": {"openai_api_type": "openai", "openai_api_key": [42]},
                },
            }
        ]
    }

    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(invalid_format_config_key_is_not_string))

    with pytest.raises(
        MlflowException,
        match="The api key provided is not a string",
    ):
        _load_route_config(conf_path)

    invalid_format_config_key_invalid_path = {
        "endpoints": [
            {
                "name": "some_name",
                "endpoint_type": "llm/v1/embeddings",
                "model": {
                    "name": "invalid",
                    "provider": "openai",
                    "config": {"openai_api_type": "openai", "openai_api_key": "/not/a/real/path"},
                },
            }
        ]
    }

    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(invalid_format_config_key_invalid_path))

    assert (
        _load_route_config(conf_path).endpoints[0].model.config.openai_api_key == "/not/a/real/path"  # pylint: disable=line-too-long
    )

    invalid_no_config = {
        "endpoints": [
            {
                "name": "some_name",
                "endpoint_type": "llm/v1/embeddings",
                "model": {
                    "name": "invalid",
                    "provider": "anthropic",
                },
            }
        ]
    }
    conf_path = tmp_path.joinpath("config2.yaml")
    conf_path.write_text(yaml.safe_dump(invalid_no_config))

    with pytest.raises(
        MlflowException,
        match="A config must be supplied when setting a provider. The provider entry",
    ):
        _load_route_config(conf_path)


@pytest.mark.parametrize(
    "route_name", ["Space Name", "bang!name", "query?name", "redirect#name", "bracket[]name"]
)
def test_invalid_route_name(tmp_path, route_name):
    bad_name = {
        "endpoints": [
            {
                "name": route_name,
                "endpoint_type": "bad/naming",
                "model": {
                    "name": "claude-v1",
                    "provider": "anthropic",
                    "config": {
                        "anthropic_api_key": "claudekey",
                    },
                },
            }
        ]
    }

    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(bad_name))

    with pytest.raises(
        MlflowException, match="The route name provided contains disallowed characters"
    ):
        _load_route_config(conf_path)


def test_default_base_api(tmp_path):
    route_no_base = {
        "endpoints": [
            {
                "name": "chat-gpt4",
                "endpoint_type": "llm/v1/chat",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                    "config": {"openai_api_key": "sk-openai"},
                },
            },
        ]
    }
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(route_no_base))
    loaded_conf = _load_route_config(conf_path)

    assert loaded_conf.endpoints[0].model.config.openai_api_base == "https://api.openai.com/v1"


def test_duplicate_routes_in_config(tmp_path):
    route = {
        "endpoints": [
            {
                "name": "classifier",
                "endpoint_type": "llm/v1/classifier",
                "model": {
                    "name": "serving-endpoints/document-classifier/Production/invocations",
                    "provider": "databricks-model-serving",
                    "config": {
                        "databricks_api_token": "MY_TOKEN",
                        "databricks_api_base": "https://my-shard-001/",
                    },
                },
            },
            {
                "name": "classifier",
                "endpoint_type": "llm/v1/classifier",
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
    }
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(route))
    with pytest.raises(
        MlflowException, match="Duplicate names found in endpoint configurations. Please"
    ):
        _load_route_config(conf_path)
