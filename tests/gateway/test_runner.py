from pathlib import Path

from tests.gateway.helper_functions import Gateway, store_conf, wait
import pytest


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
                        "openai_api_version": "2023-05-15",
                        "openai_api_type": "open_ai",
                    },
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
    }


@pytest.fixture
def basic_routes():
    return {
        "routes": [
            {
                "name": "completions-gpt4",
                "type": "llm/v1/completions",
                "model": {"name": "gpt-4", "provider": "openai"},
            },
            {
                "name": "claude-chat",
                "type": "llm/v1/chat",
                "model": {"name": "claude-v1", "provider": "anthropic"},
            },
        ]
    }


@pytest.fixture
def update_config_dict():
    return {
        "routes": [
            {
                "name": "claude-completions",
                "type": "llm/v1/completions",
                "model": {
                    "name": "claude-v1",
                    "provider": "anthropic",
                    "config": {
                        "anthropic_api_key": "MY_ANTHROPIC_KEY",
                    },
                },
            },
        ]
    }


@pytest.fixture
def update_routes():
    return {
        "routes": [
            {
                "model": {"name": "claude-v1", "provider": "anthropic"},
                "name": "claude-completions",
                "type": "llm/v1/completions",
            }
        ]
    }


@pytest.fixture
def invalid_config_dict():
    return {
        "routes": [
            {
                "invalid_name": "invalid",
                "type": "llm/v1/chat",
                "model": {"invalidkey": "invalid", "invalid_provider": "invalid"},
            }
        ]
    }


def test_server_update(
    tmp_path: Path, basic_config_dict, update_config_dict, basic_routes, update_routes
):
    config = tmp_path / "config.yaml"
    store_conf(config, basic_config_dict)

    with Gateway(config) as gateway:
        response = gateway.get("gateway/routes/")
        assert response.json() == basic_routes

        # push an update to the config file
        store_conf(config, update_config_dict)

        # Ensure there is no server downtime
        gateway.assert_health()

        # Wait for the app to restart
        wait()
        response = gateway.get("gateway/routes/")

        assert response.json() == update_routes

        # push the original file back
        store_conf(config, basic_config_dict)
        gateway.assert_health()
        wait()
        response = gateway.get("gateway/routes/")
        assert response.json() == basic_routes


def test_server_update_with_invalid_config(
    tmp_path: Path, basic_config_dict, invalid_config_dict, basic_routes
):
    config = tmp_path / "config.yaml"
    store_conf(config, basic_config_dict)

    with Gateway(config) as gateway:
        response = gateway.get("gateway/routes/")
        assert response.json() == basic_routes
        # Give filewatch a moment to cycle
        wait()
        # push an invalid config
        store_conf(config, invalid_config_dict)
        gateway.assert_health()
        # ensure that filewatch has run through the aborted config change logic
        wait()
        gateway.assert_health()
        response = gateway.get("gateway/routes/")
        assert response.json() == basic_routes


def test_server_update_config_removed_then_recreated(
    tmp_path: Path, basic_config_dict, basic_routes
):
    config = tmp_path / "config.yaml"
    store_conf(config, basic_config_dict)

    with Gateway(config) as gateway:
        response = gateway.get("gateway/routes/")
        assert response.json() == basic_routes
        # Give filewatch a moment to cycle
        wait()
        # remove config
        config.unlink()
        wait()
        gateway.assert_health()

        store_conf(config, {"routes": basic_config_dict["routes"][1:]})
        wait()
        response = gateway.get("gateway/routes/")
        assert response.json() == {"routes": basic_routes["routes"][1:]}


def test_server_static_endpoints(tmp_path, basic_config_dict, basic_routes):
    config = tmp_path / "config.yaml"
    store_conf(config, basic_config_dict)

    with Gateway(config) as gateway:
        response = gateway.get("gateway/routes/")
        assert response.json() == basic_routes

        for route in ["docs", "redoc"]:
            response = gateway.get(route)
            assert response.status_code == 200

        for index, route in enumerate(basic_config_dict["routes"]):
            response = gateway.get(f"gateway/routes/{route['name']}")
            assert response.json() == {"route": basic_routes["routes"][index]}


def test_server_dynamic_endpoints(tmp_path, basic_config_dict):
    config = tmp_path / "config.yaml"
    store_conf(config, basic_config_dict)

    with Gateway(config) as gateway:
        response = gateway.post(
            f"gateway/routes/{basic_config_dict['routes'][0]['name']}",
            json={"prompt": "hello"},
        )
        assert response.json() == {
            "candidates": [{"text": "hello", "metadata": {"finish_reason": None}}],
            "metadata": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "model": "gpt-3.5-turbo",
                "route_type": "llm/v1/completions",
            },
        }

        # response = gateway.post(
        #     f"gateway/routes/{basic_config_dict['routes'][1]['name']}",
        #     json={"input": "Say hello", "temperature": 0.35},
        # )
        # assert response.json() == {"input": "Say hello", "temperature": 0.35}


def test_request_invalid_route(tmp_path, basic_config_dict):
    config = tmp_path / "config.yaml"
    store_conf(config, basic_config_dict)

    with Gateway(config) as gateway:
        # Test get
        response = gateway.get("gateway/routes/invalid/")
        assert response.status_code == 404
        assert response.json() == {
            "detail": "The route 'invalid' is not present or active on the server. Please "
            "verify the route name."
        }

        # Test post
        response = gateway.post("gateway/routes/invalid", json={"input": "should fail"})
        assert response.status_code == 405
        assert response.json() == {"detail": "Method Not Allowed"}
