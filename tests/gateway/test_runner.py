from pathlib import Path

import pytest

from tests.gateway.tools import Gateway, save_yaml

BASE_ROUTE = "/api/2.0/endpoints/"


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
                        "openai_api_version": "2023-05-15",
                        "openai_api_type": "openai",
                    },
                },
            },
            {
                "name": "embeddings-gpt4",
                "endpoint_type": "llm/v1/embeddings",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                    "config": {
                        "openai_api_key": "mykey",
                        "openai_api_base": "https://api.openai.com/v1",
                        "openai_api_version": "2023-05-15",
                        "openai_api_type": "openai",
                    },
                },
            },
        ]
    }


@pytest.fixture
def basic_routes():
    return [
        {
            "name": "completions-gpt4",
            "endpoint_type": "llm/v1/completions",
            "endpoint_url": "/gateway/completions-gpt4/invocations",
            "model": {
                "name": "gpt-4",
                "provider": "openai",
            },
            "limit": None,
        },
        {
            "name": "embeddings-gpt4",
            "endpoint_type": "llm/v1/embeddings",
            "endpoint_url": "/gateway/embeddings-gpt4/invocations",
            "model": {
                "name": "gpt-4",
                "provider": "openai",
            },
            "limit": None,
        },
    ]


@pytest.fixture
def update_config_dict():
    return {
        "endpoints": [
            {
                "name": "chat-gpt4",
                "endpoint_type": "llm/v1/chat",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                    "config": {
                        "openai_api_key": "mykey",
                        "openai_api_base": "https://api.openai.com/v1",
                        "openai_api_version": "2023-05-15",
                        "openai_api_type": "openai",
                    },
                },
                "limit": None,
            },
        ]
    }


@pytest.fixture
def update_routes():
    return [
        {
            "name": "chat-gpt4",
            "endpoint_type": "llm/v1/chat",
            "endpoint_url": "/gateway/chat-gpt4/invocations",
            "model": {
                "name": "gpt-4",
                "provider": "openai",
            },
            "limit": None,
        },
    ]


@pytest.fixture
def invalid_config_dict():
    return {
        "endpoints": [
            {
                "invalid_name": "invalid",
                "endpoint_type": "llm/v1/chat",
                "model": {"invalidkey": "invalid", "invalid_provider": "invalid"},
            }
        ]
    }


def test_server_update(
    tmp_path: Path, basic_config_dict, update_config_dict, basic_routes, update_routes
):
    config = tmp_path / "config.yaml"
    save_yaml(config, basic_config_dict)

    with Gateway(config) as gateway:
        response = gateway.get(BASE_ROUTE)
        assert response.json()["endpoints"] == basic_routes

        # push an update to the config file
        save_yaml(config, update_config_dict)

        # Ensure there is no server downtime
        gateway.assert_health()

        # Wait for the app to restart
        gateway.wait_reload()
        response = gateway.get(BASE_ROUTE)

        assert response.json()["endpoints"] == update_routes

        # push the original file back
        save_yaml(config, basic_config_dict)
        gateway.assert_health()
        gateway.wait_reload()
        response = gateway.get(BASE_ROUTE)
        assert response.json()["endpoints"] == basic_routes


def test_server_update_with_invalid_config(
    tmp_path: Path, basic_config_dict, invalid_config_dict, basic_routes
):
    config = tmp_path / "config.yaml"
    save_yaml(config, basic_config_dict)

    with Gateway(config) as gateway:
        response = gateway.get(BASE_ROUTE)
        assert response.json()["endpoints"] == basic_routes
        # Give filewatch a moment to cycle
        gateway.wait_reload()
        # push an invalid config
        save_yaml(config, invalid_config_dict)
        gateway.assert_health()
        # ensure that filewatch has run through the aborted config change logic
        gateway.wait_reload()
        gateway.assert_health()
        response = gateway.get(BASE_ROUTE)
        assert response.json()["endpoints"] == basic_routes


def test_server_update_config_removed_then_recreated(
    tmp_path: Path, basic_config_dict, basic_routes
):
    config = tmp_path / "config.yaml"
    save_yaml(config, basic_config_dict)

    with Gateway(config) as gateway:
        response = gateway.get(BASE_ROUTE)
        assert response.json()["endpoints"] == basic_routes
        # Give filewatch a moment to cycle
        gateway.wait_reload()
        # remove config
        config.unlink()
        gateway.wait_reload()
        gateway.assert_health()

        save_yaml(config, {"endpoints": basic_config_dict["endpoints"][1:]})
        gateway.wait_reload()
        response = gateway.get(BASE_ROUTE)
        assert response.json()["endpoints"] == basic_routes[1:]


def test_server_static_endpoints(tmp_path, basic_config_dict, basic_routes):
    config = tmp_path / "config.yaml"
    save_yaml(config, basic_config_dict)

    with Gateway(config) as gateway:
        response = gateway.get(BASE_ROUTE)
        assert response.json()["endpoints"] == basic_routes

        for route in ["docs", "redoc"]:
            response = gateway.get(route)
            assert response.status_code == 200

        for index, route in enumerate(basic_config_dict["endpoints"]):
            response = gateway.get(f"{BASE_ROUTE}{route['name']}")
            assert response.json() == basic_routes[index]


def test_request_invalid_route(tmp_path, basic_config_dict):
    config = tmp_path / "config.yaml"
    save_yaml(config, basic_config_dict)

    with Gateway(config) as gateway:
        # Test get
        response = gateway.get(f"{BASE_ROUTE}invalid/")
        assert response.status_code == 404
        assert response.json() == {
            "detail": "The endpoint 'invalid' is not present or active on the server. Please "
            "verify the endpoint name."
        }

        # Test post
        response = gateway.post(f"{BASE_ROUTE}invalid", json={"input": "should fail"})
        assert response.status_code == 405
        assert response.json() == {"detail": "Method Not Allowed"}
