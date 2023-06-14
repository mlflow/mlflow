import asyncio
from pathlib import Path
import pytest
import requests
import time

from mlflow.gateway.providers.openai import OpenAIProvider
from mlflow.gateway.runner import run_app
from tests.gateway.tools import Gateway, store_conf
from tests.helper_functions import get_safe_port


@pytest.fixture
def basic_config_dict():
    return {
        "routes": [
            {
                "name": "completions",
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
                "name": "embeddings",
                "type": "llm/v1/embeddings",
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
        ]
    }


@pytest.fixture
def basic_routes():
    return {
        "routes": [
            {
                "name": "completions",
                "type": "llm/v1/completions",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                },
            },
            {
                "name": "embeddings",
                "type": "llm/v1/embeddings",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                },
            },
        ]
    }


@pytest.fixture
def update_config_dict():
    return {
        "routes": [
            {
                "name": "chat",
                "type": "llm/v1/chat",
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
        ]
    }


@pytest.fixture
def update_routes():
    return {
        "routes": [
            {
                "name": "chat",
                "type": "llm/v1/chat",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                },
            },
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
        gateway.wait_reload()
        response = gateway.get("gateway/routes/")

        assert response.json() == update_routes

        # push the original file back
        store_conf(config, basic_config_dict)
        gateway.assert_health()
        gateway.wait_reload()
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
        gateway.wait_reload()
        # push an invalid config
        store_conf(config, invalid_config_dict)
        gateway.assert_health()
        # ensure that filewatch has run through the aborted config change logic
        gateway.wait_reload()
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
        gateway.wait_reload()
        # remove config
        config.unlink()
        gateway.wait_reload()
        gateway.assert_health()

        store_conf(config, {"routes": basic_config_dict["routes"][1:]})
        gateway.wait_reload()
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


@pytest.mark.skip(reason="TODO: Figure out how to test dynamic endpoints")
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


def test_uvicorn_runner_execution(tmp_path, basic_config_dict):
    config = tmp_path / "config.yaml"
    store_conf(config, basic_config_dict)

    port = get_safe_port()
    host = "localhost"
    with run_app(str(config), host=host, port=port, use_gunicorn=False) as server:
        time.sleep(0.5)
        response = requests.get(f"http://{server.host}:{server.port}/health")
        response.raise_for_status()

    with pytest.raises(
        requests.exceptions.ConnectionError, match="Max retries exceeded with url: /health"
    ):
        requests.get(f"http://{host}:{port}/health")


def test_uvicorn_runner_gets_route(tmp_path, basic_config_dict):
    config = tmp_path / "config.yaml"
    store_conf(config, basic_config_dict)

    port = get_safe_port()
    host = "localhost"
    with run_app(str(config), host=host, port=port, use_gunicorn=False) as server:
        time.sleep(0.5)
        response = requests.get(f"http://{server.host}:{server.port}/gateway/routes/embeddings")
        assert response.json() == {
            "route": {
                "model": {"name": "gpt-4", "provider": "openai"},
                "name": "embeddings",
                "type": "llm/v1/embeddings",
            }
        }
    with pytest.raises(
        requests.exceptions.ConnectionError, match="Max retries exceeded with url: /health"
    ):
        requests.get(f"http://{host}:{port}/health")


def test_server_dynamic_endpoints_from_uvicorn(tmp_path, basic_config_dict, monkeypatch):
    config = tmp_path / "config.yaml"
    store_conf(config, basic_config_dict)

    port = get_safe_port()
    host = "localhost"

    async def mock_request(self, path, payload):
        await asyncio.sleep(0.1)  # simulate delay
        return {
            "choices": [{"text": "hello", "finish_reason": None}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "model": "gpt-3.5-turbo",
        }

    monkeypatch.setattr(OpenAIProvider, "_request", mock_request)

    with run_app(str(config), host=host, port=port, use_gunicorn=False) as server:
        time.sleep(0.5)
        response = requests.post(
            url=f"{server.url}/gateway/routes/{basic_config_dict['routes'][0]['name']}",
            json={"prompt": "hello"},
        )

        response.raise_for_status()

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
