import subprocess
import time
import requests
import signal
import os
from pathlib import Path
from typing import Dict, Any
import yaml

from tests.helper_functions import get_safe_port

import pytest
from mlflow.gateway.utils import kill_child_processes


class Gateway(subprocess.Popen):
    def __init__(self, config_path: str, *args, **kwargs):
        self.port = get_safe_port()
        self.host = "localhost"
        super().__init__(
            [
                "mlflow",
                "gateway",
                "start",
                "--config-path",
                str(config_path),
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--workers",
                "2",
            ],
            *args,
            **kwargs,
            preexec_fn=os.setsid,
        )
        self.wait_until_ready()

    def wait_until_ready(self) -> None:
        s = time.time()
        while time.time() - s < 10:
            try:
                if self.request("health").ok:
                    return
            except requests.exceptions.ConnectionError:
                time.sleep(0.1)

        raise Exception("Gateway failed to start")

    def request(self, path: str) -> requests.Response:
        return requests.get(f"http://{self.host}:{self.port}/{path}")

    def post(self, path: str, query: Dict[str, Any]) -> requests.Response:
        return requests.post(f"http://{self.host}:{self.port}/{path}", json=query)

    def __exit__(self, *args, **kwargs):
        kill_child_processes(self.pid)
        os.kill(self.pid, signal.SIGTERM)  # kill the master process first

        return super().__exit__(*args, **kwargs)


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
                    "openai_api_version": "v1",
                    "openai_api_type": "completions",
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


@pytest.fixture
def basic_routes():
    return {
        "routes": [
            {
                "name": "instruct-gpt4",
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
    return [
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
    return [
        {
            "invalid_name": "invalid",
            "type": "llm/v1/chat",
            "model": {"invalidkey": "invalid", "invalid_provider": "invalid"},
        }
    ]


def store_conf(path, name, conf):
    conf_path = path.joinpath(name)
    conf_path.write_text(yaml.safe_dump(conf))
    return conf_path


def wait():
    """
    A sleep statement for testing purposes only to ensure that the file watch and app reload
    has enough time to resolve to updated endpoints.
    """
    time.sleep(2)


def check_health(gateway: Gateway):
    health_check = gateway.request("health/")
    assert health_check.status_code == 200


def test_server_update(
    tmp_path: Path, basic_config_dict, update_config_dict, basic_routes, update_routes
):
    config = str(store_conf(tmp_path, "config.yaml", basic_config_dict))

    with Gateway(config) as gateway:

        response = gateway.request("gateway/routes/")
        assert response.json() == basic_routes

        # push an update to the config file
        store_conf(tmp_path, "config.yaml", update_config_dict)

        # Ensure there is no server downtime
        check_health(gateway)

        # Wait for the app to restart
        wait()
        response = gateway.request("gateway/routes/")

        assert response.json() == update_routes

        # push the original file back
        store_conf(tmp_path, "config.yaml", basic_config_dict)
        check_health(gateway)
        wait()
        response = gateway.request("gateway/routes/")
        assert response.json() == basic_routes


def test_server_update_with_invalid_config(
    tmp_path: Path, basic_config_dict, invalid_config_dict, basic_routes
):
    config = str(store_conf(tmp_path, "config.yaml", basic_config_dict))

    with Gateway(config) as gateway:
        response = gateway.request("gateway/routes/")
        assert response.json() == basic_routes
        # Give filewatch a moment to cycle
        wait()
        # push an invalid config
        store_conf(tmp_path, "config.yaml", invalid_config_dict)
        check_health(gateway)
        # ensure that filewatch has run through the aborted config change logic
        wait()
        check_health(gateway)
        response = gateway.request("gateway/routes/")
        assert response.json() == basic_routes


def test_server_static_endpoints(tmp_path, basic_config_dict, basic_routes):
    config = str(store_conf(tmp_path, "config.yaml", basic_config_dict))

    with Gateway(config) as gateway:
        response = gateway.request("gateway/routes/")
        assert response.json() == basic_routes

        for route in ["docs", "redoc"]:
            response = gateway.request(route)
            assert response.status_code == 200

        for i in range(len(basic_config_dict)):
            response = gateway.request(f"gateway/routes/{basic_config_dict[i]['name']}")
            assert response.json() == {"route": basic_routes["routes"][i]}


def test_server_dynamic_endpoints(tmp_path, basic_config_dict):
    config = str(store_conf(tmp_path, "config.yaml", basic_config_dict))

    with Gateway(config) as gateway:

        response = gateway.post(
            f"gateway/routes/{basic_config_dict[0]['name']}", {"input": "Tell me a joke"}
        )
        assert response.json() == {"input": "Tell me a joke"}

        response = gateway.post(
            f"gateway/routes/{basic_config_dict[1]['name']}",
            {"input": "Say hello", "temperature": 0.35},
        )
        assert response.json() == {"input": "Say hello", "temperature": 0.35}


def test_request_invalid_route(tmp_path, basic_config_dict):
    config = str(store_conf(tmp_path, "config.yaml", basic_config_dict))

    with Gateway(config) as gateway:
        # Test get
        response = gateway.request("gateway/routes/invalid/")
        assert response.status_code == 404
        assert response.json() == {
            "detail": "The route 'invalid' is not present or active on the server. Please "
                      "verify the route name."
        }

        # Test post
        response = gateway.post("gateway/routes/invalid", {"input": "should fail"})
        assert response.status_code == 405
        assert response.json() == {"detail": "Method Not Allowed"}
