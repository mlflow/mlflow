import json
import os
import sys

import pytest
import requests
import subprocess
import time
import yaml

from mlflow.exceptions import MlflowException
from mlflow.gateway import _start_server, _stop_server, _update_server
from mlflow.gateway.constants import CONF_PATH_ENV_VAR
from mlflow.gateway.utils import _delete_server_state
from tests.helper_functions import get_safe_port, LOCALHOST


class ServerManager:
    """
    Safe context manager to terminate server processes if an error raises in a test
    """

    def __init__(self, config_path, host=LOCALHOST, port=None):
        self.config_path = config_path
        self.host = host
        self.port = port or get_safe_port()
        self.pid = None

    def __enter__(self):
        _delete_server_state()
        server_process = _start_server(self.config_path, self.host, self.port)
        self.pid = server_process.pid
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _stop_server(self.pid)
        wait()


@pytest.fixture
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
def update_config_dict():
    return [
        {
            "name": "claude-chat",
            "type": "llm/v1/chat",
            "model": {
                "name": "claude-v1",
                "provider": "anthropic",
                "config": {
                    "anthropic_api_key_env_var": "MY_ANTHROPIC_KEY",
                },
            },
        },
    ]


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


def wait(delay: int = 2) -> None:
    time.sleep(delay)


def get_static_endpoint_response(endpoint, host, port):
    response = requests.get(f"http://{host}:{port}/{endpoint}")
    assert response.status_code == 200
    return response.json()


def get_dynamic_route_request(route, data, host, port):
    response = requests.post(f"http://{host}:{port}/{route}", json=data)
    assert response.status_code == 200
    return response.json()


def get_invalid_route_request(route, route_type, host, port, error_code, data=None):
    response = requests.request(method=route_type, url=f"http://{host}:{port}/{route}", json=data)
    assert response.status_code == error_code
    return response.text


def get_test_client(config, path):
    from mlflow.gateway.gateway_app import app, _add_routes
    from mlflow.gateway.handlers import _load_route_config
    from fastapi.testclient import TestClient

    conf_path = store_conf(path, "config.yaml", config)
    os.environ[CONF_PATH_ENV_VAR] = str(conf_path)

    route_config = _load_route_config(conf_path)
    _add_routes(route_config)

    return TestClient(app)


def test_server_start(basic_config_dict, tmp_path):
    conf_path = str(store_conf(tmp_path, "config.yaml", basic_config_dict))

    with ServerManager(conf_path) as sm:
        response = requests.get(f"http://{LOCALHOST}:{sm.port}/health")
        assert response.status_code == 200
        assert response.json() == {"status": "OK"}

    wait()
    with pytest.raises(
        requests.exceptions.ConnectionError, match=r"HTTPConnectionPool\(host='127.0.0.1',"
    ):
        requests.get(f"http://{LOCALHOST}:{sm.port}/health")


def test_cycle_server(basic_config_dict, tmp_path):
    conf_path = store_conf(tmp_path, "config.yaml", basic_config_dict)

    port = get_safe_port()

    with ServerManager(str(conf_path), LOCALHOST, port):
        response = requests.get(f"http://{LOCALHOST}:{port}/health")
        assert response.status_code == 200
        assert response.json() == {"status": "OK"}

    with ServerManager(str(conf_path), LOCALHOST, port):
        response = requests.get(f"http://{LOCALHOST}:{port}/health")
        assert response.status_code == 200
        assert response.json() == {"status": "OK"}

    with ServerManager(str(conf_path), LOCALHOST, port):
        response = requests.get(f"http://{LOCALHOST}:{port}/health")
        assert response.status_code == 200
        assert response.json() == {"status": "OK"}


def test_server_update(basic_config_dict, update_config_dict, tmp_path):
    conf_path = store_conf(tmp_path, "config.yaml", basic_config_dict)
    updated_conf = store_conf(tmp_path, "config2.yaml", update_config_dict)

    port = get_safe_port()

    with ServerManager(str(conf_path), port=port):
        # List all routes
        response = requests.get(f"http://{LOCALHOST}:{port}/gateway/routes/")
        assert response.status_code == 200

        route_data = response.json()
        active_routes = set(route["name"] for route in route_data["routes"])
        assert {"claude-chat", "instruct-gpt4"}.issubset(active_routes)

    # Test that the health check fails (server is down)
    with pytest.raises(
        requests.exceptions.ConnectionError, match=r"HTTPConnectionPool\(host='127.0.0.1',"
    ):
        requests.get(f"http://{LOCALHOST}:{port}/health")

    with pytest.raises(MlflowException, match="Unable to update server configuration. There is no"):
        _update_server(str(updated_conf))

    with ServerManager(str(conf_path), port=port) as sm:
        server_process = _update_server(str(updated_conf))
        # NB: the update func call will generate a new pid other than the pid that was
        # initiated during server start from the context manager. Update it so that the
        # on exit server shutdown has reference to the newly started process id.
        sm.pid = server_process.pid

        response = requests.get(f"http://{LOCALHOST}:{port}/gateway/routes/")
        assert response.status_code == 200

        route_data = response.json()
        active_routes = set(route["name"] for route in route_data["routes"])
        assert {"claude-chat"}.issubset(active_routes)
        assert not {"instruct-gpt4"}.issubset(active_routes)


def test_invalid_server_state_commands(basic_config_dict, tmp_path):
    conf_path = store_conf(tmp_path, "config.yaml", basic_config_dict)

    port = get_safe_port()

    with ServerManager(str(conf_path), host=LOCALHOST, port=port) as sm:
        with pytest.raises(MlflowException, match="There is a currently running server instance"):
            _start_server(str(conf_path), host=LOCALHOST, port=port)

    wait()
    # At this point, the server should have stopped due to the context manager
    with pytest.raises(MlflowException, match="There is no currently running gateway server"):
        _stop_server(sm.pid)

    with pytest.raises(MlflowException, match="Unable to update server configuration. There is no"):
        _update_server(basic_config_dict)


def test_server_static_endpoints(update_config_dict, tmp_path):
    conf_path = store_conf(tmp_path, "config.yaml", update_config_dict)
    port = get_safe_port()

    with ServerManager(str(conf_path), host=LOCALHOST, port=port):
        health = get_static_endpoint_response("health", LOCALHOST, port)
        assert health == {"status": "OK"}

        get_route = get_static_endpoint_response("gateway/routes/claude-chat", LOCALHOST, port)
        assert get_route == {
            "route": {
                "model": {"name": "claude-v1", "provider": "anthropic"},
                "name": "claude-chat",
                "type": "llm/v1/chat",
            }
        }

        search_routes = get_static_endpoint_response("gateway/routes/", LOCALHOST, port)
        assert search_routes == {
            "routes": [
                {
                    "model": {"name": "claude-v1", "provider": "anthropic"},
                    "name": "claude-chat",
                    "type": "llm/v1/chat",
                }
            ]
        }


def test_request_dynamic_route(basic_config_dict, tmp_path):
    conf_path = store_conf(tmp_path, "config.yaml", basic_config_dict)
    port = get_safe_port()

    with ServerManager(str(conf_path), host=LOCALHOST, port=port):
        valid_route = get_dynamic_route_request(
            "gateway/routes/instruct-gpt4", {"input": "Tell me a joke"}, LOCALHOST, port
        )
        assert valid_route == {"input": "Tell me a joke"}

        valid_route = get_dynamic_route_request(
            "gateway/routes/instruct-gpt4",
            {"input": "Tell me a joke", "temperature": 0.45},
            LOCALHOST,
            port,
        )
        assert valid_route == {"input": "Tell me a joke", "temperature": 0.45}


def test_post_to_invalid_route(basic_config_dict, tmp_path):
    conf_path = store_conf(tmp_path, "config.yaml", basic_config_dict)
    port = get_safe_port()

    with ServerManager(str(conf_path), host=LOCALHOST, port=port):
        invalid_get = get_invalid_route_request(
            route="not_a_real_endpoint", route_type="get", error_code=404, host=LOCALHOST, port=port
        )

        assert json.loads(invalid_get) == {"detail": "Not Found"}

        invalid_post = get_invalid_route_request(
            route="also/not/real",
            route_type="post",
            error_code=404,
            host=LOCALHOST,
            port=port,
            data={"some": "bogus data"},
        )

        assert json.loads(invalid_post) == {"detail": "Not Found"}


def test_invalid_route_config_terminates_server(invalid_config_dict, tmp_path):
    conf_path = store_conf(tmp_path, "config.yaml", invalid_config_dict)
    os.environ[CONF_PATH_ENV_VAR] = str(conf_path)
    port = get_safe_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow.gateway.gateway_app",
            "--host",
            LOCALHOST,
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    _, stderr = process.communicate()

    expected_error = "pydantic.error_wrappers.ValidationError"
    assert expected_error in stderr, f"Expected error '{expected_error}' not found in stderr."


def test_routes(basic_config_dict, tmp_path):
    client = get_test_client(basic_config_dict, tmp_path)

    all_routes = client.get("/gateway/routes/")
    all_routes_resp = [route["name"] for route in all_routes.json()["routes"]]
    # verify dynamic routes created
    assert all(route in all_routes_resp for route in ["instruct-gpt4", "claude-chat"])

    resp = client.post(
        "/gateway/routes/claude-chat", json={"prompt": "Write my tests for me, please"}
    )
    assert resp.status_code == 200
    assert resp.json() == {"prompt": "Write my tests for me, please"}
