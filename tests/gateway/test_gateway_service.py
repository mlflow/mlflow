import json
import os

import pytest
import requests
import subprocess
import time
import yaml

from mlflow import gateway
from mlflow.exceptions import MlflowException
from mlflow.gateway import start_server, _stop_server, update_server
from mlflow.gateway.constants import SERVER_HOUSEKEEPING_PAUSE, CONF_PATH_ENV_VAR
from tests.helper_functions import get_safe_port


LOCALHOST = "127.0.0.1"


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


def wait(delay: int = SERVER_HOUSEKEEPING_PAUSE) -> None:
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
    from mlflow.gateway.handlers import _load_gateway_config
    from fastapi.testclient import TestClient

    conf_path = store_conf(path, "config.yaml", config)
    os.environ[CONF_PATH_ENV_VAR] = str(conf_path)

    route_config = _load_gateway_config(conf_path)
    _add_routes(route_config)

    return TestClient(app)


def test_server_start(basic_config_dict, tmp_path):
    conf_path = store_conf(tmp_path, "config.yaml", basic_config_dict)

    assert gateway.server_process is None

    port = get_safe_port()

    start_server(str(conf_path), host=LOCALHOST, port=port)
    wait()

    assert gateway.server_process is not None
    # Test static endpoints
    response = requests.get(f"http://{LOCALHOST}:{port}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

    _stop_server()

    # test that the health check fails (server is down)
    wait()
    with pytest.raises(
        requests.exceptions.ConnectionError, match=r"HTTPConnectionPool\(host='127.0.0.1',"
    ):
        requests.get(f"http://{LOCALHOST}:{port}/health")


def test_cycle_server(basic_config_dict, tmp_path):
    conf_path = store_conf(tmp_path, "config.yaml", basic_config_dict)

    assert gateway.server_process is None

    port = get_safe_port()

    start_server(str(conf_path), host=LOCALHOST, port=port)
    wait()

    assert gateway.server_process is not None

    response = requests.get(f"http://{LOCALHOST}:{port}/gateway_url")
    assert response.status_code == 200
    assert response.json() == {"url": f"http://{LOCALHOST}:{port}"}

    _stop_server()
    wait()

    start_server(str(conf_path), host=LOCALHOST, port=port)
    wait()

    assert gateway.server_process is not None

    response = requests.get(f"http://{LOCALHOST}:{port}/gateway_url")
    assert response.status_code == 200
    assert response.json() == {"url": f"http://{LOCALHOST}:{port}"}
    _stop_server()
    wait()

    start_server(str(conf_path), host=LOCALHOST, port=port)
    wait()

    assert gateway.server_process is not None

    response = requests.get(f"http://{LOCALHOST}:{port}/gateway_url")
    assert response.status_code == 200
    assert response.json() == {"url": f"http://{LOCALHOST}:{port}"}
    _stop_server()


def test_server_update(basic_config_dict, update_config_dict, tmp_path):
    conf_path = store_conf(tmp_path, "config.yaml", basic_config_dict)

    assert gateway.server_process is None

    port = get_safe_port()

    start_server(str(conf_path), host=LOCALHOST, port=port)
    wait()

    assert gateway.server_process is not None

    response = requests.get(f"http://{LOCALHOST}:{port}/gateway_url")
    assert response.status_code == 200
    assert response.json() == {"url": f"http://{LOCALHOST}:{port}"}

    # List all routes
    response = requests.get(f"http://{LOCALHOST}:{port}/list_all_routes")
    assert response.status_code == 200

    route_data = response.json()
    active_routes = set(route["path"] for route in route_data)
    assert {"/gateway_url", "/search_routes", "/instruct-gpt4"}.issubset(active_routes)

    _stop_server()
    wait()

    assert gateway.server_process is None

    with pytest.raises(
        requests.exceptions.ConnectionError, match=r"HTTPConnectionPool\(host='127.0.0.1',"
    ):
        requests.get(f"http://{LOCALHOST}:{port}/health")

    updated_conf = store_conf(tmp_path, "config2.yaml", update_config_dict)

    with pytest.raises(MlflowException, match="There is no currently running gateway server"):
        update_server(str(updated_conf))

    start_server(str(conf_path), host=LOCALHOST, port=port)

    wait()

    update_server(str(updated_conf))

    wait()

    response = requests.get(f"http://{LOCALHOST}:{port}/list_all_routes")
    assert response.status_code == 200

    route_data = response.json()
    active_routes = set(route["path"] for route in route_data)
    assert {"/gateway_url", "/search_routes", "/claude-chat"}.issubset(active_routes)
    assert not {"/instruct-gpt4"}.issubset(active_routes)

    _stop_server()


def test_invalid_server_state_commands(basic_config_dict, tmp_path):
    conf_path = store_conf(tmp_path, "config.yaml", basic_config_dict)
    assert gateway.server_process is None

    port = get_safe_port()

    start_server(str(conf_path), host=LOCALHOST, port=port)
    wait()

    with pytest.raises(MlflowException, match="There is a currently running server instance"):
        start_server(str(conf_path), host=LOCALHOST, port=port)

    _stop_server()
    wait()

    with pytest.raises(MlflowException, match="There is no currently running gateway server"):
        _stop_server()

    with pytest.raises(MlflowException, match="Unable to update server configuration. There is no"):
        update_server(basic_config_dict)


def test_server_static_endpoints(update_config_dict, tmp_path):
    conf_path = store_conf(tmp_path, "config.yaml", update_config_dict)
    port = get_safe_port()
    start_server(str(conf_path), host=LOCALHOST, port=port)
    wait()

    health = get_static_endpoint_response("health", LOCALHOST, port)
    assert health == {"status": "OK"}

    url = get_static_endpoint_response("gateway_url", LOCALHOST, port)
    assert url == {"url": f"http://{LOCALHOST}:{port}"}

    get_route = get_static_endpoint_response("get_route/claude-chat", LOCALHOST, port)
    assert get_route == {
        "route": {
            "model": {"name": "claude-v1", "provider": "anthropic"},
            "name": "claude-chat",
            "type": "llm/v1/chat",
        }
    }

    search_routes = get_static_endpoint_response(
        "search_routes?filter_condition=''", LOCALHOST, port
    )
    assert search_routes == {
        "routes": [
            '{"name": "claude-chat", "type": "llm/v1/chat", "model": {"name": '
            '"claude-v1", "provider": "anthropic"}}'
        ]
    }

    list_routes = get_static_endpoint_response("list_all_routes", LOCALHOST, port)
    assert list_routes == [
        {"methods": ["GET", "HEAD"], "name": "openapi", "path": "/openapi.json"},
        {"methods": ["GET", "HEAD"], "name": "swagger_ui_html", "path": "/docs"},
        {
            "methods": ["GET", "HEAD"],
            "name": "swagger_ui_redirect",
            "path": "/docs/oauth2-redirect",
        },
        {"methods": ["GET", "HEAD"], "name": "redoc_html", "path": "/redoc"},
        {"methods": ["GET"], "name": "health", "path": "/health"},
        {"methods": ["GET"], "name": "gateway_url", "path": "/gateway_url"},
        {"methods": ["GET"], "name": "get_route", "path": "/get_route/{route:path}"},
        {"methods": ["GET"], "name": "search_routes", "path": "/search_routes"},
        {"methods": ["GET"], "name": "list_all_routes", "path": "/list_all_routes"},
        {"methods": ["POST"], "name": "route_endpoint", "path": "/claude-chat"},
    ]

    _stop_server()


def test_request_dynamic_route(basic_config_dict, tmp_path):
    conf_path = store_conf(tmp_path, "config.yaml", basic_config_dict)
    port = get_safe_port()
    start_server(str(conf_path), host=LOCALHOST, port=port)
    wait()

    valid_route = get_dynamic_route_request(
        "instruct-gpt4", {"input": "Tell me a joke"}, LOCALHOST, port
    )
    assert valid_route == {"input": "Tell me a joke"}

    valid_route = get_dynamic_route_request(
        "instruct-gpt4", {"input": "Tell me a joke", "temperature": 0.45}, LOCALHOST, port
    )
    assert valid_route == {"input": "Tell me a joke", "temperature": 0.45}

    _stop_server()


def test_post_to_invalid_route(basic_config_dict, tmp_path):
    conf_path = store_conf(tmp_path, "config.yaml", basic_config_dict)
    port = get_safe_port()
    start_server(str(conf_path), host=LOCALHOST, port=port)
    wait()

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

    _stop_server()


def test_invalid_route_config_terminates_server(invalid_config_dict, tmp_path):
    conf_path = store_conf(tmp_path, "config.yaml", invalid_config_dict)
    os.environ[CONF_PATH_ENV_VAR] = str(conf_path)
    port = get_safe_port()
    process = subprocess.Popen(
        ["python", "-m", "mlflow.gateway.gateway_app", "--host", LOCALHOST, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    _, stderr = process.communicate()

    expected_error = "pydantic.error_wrappers.ValidationError"
    assert expected_error in stderr, f"Expected error '{expected_error}' not found in stderr."


def test_routes(basic_config_dict, tmp_path):
    client = get_test_client(basic_config_dict, tmp_path)

    all_routes = client.get("/list_all_routes")
    all_routes_resp = [route["path"] for route in all_routes.json()]
    # verify dynamic routes created
    assert all(route in all_routes_resp for route in ["/instruct-gpt4", "/claude-chat"])

    resp = client.post("/claude-chat", json={"prompt": "Write my tests for me, please"})
    assert resp.status_code == 200
    assert resp.json() == {"prompt": "Write my tests for me, please"}
