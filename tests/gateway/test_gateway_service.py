import pytest
import requests
import time
import yaml

from mlflow import gateway
from mlflow.gateway import start_service, update_service
from mlflow.gateway.gateway_app import app

from tests.helper_functions import get_safe_port


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


def test_server_start(basic_config_dict, tmp_path):
    conf_path = tmp_path.joinpath("config.yaml")
    conf_path.write_text(yaml.safe_dump(basic_config_dict))

    assert gateway.server_manager is None

    port = get_safe_port()

    print(port)

    start_service(str(conf_path), app="mlflow.gateway.gateway_app:app", host="127.0.0.1", port=port)

    time.sleep(2)

    assert gateway.server_manager is not None

    # Test static endpoints
    response = requests.get(f"http://127.0.0.1:{port}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

    print(gateway.server_manager)
    gateway.server_manager._stop_server()

    # test that the health check fails (server is down)

    time.sleep(2)
    with pytest.raises(requests.exceptions.ConnectionError):
        requests.get(f"http://127.0.0.1:{port}/health")


# TODO: test server update, start when already started, update when not started, test
# all static routes
