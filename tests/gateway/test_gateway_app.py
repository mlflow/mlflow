from mlflow.gateway.app import create_app
from fastapi.testclient import TestClient
import yaml


def test_create_app(tmp_path):
    config = tmp_path.joinpath("config.yml")
    conf = [
        {
            "name": "some_name",
            "type": "some/type",
            "model": {
                "name": "my_custom_provider",
                "provider": "my_provider",
                "config": {"api_key": "mykey", "api_base": "http://my.endpoint.com/"},
            },
        },
        {
            "name": "another_custom",
            "type": "another/type",
            "model": {
                "name": "another_model_type",
                "provider": "another_provider",
                "config": {"api_key": "a_key_for_my_route", "api_base": "http://my.endpoint.com/"},
            },
        },
    ]
    config.write_text(yaml.safe_dump(conf))

    app = create_app(config)

    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

    response = client.get("/gateway/routes")
    assert response.status_code == 200
    assert response.json() == {
        "routes": [
            {
                "model": {"name": "my_custom_provider", "provider": "custom"},
                "name": "some_name",
                "type": "custom",
            },
            {
                "model": {"name": "another_model_type", "provider": "custom"},
                "name": "another_custom",
                "type": "custom",
            },
        ]
    }

    response = client.get("/gateway/routes/some_name")
    assert response.status_code == 200
    assert response.json() == {
        "route": {
            "model": {"name": "my_custom_provider", "provider": "custom"},
            "name": "some_name",
            "type": "custom",
        }
    }
