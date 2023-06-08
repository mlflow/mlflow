from mlflow.gateway.app import create_app
from fastapi.testclient import TestClient


def test_create_app(tmp_path):
    config = tmp_path.joinpath("config.yml")
    config.write_text(
        """
routes:
    - a
    - b
"""
    )
    app = create_app(config)

    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    response = client.get("/a")
    assert response.status_code == 200
    assert response.json() == {"message": "a"}

    response = client.get("/b")
    assert response.status_code == 200
    assert response.json() == {"message": "b"}
