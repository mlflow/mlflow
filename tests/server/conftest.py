import pytest
from fastapi import FastAPI
from flask import Flask
from starlette.testclient import TestClient
from werkzeug.test import Client

from mlflow.server.fastapi_security import init_fastapi_security


@pytest.fixture
def test_app():
    """Minimal Flask app for unit testing."""
    app = Flask(__name__)

    @app.route("/test")
    def test_endpoint():
        return "OK"

    @app.route("/api/2.0/mlflow/experiments/list", methods=["GET", "POST", "OPTIONS"])
    def api_endpoint():
        return "OK"

    @app.route("/health")
    def health():
        return "OK"

    @app.route("/version")
    def version():
        return "OK"

    return app


@pytest.fixture
def test_app_context(test_app):
    with test_app.app_context():
        yield


@pytest.fixture
def mlflow_app_client():
    """Test client for the MLflow Flask application with security middleware."""
    from flask import Flask

    from mlflow.server import handlers, security

    # Create a fresh app for each test to avoid state pollution
    app = Flask(__name__)
    for http_path, handler, methods in handlers.get_endpoints():
        app.add_url_rule(http_path, handler.__name__, handler, methods=methods)

    security.init_security_middleware(app)
    return Client(app)


@pytest.fixture
def fastapi_client():
    """Minimal FastAPI app for unit testing."""
    app = FastAPI()

    @app.api_route("/api/2.0/mlflow/experiments/list", methods=["GET", "POST", "OPTIONS"])
    async def api_endpoint():
        return {"ok": True}

    init_fastapi_security(app)
    return TestClient(app, raise_server_exceptions=False)
