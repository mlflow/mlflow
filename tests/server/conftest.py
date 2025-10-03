import pytest
from flask import Flask
from werkzeug.test import Client

from mlflow.server import app as mlflow_app


@pytest.fixture
def test_app():
    """Minimal Flask app for unit testing."""
    app = Flask(__name__)

    @app.route("/test")
    def test_endpoint():
        return "OK"

    @app.route("/api/test", methods=["GET", "POST", "OPTIONS"])
    def api_endpoint():
        return "OK"

    @app.route("/health")
    def health():
        return "OK"

    @app.route("/version")
    def version():
        return "OK"

    return app


@pytest.fixture(scope="module")
def mlflow_app_client():
    """Test client for the MLflow Flask application."""
    from mlflow.server import security

    if not hasattr(mlflow_app, "extensions") or "cors" not in mlflow_app.extensions:
        security.init_security_middleware(mlflow_app)
    return Client(mlflow_app)
