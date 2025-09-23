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


@pytest.fixture
def mlflow_app_client():
    """Test client for the MLflow Flask application."""
    return Client(mlflow_app)
