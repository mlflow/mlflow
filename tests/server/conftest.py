import pytest
from flask import Flask, request
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


@pytest.fixture
def setup_middleware(test_app):
    """Factory fixture to set up middleware on a test app."""
    def _setup(middleware):
        @test_app.before_request
        def before_request():
            if response := middleware.process_request(request):
                return response

        @test_app.after_request
        def after_request(response):
            return middleware.process_response(response, request)

    return _setup