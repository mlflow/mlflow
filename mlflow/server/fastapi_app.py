"""
FastAPI application wrapper for MLflow server.

This module provides a FastAPI application that wraps the existing Flask application
using WSGIMiddleware to maintain 100% API compatibility while enabling future migration
to FastAPI endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from mlflow.server import app as flask_app
from mlflow.version import VERSION


def create_fastapi_app():
    """
    Create a FastAPI application that wraps the existing Flask app.

    Returns:
        FastAPI application instance with the Flask app mounted via WSGIMiddleware.
    """
    # Create FastAPI app with metadata
    fastapi_app = FastAPI(
        title="MLflow Tracking Server",
        description="MLflow Tracking Server API",
        version=VERSION,
        # TODO: Enable API documentation when we have native FastAPI endpoints
        # For now, disable docs since we only have Flask routes via WSGI
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # Mount the entire Flask application at the root path
    # This ensures compatibility with existing APIs
    fastapi_app.mount("/", WSGIMiddleware(flask_app))

    return fastapi_app


# Create the app instance that can be used by ASGI servers
app = create_fastapi_app()
