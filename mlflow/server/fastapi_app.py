"""
FastAPI application wrapper for MLflow server.

This module provides a FastAPI application that wraps the existing Flask application
using WSGIMiddleware to maintain 100% API compatibility while enabling future migration
to FastAPI endpoints.
"""

from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware

# Import the existing Flask app
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
        # Enable API documentation for FastAPI endpoints
        docs_url="/fastapi/docs",
        redoc_url="/fastapi/redoc",
        openapi_url="/fastapi/openapi.json",
    )

    # Import and include OTel API router for native FastAPI endpoints
    from mlflow.server.otel_api import otel_router

    fastapi_app.include_router(otel_router)

    # Mount the entire Flask application at the root path
    # This ensures 100% compatibility with existing APIs
    fastapi_app.mount("/", WSGIMiddleware(flask_app))

    return fastapi_app


# Create the app instance that can be used by ASGI servers
app = create_fastapi_app()
