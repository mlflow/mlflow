"""
Gateway launcher with overhead measurement middleware.

Provides a create_app() factory for gunicorn/uvicorn that wraps the
standard MLflow Gateway app with OverheadMiddleware.

Usage with gunicorn:
    BENCHMARK_GATEWAY_CONFIG=gateway_config.yaml \
    gunicorn 'run_gateway_with_middleware:create_app()' \
        -k uvicorn.workers.UvicornWorker \
        -w 2 -b 0.0.0.0:5000

Usage with uvicorn (single worker, for debugging):
    BENCHMARK_GATEWAY_CONFIG=gateway_config.yaml \
    uvicorn --factory run_gateway_with_middleware:create_app --port 5000
"""

import os

from overhead_middleware import OverheadMiddleware

from mlflow.gateway.app import create_app_from_path

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "gateway_config.yaml")


def create_app():
    config_path = os.environ.get("BENCHMARK_GATEWAY_CONFIG", DEFAULT_CONFIG)
    gateway_app = create_app_from_path(config_path)
    gateway_app.add_middleware(OverheadMiddleware)
    return gateway_app
