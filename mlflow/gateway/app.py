from fastapi import FastAPI
import os

from .config import _validate_config
from mlflow.exceptions import MlflowException


MLFLOW_GATEWAY_CONFIG = "MLFLOW_GATEWAY_CONFIG"


def create_route(s: str):
    async def route():
        return {"message": s}

    return route


def create_app(config_path: str) -> FastAPI:
    config = _validate_config(config_path)
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    for s in config.routes:
        route = create_route(s)

        app.add_api_route(f"/{s}", route, methods=["GET"])

    return app


def create_app_from_env():
    config_path = os.environ.get(MLFLOW_GATEWAY_CONFIG)
    if not config_path:
        raise MlflowException(f"{MLFLOW_GATEWAY_CONFIG} environment variable is not set")
    return create_app(config_path)
