from fastapi import FastAPI, HTTPException
import logging
import os
from pathlib import Path
from typing import List, Union

from mlflow.version import VERSION
from mlflow.gateway.config import (
    Route,
    RouteConfig,
    RouteType,
    _route_config_to_route,
    _load_route_config,
)
from mlflow.gateway.schemas import chat, completions, embeddings

_logger = logging.getLogger(__name__)

MLFLOW_GATEWAY_CONFIG = "MLFLOW_GATEWAY_CONFIG"

# Configured and initialized Gateway Routes state index
ACTIVE_ROUTES: List[Route] = []

app = FastAPI(
    title="MLflow Gateway API",
    description="The core gateway API for reverse proxy interface using remote inference "
    "endpoints within MLflow",
    version=VERSION,
)


@app.get("/health")
async def health():
    return {"status": "OK"}


@app.get("/gateway/routes/{route_name}")
async def get_route(route_name: str):
    filtered = next((x for x in ACTIVE_ROUTES if x.name == route_name), None)
    if not filtered:
        raise HTTPException(
            status_code=404,
            detail=f"The route '{route_name}' is not present or active on the server. Please "
            "verify the route name.",
        )
    return {"route": filtered}


@app.get("/gateway/routes/")
async def search_routes():
    # placeholder route listing functionality

    return {"routes": ACTIVE_ROUTES}


async def _chat(request: chat.RequestPayload) -> chat.ResponsePayload:
    return chat.ResponsePayload(**{"candidates": request.messages})


async def _completions(request: completions.RequestPayload) -> completions.ResponsePayload:
    return completions.ResponsePayload(**{"candidates": request.messages})


async def _embeddings(request: embeddings.RequestPayload) -> embeddings.ResponsePayload:
    import random

    length = len(request.text)
    return embeddings.ResponsePayload(
        **{"embeddings": [random.randint(0, length - 1) / length for _ in range(10)]}
    )


def _route_type_to_endpoint(route_type: RouteType):
    if route_type == RouteType.Chat:
        return _chat
    elif route_type == RouteType.Completions:
        return _completions
    # elif route_type == RouteType.Embeddings:
    #     return _embeddings


def _add_dynamic_route(route: RouteConfig):
    app.add_api_route(
        path=f"/gateway/routes/{route.name}",
        endpoint=_route_type_to_endpoint(route.type),
        methods=["POST"],
    )
    ACTIVE_ROUTES.append(_route_config_to_route(route))


def _add_routes(routes: List[RouteConfig]):
    for route in routes:
        _add_dynamic_route(route)


def create_app(gateway_conf_path: Union[str, Path]) -> FastAPI:
    """
    Create the FastAPI app by loading the dynamic route configuration file from the
    specified local path and generating POST methods
    """
    route_config = _load_route_config(gateway_conf_path)
    _add_routes(route_config.routes)
    return app


def create_app_from_env() -> FastAPI:
    """
    Load the path from the environment variable and generate the FastAPI app instance
    """
    gateway_conf_path = os.environ.get(MLFLOW_GATEWAY_CONFIG)

    return create_app(gateway_conf_path)
