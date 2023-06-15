from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
import logging
import os
from pathlib import Path
from typing import List, Union

from mlflow.version import VERSION
from mlflow.gateway.constants import MLFLOW_GATEWAY_HEALTH_ENDPOINT
from mlflow.gateway.config import (
    Route,
    RouteConfig,
    RouteType,
    _load_route_config,
)
from mlflow.gateway.schemas import chat, completions, embeddings
from .providers import get_provider
from .schemas import chat, completions, embeddings

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


@app.get("/")
async def index():
    return RedirectResponse(url="/docs")


@app.get(MLFLOW_GATEWAY_HEALTH_ENDPOINT)
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


def _create_chat_endpoint(config: RouteConfig):
    prov = get_provider(config.model.provider)(config)

    async def _chat(payload: chat.RequestPayload) -> chat.ResponsePayload:
        return await prov.chat(payload)

    return _chat


def _create_completions_endpoint(config: RouteConfig):
    prov = get_provider(config.model.provider)(config)

    async def _completions(
        payload: completions.RequestPayload,
    ) -> completions.ResponsePayload:
        return await prov.completions(payload)

    return _completions


def _create_embeddings_endpoint(config: RouteConfig):
    prov = get_provider(config.model.provider)(config)

    async def _embeddings(payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        return await prov.embeddings(payload)

    return _embeddings


async def _custom(request: Request):
    return request.json()


def _route_type_to_endpoint(config: RouteConfig):
    provider_to_factory = {
        RouteType.LLM_V1_CHAT: _create_chat_endpoint,
        RouteType.LLM_V1_COMPLETIONS: _create_completions_endpoint,
        RouteType.LLM_V1_EMBEDDINGS: _create_embeddings_endpoint,
    }
    if factory := provider_to_factory.get(config.type):
        return factory(config)

    raise HTTPException(
        status_code=404,
        detail=f"Unexpected route type {config.type!r} for route {config.name!r}.",
    )


def _add_dynamic_route(route_config: RouteConfig):
    app.add_api_route(
        path=f"/gateway/routes/{route_config.name}",
        endpoint=_route_type_to_endpoint(route_config),
        methods=["POST"],
    )
    ACTIVE_ROUTES.append(route_config.to_route())


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
