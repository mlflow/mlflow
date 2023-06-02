from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from typing import Dict, Any, List

from mlflow.gateway.config import GatewayConfigSingleton
from mlflow.gateway.handlers import Route
from mlflow.gateway.utils import _parse_url_path_for_base_url

GATEWAY_VERSION = "0.1.0"
# Configured and initialized Gateway Routes state index
ACTIVE_ROUTES: List[Route] = []

# Get the routes configuration from the singleton instance
route_config = GatewayConfigSingleton.getInstance().gateway_config

app = FastAPI(
    titls="MLflow Model Gateway API",
    description="The core gateway API for reverse proxy interface using remote inference "
    "endpoints within MLflow",
    version=GATEWAY_VERSION,
)


@app.get("/health")
async def health():
    return {"status": "OK"}


@app.get("/gateway_url")
async def gateway_url(request: Request):
    return {"Gateway URL": _parse_url_path_for_base_url(str(request.url))}


@app.get("/get_route/{route:path}")
async def get_route(route: str):
    filtered = [x for x in ACTIVE_ROUTES if x.name == route]
    pass


@app.get("/search_routes")
async def search_routes(filter: str):
    # placeholder route listing functionality
    routes = [route.json() for route in ACTIVE_ROUTES]
    return {"routes": routes}


# TODO: build out dynamic routes for the FastAPI app
