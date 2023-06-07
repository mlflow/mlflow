import argparse
from fastapi import FastAPI, Request, HTTPException
import os
from traceback import format_exc
from typing import List
import uvicorn

from mlflow.version import VERSION
from mlflow.gateway.constants import CONF_PATH_ENV_VAR
from mlflow.gateway.handlers import Route, RouteConfig, _route_config_to_route, _load_route_config


# Configured and initialized Gateway Routes state index
ACTIVE_ROUTES: List[Route] = []

app = FastAPI(
    title="MLflow Model Gateway API",
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


def _add_dynamic_route(route: RouteConfig):
    async def route_endpoint(request: Request):
        try:
            # TODO: handle model-specific route logic by mapping the provider to plugin logic

            return await request.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=format_exc()) from e

    app.add_api_route(
        path=f"/gateway/routes/{route.name}", endpoint=route_endpoint, methods=["POST"]
    )
    ACTIVE_ROUTES.append(_route_config_to_route(route))


def _add_routes(routes: List[RouteConfig]):
    for route in routes:
        _add_dynamic_route(route)


def create_app(route_config: List[RouteConfig]) -> FastAPI:
    _add_routes(route_config)
    return app


def initialize_server(host: str, port: int, route_config: List[RouteConfig]):
    server_app = create_app(route_config)
    uvicorn.run(server_app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    conf_path = os.environ.get(CONF_PATH_ENV_VAR)
    route_config = _load_route_config(conf_path)

    initialize_server(args.host, args.port, route_config)
