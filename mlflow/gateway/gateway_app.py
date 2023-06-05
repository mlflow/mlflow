import argparse
from fastapi import FastAPI, Request, HTTPException
import os
from pydantic import BaseModel
from traceback import format_exc
from typing import List, Optional
import uvicorn

from mlflow.gateway.constants import GATEWAY_VERSION, CONF_PATH_ENV_VAR
from mlflow.gateway.handlers import Route, RouteConfig, _route_config_to_route, _load_gateway_config
from mlflow.gateway.utils import _parse_url_path_for_base_url

# Configured and initialized Gateway Routes state index
ACTIVE_ROUTES: List[Route] = []

app = FastAPI(
    title="MLflow Model Gateway API",
    description="The core gateway API for reverse proxy interface using remote inference "
    "endpoints within MLflow",
    version=GATEWAY_VERSION,
)


@app.get("/health")
async def health():
    return {"status": "OK"}


@app.get("/gateway_url")
async def gateway_url(request: Request):
    return {"url": _parse_url_path_for_base_url(str(request.url))}


@app.get("/get_route/{route:path}")
async def get_route(route: str):
    filtered = [x for x in ACTIVE_ROUTES if x.name == route]
    if not filtered:
        raise HTTPException(
            status_code=404,
            detail=f"The route '{route}' is not present or active on the server. Please "
            "verify the route name.",
        )
    return {"route": filtered[0]}


@app.get("/search_routes")
async def search_routes(filter: str):
    # placeholder route listing functionality
    routes = [route.json() for route in ACTIVE_ROUTES]
    return {"routes": routes}


@app.get("/list_all_routes")
async def list_all_routes():
    routes = []
    for route in app.routes:
        routes.append(
            {
                "path": route.path,
                "name": route.name,
                "methods": sorted(route.methods),
            }
        )
    return routes


# Placeholder logic for dynamic route configuration
def _add_dynamic_route(route: RouteConfig):
    # TODO: Provide a mapping that returns the constructed async endpoint config from the
    # provider within the RouteConfig. The `route_endpoint` configuration below is a
    # placeholder only and should be generated based on the expected endpoint configuration
    # within the given RouteConfig. The pydantic class below is also a placeholder and should
    # be externalized.

    class RouteInput(BaseModel):
        input: str
        temperature: Optional[float]

    async def route_endpoint(data: RouteInput):
        try:
            # TODO: handle model-specific route logic by mapping the provider to plugin logic

            return {"response": data.input}
        except Exception as e:
            raise HTTPException(status_code=500, detail=format_exc()) from e

    app.add_api_route(path=f"/{route.name}", endpoint=route_endpoint, methods=["POST"])
    ACTIVE_ROUTES.append(_route_config_to_route(route))


def _add_routes(routes: List[RouteConfig]):
    for route in routes:
        _add_dynamic_route(route)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    conf_path = os.environ.get(CONF_PATH_ENV_VAR)
    route_config = _load_gateway_config(conf_path)
    _add_routes(route_config)

    uvicorn.run(app, host=args.host, port=args.port)
