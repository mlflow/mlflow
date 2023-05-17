import argparse
import signal
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import os
import sys
from typing import Dict, Callable

from mlflow.proxy.proxy_service_constants import _Route
from mlflow.proxy.utils import (
    _error_response,
    _load_route_config,
    _get_route_limits,
    _parse_request_path,
    _parse_url_path_for_base_url,
)

app = FastAPI()

active_routes: Dict[str, _Route] = {}

config_path = os.environ["MLFLOW_PROXY_CONFIG_PATH"]
configs = _load_route_config(config_path)
route_limits = _get_route_limits(configs)

request_counts: Dict[str, int] = {}
first_requests: Dict[str, datetime] = {}

TIME_WINDOW = timedelta(minutes=1)

PID = os.getpid()


@app.get("/health")
def health():
    return {"status": "OK"}


@app.get("/route_statuses")
async def get_route_statuses():
    return active_routes


@app.get("/disable_route/{route:path}")
async def disable_route(route: str):
    if route in active_routes:
        if active_routes[route].active:
            active_routes[route].active = False
            return {"status": f"Route {route} disabled"}
        else:
            return _error_response(405, f"Route {route} is already disabled")
    else:
        return _error_response(404, f"There is no route {route} activated on this server")


@app.get("/enable_route/{route:path}")
async def enable_route(route: str):
    if route in active_routes:
        if not active_routes[route].active:
            active_routes[route].active = True
            return {"status": f"Route {route} enabled"}
        else:
            _error_response(405, f"Route {route} is already enabled")
    else:
        return _error_response(404, f"There is no route {route} actived on this server")


@app.get("/proxy_endpoint_url")
async def proxy_endpoint_url(request: Request):
    return {"Proxy URL": _parse_url_path_for_base_url(str(request.url))}


@app.get("/disable_all_routes")
async def disable_all_routes():
    for route, status in active_routes.items():
        if status.active:
            active_routes[route].active = False
    return {"status": "All routes disabled"}


@app.get("/enable_all_routes")
async def enable_all_routes():
    for route, status in active_routes.items():
        active_routes[route].active = True
    return {"status": "All routes enabled"}


@app.post("/shutdown_server")
async def shutdown_server():
    # This endpoint should be secured
    global PID
    try:
        # sigterm
        os.kill(PID, signal.SIGTERM)
        sys.exit(0)
    except ProcessLookupError:
        return {"status": "Server is not running"}


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next: Callable):
    request_path = _parse_request_path(request)
    if request_path not in active_routes:
        return await call_next(request)
    else:
        global request_counts, first_requests, configs, route_limits

        route_limit = route_limits[request_path]

        path = request.url.path

        if path not in request_counts:
            request_counts[path] = 1
            first_requests[path] = datetime.now()
        else:
            request_counts[path] += 1

        time_passed = datetime.now() - first_requests[path]

        if time_passed <= TIME_WINDOW:
            if request_counts[path] > route_limit:
                raise HTTPException(status_code=429, detail="Too Many Requests")
        else:
            request_counts[path] = 1
            first_requests[path] = datetime.now()

        response = await call_next(request)
        return response


@app.middleware("http")
async def check_route(request: Request, call_next: Callable):
    request_path = _parse_request_path(request)
    if request_path in active_routes and not active_routes.get(request_path).active:
        return JSONResponse(
            status_code=403,
            content={"detail": "Route is disabled"},
        )
    return await call_next(request)


def _add_route(route_config: _Route):
    app.add_api_route(
        path=f"/{route_config.route}",
        endpoint=route_config.function,
        description=route_config.description,
        methods=["POST"],
    )
    route_config.active = True
    active_routes[route_config.route] = route_config


def _add_endpoints(route_configs):
    if isinstance(route_configs, _Route):
        route_configs = [route_configs]
    for route in route_configs:
        _add_route(route)


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    _add_endpoints(configs)

    uvicorn.run(app, host=args.host, port=args.port)
