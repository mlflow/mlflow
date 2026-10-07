"""Native FastAPI routes for the server-info capability endpoint."""

import anyio
from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from mlflow.exceptions import MlflowException
from mlflow.server.handlers import build_server_info_payload
from mlflow.server.mlflow_response import mlflow_exception_response

# Separate from AnyIO's default thread limiter, which the Flask WSGI mount uses.
# Overlapping Kubelet, UI, and Python client calls for this one endpoint.
# Per process, and only for server-info. /health does not use it.
_SERVER_INFO_LIMITER = anyio.CapacityLimiter(8)

server_info_router = APIRouter(tags=["Server Info"])


@server_info_router.get("/health")
async def health() -> PlainTextResponse:
    # Probes must not wait for a Flask worker. Gunicorn and Waitress keep the Flask route.
    return PlainTextResponse("OK")


@server_info_router.get("/api/3.0/mlflow/server-info")
@server_info_router.get("/ajax-api/3.0/mlflow/server-info")
async def get_server_info() -> Response:
    try:
        payload = await anyio.to_thread.run_sync(
            build_server_info_payload,
            limiter=_SERVER_INFO_LIMITER,
        )
    except MlflowException as exc:
        return mlflow_exception_response(exc, handler_name="get_server_info")
    return JSONResponse(content=payload)
