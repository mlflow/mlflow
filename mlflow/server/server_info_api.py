"""Native FastAPI routes for the server-info capability endpoint."""

import json
import logging

import anyio
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from mlflow.exceptions import MlflowException
from mlflow.server.handlers import build_server_info_payload

_logger = logging.getLogger(__name__)

# Separate from AnyIO's default thread limiter, which the Flask WSGI mount uses.
_SERVER_INFO_LIMITER = anyio.CapacityLimiter(2)

server_info_router = APIRouter(tags=["Server Info"])


def _mlflow_exception_response(exc: MlflowException) -> JSONResponse:
    status_code = exc.get_http_status_code()
    if status_code >= 500:
        is_debug = _logger.isEnabledFor(logging.DEBUG)
        msg = f"Error in get_server_info: {exc}"
        if not is_debug:
            msg += ". Set MLFLOW_LOGGING_LEVEL=DEBUG for traceback."
        _logger.error(msg, exc_info=is_debug)
    return JSONResponse(
        status_code=status_code,
        content=json.loads(exc.serialize_as_json()),
    )


@server_info_router.get("/api/3.0/mlflow/server-info")
@server_info_router.get("/ajax-api/3.0/mlflow/server-info")
async def get_server_info() -> JSONResponse:
    try:
        payload = await anyio.to_thread.run_sync(
            build_server_info_payload,
            limiter=_SERVER_INFO_LIMITER,
        )
    except MlflowException as exc:
        return _mlflow_exception_response(exc)
    return JSONResponse(content=payload)
