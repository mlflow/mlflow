from fastapi import FastAPI

from mlflow.server.fastapi_app import add_mcp_exception_handlers
from mlflow.server.mcp_server_api import _MCP_SERVER_API_PREFIX, mcp_server_router


def create_registry_fastapi_app(route_prefix: str = _MCP_SERVER_API_PREFIX) -> FastAPI:
    fastapi_app = FastAPI()
    add_mcp_exception_handlers(fastapi_app)
    fastapi_app.include_router(mcp_server_router, prefix=route_prefix)
    return fastapi_app
