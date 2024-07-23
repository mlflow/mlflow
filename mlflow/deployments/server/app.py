from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from mlflow.deployments.server.config import Endpoint
from mlflow.deployments.server.constants import (
    MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE,
    MLFLOW_DEPLOYMENTS_ENDPOINTS_BASE,
    MLFLOW_DEPLOYMENTS_HEALTH_ENDPOINT,
    MLFLOW_DEPLOYMENTS_LIMITS_BASE,
    MLFLOW_DEPLOYMENTS_LIST_ENDPOINTS_PAGE_SIZE,
    MLFLOW_DEPLOYMENTS_QUERY_SUFFIX,
)
from mlflow.environment_variables import (
    MLFLOW_DEPLOYMENTS_CONFIG,
    MLFLOW_GATEWAY_RATE_LIMITS_STORAGE_URI,
)
from mlflow.exceptions import MlflowException
from mlflow.gateway.base_models import SetLimitsModel
from mlflow.gateway.config import (
    GatewayConfig,
    LimitsConfig,
    Route,
    RouteConfig,
    RouteType,
    _load_route_config,
)
from mlflow.gateway.constants import (
    MLFLOW_GATEWAY_CRUD_ROUTE_BASE,
    MLFLOW_GATEWAY_HEALTH_ENDPOINT,
    MLFLOW_GATEWAY_LIMITS_BASE,
    MLFLOW_GATEWAY_ROUTE_BASE,
    MLFLOW_GATEWAY_SEARCH_ROUTES_PAGE_SIZE,
    MLFLOW_QUERY_SUFFIX,
)
from mlflow.gateway.providers import get_provider
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.gateway.utils import SearchRoutesToken, make_streaming_response
from mlflow.version import VERSION


class GatewayAPI(FastAPI):
    def __init__(self, config: GatewayConfig, limiter: Limiter, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.state.limiter = limiter
        self.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        self.dynamic_routes: Dict[str, RouteConfig] = {}
        self.set_dynamic_routes(config, limiter)

    def set_dynamic_routes(self, config: GatewayConfig, limiter: Limiter) -> None:
        self.dynamic_routes.clear()
        for route in config.routes:
            self.add_api_route(
                path=(
                    MLFLOW_DEPLOYMENTS_ENDPOINTS_BASE + route.name + MLFLOW_DEPLOYMENTS_QUERY_SUFFIX
                ),
                endpoint=_route_type_to_endpoint(route, limiter, "deployments"),
                methods=["POST"],
            )
            # TODO: Remove Gateway server URLs after deprecation window elapses
            self.add_api_route(
                path=f"{MLFLOW_GATEWAY_ROUTE_BASE}{route.name}{MLFLOW_QUERY_SUFFIX}",
                endpoint=_route_type_to_endpoint(route, limiter, "gateway"),
                methods=["POST"],
                include_in_schema=False,
            )
            self.dynamic_routes[route.name] = route

    def get_dynamic_route(self, route_name: str) -> Optional[Route]:
        return r.to_route() if (r := self.dynamic_routes.get(route_name)) else None


def _create_chat_endpoint(config: RouteConfig):
    prov = get_provider(config.model.provider)(config)

    # https://slowapi.readthedocs.io/en/latest/#limitations-and-known-issues
    async def _chat(
        request: Request, payload: chat.RequestPayload
    ) -> Union[chat.ResponsePayload, chat.StreamResponsePayload]:
        if payload.stream:
            return await make_streaming_response(prov.chat_stream(payload))
        else:
            return await prov.chat(payload)

    return _chat


def _create_completions_endpoint(config: RouteConfig):
    prov = get_provider(config.model.provider)(config)

    async def _completions(
        request: Request, payload: completions.RequestPayload
    ) -> Union[completions.ResponsePayload, completions.StreamResponsePayload]:
        if payload.stream:
            return await make_streaming_response(prov.completions_stream(payload))
        else:
            return await prov.completions(payload)

    return _completions


def _create_embeddings_endpoint(config: RouteConfig):
    prov = get_provider(config.model.provider)(config)

    async def _embeddings(
        request: Request, payload: embeddings.RequestPayload
    ) -> embeddings.ResponsePayload:
        return await prov.embeddings(payload)

    return _embeddings


async def _custom(request: Request):
    return request.json()


def _route_type_to_endpoint(config: RouteConfig, limiter: Limiter, key: str):
    provider_to_factory = {
        RouteType.LLM_V1_CHAT: _create_chat_endpoint,
        RouteType.LLM_V1_COMPLETIONS: _create_completions_endpoint,
        RouteType.LLM_V1_EMBEDDINGS: _create_embeddings_endpoint,
    }
    if factory := provider_to_factory.get(config.route_type):
        handler = factory(config)
        if limit := config.limit:
            limit_value = f"{limit.calls}/{limit.renewal_period}"
            handler.__name__ = f"{handler.__name__}_{config.name}_{key}"
            return limiter.limit(limit_value)(handler)
        else:
            return handler

    raise HTTPException(
        status_code=404,
        detail=f"Unexpected route type {config.route_type!r} for route {config.name!r}.",
    )


class HealthResponse(BaseModel):
    status: str


class ListEndpointsResponse(BaseModel):
    endpoints: List[Endpoint]
    next_page_token: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "endpoints": [
                    {
                        "name": "openai-chat",
                        "endpoint_type": "llm/v1/chat",
                        "model": {
                            "name": "gpt-3.5-turbo",
                            "provider": "openai",
                        },
                        "limit": {"calls": 1, "key": None, "renewal_period": "minute"},
                    },
                    {
                        "name": "anthropic-completions",
                        "endpoint_type": "llm/v1/completions",
                        "model": {
                            "name": "claude-instant-100k",
                            "provider": "anthropic",
                        },
                    },
                    {
                        "name": "cohere-embeddings",
                        "endpoint_type": "llm/v1/embeddings",
                        "model": {
                            "name": "embed-english-v2.0",
                            "provider": "cohere",
                        },
                    },
                ],
                "next_page_token": "eyJpbmRleCI6IDExfQ==",
            }
        }


class SearchRoutesResponse(BaseModel):
    routes: List[Route]
    next_page_token: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "routes": [
                    {
                        "name": "openai-chat",
                        "route_type": "llm/v1/chat",
                        "model": {
                            "name": "gpt-3.5-turbo",
                            "provider": "openai",
                        },
                    },
                    {
                        "name": "anthropic-completions",
                        "route_type": "llm/v1/completions",
                        "model": {
                            "name": "claude-instant-100k",
                            "provider": "anthropic",
                        },
                    },
                    {
                        "name": "cohere-embeddings",
                        "route_type": "llm/v1/embeddings",
                        "model": {
                            "name": "embed-english-v2.0",
                            "provider": "cohere",
                        },
                    },
                ],
                "next_page_token": "eyJpbmRleCI6IDExfQ==",
            }
        }


def create_app_from_config(config: GatewayConfig) -> GatewayAPI:
    """
    Create the GatewayAPI app from the gateway configuration.
    """
    limiter = Limiter(
        key_func=get_remote_address, storage_uri=MLFLOW_GATEWAY_RATE_LIMITS_STORAGE_URI.get()
    )
    app = GatewayAPI(
        config=config,
        limiter=limiter,
        title="MLflow Deployments Server",
        description="The core deployments API for reverse proxy interface using remote inference "
        "endpoints within MLflow",
        version=VERSION,
        docs_url=None,
    )

    @app.get("/", include_in_schema=False)
    async def index():
        return RedirectResponse(url="/docs")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        for directory in ["build", "public"]:
            favicon = Path(__file__).parent.parent.parent.joinpath(
                "server", "js", directory, "favicon.ico"
            )
            if favicon.exists():
                return FileResponse(favicon)
        raise HTTPException(status_code=404, detail="favicon.ico not found")

    @app.get("/docs", include_in_schema=False)
    async def docs():
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title="MLflow Deployments Server",
            swagger_favicon_url="/favicon.ico",
        )

    @app.get(MLFLOW_DEPLOYMENTS_HEALTH_ENDPOINT)
    # TODO: Remove Gateway server URLs after deprecation window elapses
    @app.get(MLFLOW_GATEWAY_HEALTH_ENDPOINT, include_in_schema=False)
    async def health() -> HealthResponse:
        return {"status": "OK"}

    @app.get(MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE + "{endpoint_name}")
    async def get_endpoint(endpoint_name: str) -> Endpoint:
        if matched := app.get_dynamic_route(endpoint_name):
            return matched.to_endpoint()

        raise HTTPException(
            status_code=404,
            detail=f"The endpoint '{endpoint_name}' is not present or active on the server. Please "
            "verify the endpoint name.",
        )

    # TODO: Remove Gateway server URLs after deprecation window elapses
    @app.get(MLFLOW_GATEWAY_CRUD_ROUTE_BASE + "{route_name}", include_in_schema=False)
    async def get_route(route_name: str) -> Route:
        if matched := app.get_dynamic_route(route_name):
            return matched

        raise HTTPException(
            status_code=404,
            detail=f"The route '{route_name}' is not present or active on the server. Please "
            "verify the route name.",
        )

    @app.get(MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE)
    async def list_endpoints(page_token: Optional[str] = None) -> ListEndpointsResponse:
        start_idx = SearchRoutesToken.decode(page_token).index if page_token is not None else 0

        end_idx = start_idx + MLFLOW_DEPLOYMENTS_LIST_ENDPOINTS_PAGE_SIZE
        routes = list(app.dynamic_routes.values())
        result = {
            "endpoints": [route.to_route().to_endpoint() for route in routes[start_idx:end_idx]]
        }
        if len(routes[end_idx:]) > 0:
            next_page_token = SearchRoutesToken(index=end_idx)
            result["next_page_token"] = next_page_token.encode()

        return result

    # TODO: Remove Gateway server URLs after deprecation window elapses
    @app.get(MLFLOW_GATEWAY_CRUD_ROUTE_BASE, include_in_schema=False)
    async def search_routes(page_token: Optional[str] = None) -> SearchRoutesResponse:
        start_idx = SearchRoutesToken.decode(page_token).index if page_token is not None else 0

        end_idx = start_idx + MLFLOW_GATEWAY_SEARCH_ROUTES_PAGE_SIZE
        routes = list(app.dynamic_routes.values())
        result = {"routes": [r.to_route() for r in routes[start_idx:end_idx]]}
        if len(routes[end_idx:]) > 0:
            next_page_token = SearchRoutesToken(index=end_idx)
            result["next_page_token"] = next_page_token.encode()

        return result

    @app.get(MLFLOW_DEPLOYMENTS_LIMITS_BASE + "{endpoint}")
    # TODO: Remove Gateway server URLs after deprecation window elapses
    @app.get(MLFLOW_GATEWAY_LIMITS_BASE + "{endpoint}", include_in_schema=False)
    async def get_limits(endpoint: str) -> LimitsConfig:
        raise HTTPException(status_code=501, detail="The get_limits API is not available yet.")

    @app.post(MLFLOW_DEPLOYMENTS_LIMITS_BASE)
    # TODO: Remove Gateway server URLs after deprecation window elapses
    @app.post(MLFLOW_GATEWAY_LIMITS_BASE, include_in_schema=False)
    async def set_limits(payload: SetLimitsModel) -> LimitsConfig:
        raise HTTPException(status_code=501, detail="The set_limits API is not available yet.")

    def _look_up_route(name: str) -> Optional[Route]:
        if r := app.dynamic_routes.get(name):
            return r

        raise HTTPException(
            status_code=400,
            detail=f"Route {name} not found in the configuration.",
        )

    @app.post("/v1/chat/completions")
    async def openai_chat_handler(
        request: Request, payload: chat.RequestPayload
    ) -> chat.ResponsePayload:
        route = _look_up_route(payload.model)
        if route.route_type != RouteType.LLM_V1_CHAT:
            raise HTTPException(
                status_code=400,
                detail=f"Endpoint {route.name!r} is not a chat endpoint.",
            )

        prov = get_provider(route.model.provider)(route)
        payload.model = None  # provider rejects a request with model field, must be set to None
        if payload.stream:
            return await make_streaming_response(prov.chat_stream(payload))
        else:
            return await prov.chat(payload)

    @app.post("/v1/completions")
    async def openai_completions_handler(
        request: Request, payload: completions.RequestPayload
    ) -> completions.ResponsePayload:
        route = _look_up_route(payload.model)
        if route.route_type != RouteType.LLM_V1_COMPLETIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Endpoint {route.name!r} is not a completions endpoint.",
            )

        prov = get_provider(route.model.provider)(route)
        payload.model = None  # provider rejects a request with model field, must be set to None
        if payload.stream:
            return await make_streaming_response(prov.completions_stream(payload))
        else:
            return await prov.completions(payload)

    @app.post("/v1/embeddings")
    async def openai_embeddings_handler(
        request: Request, payload: embeddings.RequestPayload
    ) -> embeddings.ResponsePayload:
        route = _look_up_route(payload.model)
        if route.route_type != RouteType.LLM_V1_EMBEDDINGS:
            raise HTTPException(
                status_code=400,
                detail=f"Endpoint {route.name!r} is not an embeddings endpoint.",
            )

        prov = get_provider(route.model.provider)(route)
        payload.model = None  # provider rejects a request with model field, must be set to None
        return await prov.embeddings(payload)

    return app


def create_app_from_path(config_path: Union[str, Path]) -> GatewayAPI:
    """
    Load the path and generate the GatewayAPI app instance.
    """
    config = _load_route_config(config_path)
    return create_app_from_config(config)


def create_app_from_env() -> GatewayAPI:
    """
    Load the path from the environment variable and generate the GatewayAPI app instance.
    """
    if config_path := MLFLOW_DEPLOYMENTS_CONFIG.get():
        return create_app_from_path(config_path)

    raise MlflowException(
        f"Environment variable {MLFLOW_DEPLOYMENTS_CONFIG!r} is not set. "
        "Please set it to the path of the gateway configuration file."
    )
