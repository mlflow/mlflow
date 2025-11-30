import functools
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict
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
    MLFLOW_GATEWAY_CONFIG,
    MLFLOW_GATEWAY_RATE_LIMITS_STORAGE_URI,
)
from mlflow.exceptions import MlflowException
from mlflow.gateway.base_models import SetLimitsModel
from mlflow.gateway.config import (
    EndpointConfig,
    EndpointType,
    GatewayConfig,
    LimitsConfig,
    Provider,
    TrafficRouteConfig,
    _LegacyRoute,
    _load_gateway_config,
)
from mlflow.gateway.constants import (
    MLFLOW_GATEWAY_CRUD_ENDPOINT_V3_BASE,
    MLFLOW_GATEWAY_CRUD_ROUTE_BASE,
    MLFLOW_GATEWAY_CRUD_ROUTE_V3_BASE,
    MLFLOW_GATEWAY_HEALTH_ENDPOINT,
    MLFLOW_GATEWAY_LIMITS_BASE,
    MLFLOW_GATEWAY_ROUTE_BASE,
    MLFLOW_GATEWAY_SEARCH_ROUTES_PAGE_SIZE,
    MLFLOW_QUERY_SUFFIX,
)
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers import get_provider
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.gateway.utils import SearchRoutesToken, make_streaming_response
from mlflow.version import VERSION


class GatewayAPI(FastAPI):
    def __init__(self, config: GatewayConfig, limiter: Limiter, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.state.limiter = limiter
        self.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        self.dynamic_endpoints: dict[str, EndpointConfig] = {
            endpoint.name: endpoint for endpoint in config.endpoints
        }
        self.traffic_routes: dict[str, TrafficRouteConfig] = {
            route.name: route for route in (config.routes or [])
        }

        # config API routes
        for name in self.dynamic_endpoints.keys() | self.traffic_routes.keys():
            # TODO: Remove deployments server URLs after deprecation window elapses
            self.add_api_route(
                path=(MLFLOW_DEPLOYMENTS_ENDPOINTS_BASE + name + MLFLOW_DEPLOYMENTS_QUERY_SUFFIX),
                endpoint=_get_endpoint_handler(self, name, limiter, "deployments"),
                methods=["POST"],
            )
            self.add_api_route(
                path=f"{MLFLOW_GATEWAY_ROUTE_BASE}{name}{MLFLOW_QUERY_SUFFIX}",
                endpoint=_get_endpoint_handler(self, name, limiter, "gateway"),
                methods=["POST"],
                include_in_schema=False,
            )

    def _get_provider_by_name(self, name: str) -> tuple[Provider, EndpointType]:
        """
        If the name is an endpoint name, return the endpoint's provider
        If the name is a traffic route name, return a `TrafficRouteProvider`
        """
        from mlflow.gateway.providers.base import TrafficRouteProvider

        if name in self.dynamic_endpoints:
            config = self.dynamic_endpoints[name]
            return get_provider(config.model.provider)(config), config.endpoint_type
        if name in self.traffic_routes:
            route_config = self.traffic_routes[name]
            endpoint_configs = [
                self.dynamic_endpoints[destination.name]
                for destination in route_config.destinations
            ]
            traffic_splits = [
                destination.traffic_percentage for destination in route_config.destinations
            ]
            return TrafficRouteProvider(
                endpoint_configs,
                traffic_splits,
                route_config.routing_strategy,
            ), route_config.task_type
        raise MlflowException.invalid_parameter_value(f"Invalid endpoint / route name: '{name}'")

    def get_dynamic_endpoint(self, endpoint_name: str) -> Endpoint | None:
        return r.to_endpoint() if (r := self.dynamic_endpoints.get(endpoint_name)) else None

    def _get_legacy_dynamic_route(self, route_name: str) -> _LegacyRoute | None:
        return r._to_legacy_route() if (r := self.dynamic_endpoints.get(route_name)) else None


def _translate_http_exception(func):
    """
    Decorator for translating MLflow exceptions to HTTP exceptions
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except AIGatewayException as e:
            raise HTTPException(status_code=e.status_code, detail=e.detail)

    return wrapper


def _create_chat_endpoint(prov: Provider):
    # https://slowapi.readthedocs.io/en/latest/#limitations-and-known-issues
    @_translate_http_exception
    async def _chat(
        request: Request, payload: chat.RequestPayload
    ) -> chat.ResponsePayload | chat.StreamResponsePayload:
        if payload.stream:
            return await make_streaming_response(prov.chat_stream(payload))
        else:
            return await prov.chat(payload)

    return _chat


def _create_completions_endpoint(prov: Provider):
    @_translate_http_exception
    async def _completions(
        request: Request, payload: completions.RequestPayload
    ) -> completions.ResponsePayload | completions.StreamResponsePayload:
        if payload.stream:
            return await make_streaming_response(prov.completions_stream(payload))
        else:
            return await prov.completions(payload)

    return _completions


def _create_embeddings_endpoint(prov: Provider):
    @_translate_http_exception
    async def _embeddings(
        request: Request, payload: embeddings.RequestPayload
    ) -> embeddings.ResponsePayload:
        return await prov.embeddings(payload)

    return _embeddings


async def _custom(request: Request):
    return request.json()


def _get_endpoint_handler(gateway_api: GatewayAPI, name: str, limiter: Limiter, key: str):
    endpoint_type_to_factory = {
        EndpointType.LLM_V1_CHAT: _create_chat_endpoint,
        EndpointType.LLM_V1_COMPLETIONS: _create_completions_endpoint,
        EndpointType.LLM_V1_EMBEDDINGS: _create_embeddings_endpoint,
    }
    provider, endpoint_type = gateway_api._get_provider_by_name(name)

    if factory := endpoint_type_to_factory.get(endpoint_type):
        handler = factory(provider)

        if name in gateway_api.dynamic_endpoints:
            limit = gateway_api.dynamic_endpoints[name].limit
        else:
            limit = None

        if limit:
            limit_value = f"{limit.calls}/{limit.renewal_period}"
            handler.__name__ = f"{handler.__name__}_{name}_{key}"
            return limiter.limit(limit_value)(handler)
        else:
            return handler

    raise HTTPException(
        status_code=404,
        detail=f"Unexpected route type {endpoint_type!r} for route {name!r}.",
    )


class HealthResponse(BaseModel):
    status: str


class ListEndpointsResponse(BaseModel):
    endpoints: list[Endpoint]
    next_page_token: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "endpoints": [
                    {
                        "name": "openai-chat",
                        "endpoint_type": "llm/v1/chat",
                        "model": {
                            "name": "gpt-4o-mini",
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
    )


class _LegacySearchRoutesResponse(BaseModel):
    routes: list[_LegacyRoute]
    next_page_token: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "endpoints": [
                    {
                        "name": "openai-chat",
                        "route_type": "llm/v1/chat",
                        "model": {
                            "name": "gpt-4o-mini",
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
    )


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
        title="MLflow AI Gateway",
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
            favicon_file = Path(__file__).parent.parent.joinpath(
                "server", "js", directory, "favicon.ico"
            )
            if favicon_file.exists():
                return FileResponse(favicon_file)
        raise HTTPException(status_code=404, detail="favicon.ico not found")

    @app.get("/docs", include_in_schema=False)
    async def docs():
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title="MLflow AI Gateway",
            swagger_favicon_url="/favicon.ico",
        )

    # TODO: Remove deployments server URLs after deprecation window elapses
    @app.get(MLFLOW_DEPLOYMENTS_HEALTH_ENDPOINT)
    @app.get(MLFLOW_GATEWAY_HEALTH_ENDPOINT, include_in_schema=False)
    async def health() -> HealthResponse:
        return {"status": "OK"}

    # TODO: Remove deployments server URLs after deprecation window elapses
    @app.get(MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE + "{endpoint_name}")
    async def get_endpoint(endpoint_name: str) -> Endpoint:
        if matched := app.get_dynamic_endpoint(endpoint_name):
            return matched

        raise HTTPException(
            status_code=404,
            detail=f"The endpoint '{endpoint_name}' is not present or active on the server. Please "
            "verify the endpoint name.",
        )

    # TODO: Remove the deprecated endpoint
    @app.get(
        MLFLOW_GATEWAY_CRUD_ROUTE_BASE + "{route_name}", include_in_schema=False, deprecated=True
    )
    async def _legacy_get_route(route_name: str) -> _LegacyRoute:
        if matched := app._get_legacy_dynamic_route(route_name):
            return matched

        raise HTTPException(
            status_code=404,
            detail=f"The route '{route_name}' is not present or active on the server. Please "
            "verify the route name.",
        )

    @app.get(MLFLOW_GATEWAY_CRUD_ENDPOINT_V3_BASE + "{endpoint_name}", include_in_schema=False)
    async def get_endpoint_v3(endpoint_name: str) -> Endpoint:
        if matched := app.dynamic_endpoints.get(endpoint_name):
            return matched.to_endpoint()

        raise HTTPException(
            status_code=404,
            detail=f"The endpoint '{endpoint_name}' is not present or active on the server. "
            f"Please verify the endpoint name.",
        )

    @app.get(MLFLOW_GATEWAY_CRUD_ROUTE_V3_BASE + "{route_name}", include_in_schema=False)
    async def get_route_v3(route_name: str) -> TrafficRouteConfig:
        if matched := app.traffic_routes.get(route_name):
            return matched

        raise HTTPException(
            status_code=404,
            detail=f"The route '{route_name}' is not present or active on the server. "
            f"Please verify the route name.",
        )

    # TODO: Remove deployments server URLs after deprecation window elapses
    @app.get(MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE)
    async def list_endpoints(page_token: str | None = None) -> ListEndpointsResponse:
        start_idx = SearchRoutesToken.decode(page_token).index if page_token is not None else 0

        end_idx = start_idx + MLFLOW_DEPLOYMENTS_LIST_ENDPOINTS_PAGE_SIZE
        endpoints = list(app.dynamic_endpoints.values())
        result = {
            "endpoints": [endpoint.to_endpoint() for endpoint in endpoints[start_idx:end_idx]]
        }
        if len(endpoints[end_idx:]) > 0:
            next_page_token = SearchRoutesToken(index=end_idx)
            result["next_page_token"] = next_page_token.encode()

        return result

    # TODO: Remove the deprecated endpoint
    @app.get(MLFLOW_GATEWAY_CRUD_ROUTE_BASE, include_in_schema=False, deprecated=True)
    async def _legacy_search_routes(page_token: str | None = None) -> _LegacySearchRoutesResponse:
        start_idx = SearchRoutesToken.decode(page_token).index if page_token is not None else 0

        end_idx = start_idx + MLFLOW_GATEWAY_SEARCH_ROUTES_PAGE_SIZE
        routes = list(app.dynamic_endpoints.values())
        result = {"routes": [r._to_legacy_route() for r in routes[start_idx:end_idx]]}
        if len(routes[end_idx:]) > 0:
            next_page_token = SearchRoutesToken(index=end_idx)
            result["next_page_token"] = next_page_token.encode()

        return result

    # TODO: Remove deployments server URLs after deprecation window elapses
    @app.get(MLFLOW_DEPLOYMENTS_LIMITS_BASE + "{endpoint}")
    @app.get(MLFLOW_GATEWAY_LIMITS_BASE + "{endpoint}", include_in_schema=False)
    async def get_limits(endpoint: str) -> LimitsConfig:
        raise HTTPException(status_code=501, detail="The get_limits API is not available yet.")

    # TODO: Remove deployments server URLs after deprecation window elapses
    @app.post(MLFLOW_DEPLOYMENTS_LIMITS_BASE)
    @app.post(MLFLOW_GATEWAY_LIMITS_BASE, include_in_schema=False)
    async def set_limits(payload: SetLimitsModel) -> LimitsConfig:
        raise HTTPException(status_code=501, detail="The set_limits API is not available yet.")

    @app.post("/v1/chat/completions")
    async def openai_chat_handler(
        request: Request, payload: chat.RequestPayload
    ) -> chat.ResponsePayload:
        name = payload.model
        prov, endpoint_type = app._get_provider_by_name(name)

        if endpoint_type != EndpointType.LLM_V1_CHAT:
            raise HTTPException(
                status_code=400,
                detail=f"Endpoint {name!r} is not a chat endpoint.",
            )

        payload.model = None  # provider rejects a request with model field, must be set to None
        if payload.stream:
            return await make_streaming_response(prov.chat_stream(payload))
        else:
            return await prov.chat(payload)

    @app.post("/v1/completions")
    async def openai_completions_handler(
        request: Request, payload: completions.RequestPayload
    ) -> completions.ResponsePayload:
        name = payload.model
        prov, endpoint_type = app._get_provider_by_name(name)

        if endpoint_type != EndpointType.LLM_V1_COMPLETIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Endpoint {name!r} is not a completions endpoint.",
            )

        payload.model = None  # provider rejects a request with model field, must be set to None
        if payload.stream:
            return await make_streaming_response(prov.completions_stream(payload))
        else:
            return await prov.completions(payload)

    @app.post("/v1/embeddings")
    async def openai_embeddings_handler(
        request: Request, payload: embeddings.RequestPayload
    ) -> embeddings.ResponsePayload:
        name = payload.model
        prov, endpoint_type = app._get_provider_by_name(name)

        if endpoint_type != EndpointType.LLM_V1_EMBEDDINGS:
            raise HTTPException(
                status_code=400,
                detail=f"Endpoint {name!r} is not an embeddings endpoint.",
            )

        payload.model = None  # provider rejects a request with model field, must be set to None
        return await prov.embeddings(payload)

    return app


def create_app_from_path(config_path: str | Path) -> GatewayAPI:
    """
    Load the path and generate the GatewayAPI app instance.
    """
    config = _load_gateway_config(config_path)
    return create_app_from_config(config)


def create_app_from_env() -> GatewayAPI:
    """
    Load the path from the environment variable and generate the GatewayAPI app instance.
    """
    if config_path := MLFLOW_GATEWAY_CONFIG.get():
        return create_app_from_path(config_path)

    raise MlflowException(
        f"Environment variable {MLFLOW_GATEWAY_CONFIG!r} is not set. "
        "Please set it to the path of the gateway configuration file."
    )
