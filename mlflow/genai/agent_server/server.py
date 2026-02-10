import argparse
import functools
import inspect
import json
import logging
import os
import posixpath
from typing import Any, AsyncGenerator, Callable, Literal, ParamSpec, TypeVar

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

import mlflow
from mlflow.genai.agent_server.utils import get_request_headers, set_request_headers
from mlflow.genai.agent_server.validator import BaseAgentValidator, ResponsesAgentValidator
from mlflow.pyfunc import ResponsesAgent
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.utils.annotations import experimental

logger = logging.getLogger(__name__)
STREAM_KEY = "stream"
RETURN_TRACE_HEADER = "x-mlflow-return-trace-id"

AgentType = Literal["ResponsesAgent"]

_P = ParamSpec("_P")
_R = TypeVar("_R")

_invoke_function: Callable[..., Any] | None = None
_stream_function: Callable[..., Any] | None = None


@experimental(version="3.6.0")
def get_invoke_function():
    return _invoke_function


@experimental(version="3.6.0")
def get_stream_function():
    return _stream_function


@experimental(version="3.6.0")
def invoke() -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorator to register a function as an invoke endpoint. Can only be used once."""

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        global _invoke_function
        if _invoke_function is not None:
            raise ValueError("invoke decorator can only be used once")
        _invoke_function = func

        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return func(*args, **kwargs)

        return wrapper

    return decorator


@experimental(version="3.6.0")
def stream() -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorator to register a function as a stream endpoint. Can only be used once."""

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        global _stream_function
        if _stream_function is not None:
            raise ValueError("stream decorator can only be used once")
        _stream_function = func

        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return func(*args, **kwargs)

        return wrapper

    return decorator


@experimental(version="3.6.0")
class AgentServer:
    """FastAPI-based server for hosting agents.

    Args:
        agent_type: An optional parameter to specify the type of agent to serve. If provided,
        input/output validation and streaming tracing aggregation will be done automatically.

        Currently only "ResponsesAgent" is supported.

        If ``None``, no input/output validation and streaming tracing aggregation will be done.
        Default to ``None``.

        enable_chat_proxy: If ``True``, enables a proxy middleware that forwards unmatched requests
        to a chat app running on the port specified by the CHAT_APP_PORT environment variable
        (defaults to 3000) with a timeout specified by the CHAT_PROXY_TIMEOUT_SECONDS environment
        variable, (defaults to 300 seconds). ``enable_chat_proxy`` defaults to ``False``.

        The proxy allows requests to ``/``, ``/favicon.ico``, ``/assets/*``, and ``/api/*`` by
        default. Additional paths can be configured via environment variables:

        - ``CHAT_PROXY_ALLOWED_EXACT_PATHS``: Comma-separated list of additional exact paths
          to allow (e.g., ``/custom,/another``).
        - ``CHAT_PROXY_ALLOWED_PATH_PREFIXES``: Comma-separated list of additional path prefixes
          to allow (e.g., ``/custom/,/another/``).

    See https://mlflow.org/docs/latest/genai/serving/agent-server for more information.
    """

    def __init__(self, agent_type: AgentType | None = None, enable_chat_proxy: bool = False):
        self.agent_type = agent_type
        if agent_type == "ResponsesAgent":
            self.validator = ResponsesAgentValidator()
        else:
            self.validator = BaseAgentValidator()

        self.app = FastAPI(title="Agent Server")

        if enable_chat_proxy:
            self._setup_chat_proxy_middleware()

        self._setup_routes()

    def _setup_chat_proxy_middleware(self) -> None:
        """Set up middleware to proxy static asset requests to the chat app.

        Only forwards requests to allowed paths (/, /assets/*, /favicon.ico) to prevent
        SSRF vulnerabilities.
        """
        self.chat_app_port = os.getenv("CHAT_APP_PORT", "3000")
        self.chat_proxy_timeout = float(os.getenv("CHAT_PROXY_TIMEOUT_SECONDS", "300.0"))
        self.proxy_client = httpx.AsyncClient(timeout=self.chat_proxy_timeout)

        # Only proxy static assets to prevent SSRF vulnerabilities
        allowed_exact_paths = {"/", "/favicon.ico"}
        allowed_path_prefixes = {"/assets/", "/api/"}

        # Add additional paths from environment variables
        if additional_exact_paths := os.getenv("CHAT_PROXY_ALLOWED_EXACT_PATHS", ""):
            allowed_exact_paths |= {
                p.strip() for p in additional_exact_paths.split(",") if p.strip()
            }

        if additional_path_prefixes := os.getenv("CHAT_PROXY_ALLOWED_PATH_PREFIXES", ""):
            allowed_path_prefixes |= {
                p.strip() for p in additional_path_prefixes.split(",") if p.strip()
            }

        @self.app.middleware("http")
        async def chat_proxy_middleware(request: Request, call_next):
            """
            Forward static asset requests to the chat app on the port specified by the
            CHAT_APP_PORT environment variable (defaults to 3000).

            Only forwards requests to:
            - / (base path for index.html)
            - /assets/* (Vite static assets)
            - /favicon.ico

            The timeout for the proxy request is specified by the CHAT_PROXY_TIMEOUT_SECONDS
            environment variable (defaults to 300.0 seconds).

            For streaming responses (SSE), the proxy streams chunks as they arrive
            rather than buffering the entire response.
            """
            for route in self.app.routes:
                if hasattr(route, "path_regex") and route.path_regex.match(request.url.path):
                    return await call_next(request)

            # Normalize path to prevent traversal attacks (e.g., /assets/../.env)
            path = posixpath.normpath(request.url.path)

            # Only allow proxying static assets
            is_allowed = path in allowed_exact_paths or any(
                path.startswith(p) for p in allowed_path_prefixes
            )
            if not is_allowed:
                return Response("Not found", status_code=404, media_type="text/plain")

            path = path.lstrip("/")
            try:
                body = await request.body() if request.method in ["POST", "PUT", "PATCH"] else None
                target_url = f"http://localhost:{self.chat_app_port}/{path}"

                # Build and send request with streaming enabled
                req = self.proxy_client.build_request(
                    method=request.method,
                    url=target_url,
                    params=dict(request.query_params),
                    headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
                    content=body,
                )
                proxy_response = await self.proxy_client.send(req, stream=True)

                # Check if this is a streaming response (SSE)
                content_type = proxy_response.headers.get("content-type", "")
                if "text/event-stream" in content_type:
                    # Stream SSE responses chunk by chunk
                    async def stream_generator():
                        try:
                            async for chunk in proxy_response.aiter_bytes():
                                yield chunk
                        except Exception as e:
                            logger.error(f"Streaming error: {e}")
                            raise
                        finally:
                            await proxy_response.aclose()

                    return StreamingResponse(
                        stream_generator(),
                        status_code=proxy_response.status_code,
                        headers=dict(proxy_response.headers),
                    )
                else:
                    # Non-streaming response - read fully then close
                    content = await proxy_response.aread()
                    await proxy_response.aclose()
                    return Response(
                        content,
                        proxy_response.status_code,
                        headers=dict(proxy_response.headers),
                    )
            except httpx.ConnectError:
                return Response("Service unavailable", status_code=503, media_type="text/plain")
            except Exception as e:
                return Response(f"Proxy error: {e!s}", status_code=502, media_type="text/plain")

    def _setup_routes(self) -> None:
        @self.app.post("/invocations")
        async def invocations_endpoint(request: Request):
            return await self._handle_invocations_request(request)

        # Only expose /responses endpoint for ResponsesAgent
        if self.agent_type == "ResponsesAgent":

            @self.app.post("/responses")
            async def responses_endpoint(request: Request):
                """
                For compatibility with the OpenAI Client `client.responses.create(...)` method.
                https://platform.openai.com/docs/api-reference/responses/create
                """
                return await self._handle_invocations_request(request)

        @self.app.get("/agent/info")
        async def agent_info_endpoint() -> dict[str, Any]:
            # Get app name from environment or use default
            app_name = os.getenv("DATABRICKS_APP_NAME", "mlflow_agent_server")

            # Base info payload
            info = {
                "name": app_name,
                "use_case": "agent",
                "mlflow_version": mlflow.__version__,
            }

            # Conditionally add agent_api field for ResponsesAgent only
            if self.agent_type == "ResponsesAgent":
                info["agent_api"] = "responses"

            return info

        @self.app.get("/health")
        async def health_check() -> dict[str, str]:
            return {"status": "healthy"}

    async def _handle_invocations_request(
        self, request: Request
    ) -> dict[str, Any] | StreamingResponse:
        # Capture headers such as x-forwarded-access-token
        # https://docs.databricks.com/aws/en/dev-tools/databricks-apps/auth?language=Streamlit#retrieve-user-authorization-credentials
        set_request_headers(dict(request.headers))

        try:
            data = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in request body: {e!s}")

        # Use actual request path for logging differentiation
        endpoint_path = request.url.path
        logger.debug(
            f"Request received at {endpoint_path}",
            extra={
                "agent_type": self.agent_type,
                "request_size": len(json.dumps(data)),
                "stream_requested": data.get(STREAM_KEY, False),
                "endpoint": endpoint_path,
            },
        )

        is_streaming = data.pop(STREAM_KEY, False)
        return_trace_id = (get_request_headers().get(RETURN_TRACE_HEADER) or "").lower() == "true"

        try:
            request_data = self.validator.validate_and_convert_request(data)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid parameters for {self.agent_type}: {e}",
            )

        if is_streaming:
            return await self._handle_stream_request(request_data, return_trace_id)
        else:
            return await self._handle_invoke_request(request_data, return_trace_id)

    async def _handle_invoke_request(
        self, request: dict[str, Any], return_trace_id: bool
    ) -> dict[str, Any]:
        if _invoke_function is None:
            raise HTTPException(status_code=500, detail="No invoke function registered")

        func = _invoke_function
        func_name = func.__name__

        try:
            with mlflow.start_span(name=f"{func_name}") as span:
                span.set_inputs(request)
                if inspect.iscoroutinefunction(func):
                    result = await func(request)
                else:
                    result = func(request)

                result = self.validator.validate_and_convert_result(result)
                if self.agent_type == "ResponsesAgent":
                    span.set_attribute(SpanAttributeKey.MESSAGE_FORMAT, "openai")
                    if return_trace_id:
                        result["metadata"] = (result.get("metadata") or {}) | {
                            "trace_id": span.trace_id
                        }
                span.set_outputs(result)

            logger.debug(
                "Response sent",
                extra={
                    "endpoint": "invoke",
                    "response_size": len(json.dumps(result)),
                    "function_name": func_name,
                },
            )

            return result

        except Exception as e:
            logger.debug(
                "Error response sent",
                extra={
                    "endpoint": "invoke",
                    "error": str(e),
                    "function_name": func_name,
                },
            )

            raise HTTPException(status_code=500, detail=str(e))

    async def _generate(
        self,
        func: Callable[..., Any],
        request: dict[str, Any],
        return_trace_id: bool,
    ) -> AsyncGenerator[str, None]:
        func_name = func.__name__
        all_chunks: list[dict[str, Any]] = []
        try:
            with mlflow.start_span(name=f"{func_name}") as span:
                span.set_inputs(request)
                if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
                    async for chunk in func(request):
                        chunk = self.validator.validate_and_convert_result(chunk, stream=True)
                        all_chunks.append(chunk)
                        yield f"data: {json.dumps(chunk)}\n\n"
                else:
                    for chunk in func(request):
                        chunk = self.validator.validate_and_convert_result(chunk, stream=True)
                        all_chunks.append(chunk)
                        yield f"data: {json.dumps(chunk)}\n\n"

                if self.agent_type == "ResponsesAgent":
                    span.set_attribute(SpanAttributeKey.MESSAGE_FORMAT, "openai")
                    span.set_outputs(ResponsesAgent.responses_agent_output_reducer(all_chunks))
                    if return_trace_id:
                        yield f"data: {json.dumps({'trace_id': span.trace_id})}\n\n"
                else:
                    span.set_outputs(all_chunks)

                yield "data: [DONE]\n\n"

            logger.debug(
                "Streaming response completed",
                extra={
                    "endpoint": "stream",
                    "total_chunks": len(all_chunks),
                    "function_name": func_name,
                },
            )

        except Exception as e:
            logger.debug(
                "Streaming response error",
                extra={
                    "endpoint": "stream",
                    "error": str(e),
                    "function_name": func_name,
                    "chunks_sent": len(all_chunks),
                },
            )

            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    async def _handle_stream_request(
        self, request: dict[str, Any], return_trace_id: bool
    ) -> StreamingResponse:
        if _stream_function is None:
            raise HTTPException(status_code=500, detail="No stream function registered")
        return StreamingResponse(
            self._generate(_stream_function, request, return_trace_id),
            media_type="text/event-stream",
        )

    @staticmethod
    def _parse_server_args():
        """Parse command line arguments for the agent server"""
        parser = argparse.ArgumentParser(description="Start the agent server")
        parser.add_argument(
            "--port", type=int, default=8000, help="Port to run the server on (default: 8000)"
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=1,
            help="Number of workers to run the server on (default: 1)",
        )
        parser.add_argument(
            "--reload",
            action="store_true",
            help="Reload the server on code changes (default: False)",
        )
        return parser.parse_args()

    def run(
        self,
        app_import_string: str,
        host: str = "0.0.0.0",
    ) -> None:
        """Run the agent server with command line argument parsing."""
        args = self._parse_server_args()
        uvicorn.run(
            app_import_string, host=host, port=args.port, workers=args.workers, reload=args.reload
        )
