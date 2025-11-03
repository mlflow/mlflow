import argparse
import inspect
import json
import logging
from typing import Any, AsyncGenerator, Callable, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

import mlflow
from mlflow.genai.agent_server.utils import set_request_headers
from mlflow.genai.agent_server.validator import BaseAgentValidator, ResponsesAgentValidator
from mlflow.pyfunc import ResponsesAgent
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.utils.annotations import experimental

logger = logging.getLogger(__name__)
STREAM_KEY = "stream"

AgentType = Literal["ResponsesAgent"]


@experimental(version="3.6.0")
class AgentServer:
    """FastAPI-based server for hosting agents.

    Args:
        agent_type: An optional parameter to specify the type of agent to serve. If provided,
        input/output validation and streaming tracing aggregation will be done automatically.

        Currently only "ResponsesAgent" is supported.

        If ``None``, no input/output validation and streaming tracing aggregation will be done.
        Default to ``None``.
    """

    def __init__(self, agent_type: AgentType | None = None):
        self.agent_type = agent_type
        if agent_type == "ResponsesAgent":
            self.validator = ResponsesAgentValidator()
        else:
            self.validator = BaseAgentValidator()

        self.app = FastAPI(title="Agent Server")
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self.app.post("/invocations")
        async def invocations_endpoint(request: Request):
            # Capture headers such as x-forwarded-access-token
            # https://docs.databricks.com/aws/en/dev-tools/databricks-apps/auth?language=Streamlit#retrieve-user-authorization-credentials
            set_request_headers(dict(request.headers))

            try:
                data = await request.json()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON in request body: {e!s}")

            logger.debug(
                "Request received",
                extra={
                    "agent_type": self.agent_type,
                    "request_size": len(json.dumps(data)),
                    "stream_requested": data.get(STREAM_KEY, False),
                },
            )

            is_streaming = data.pop(STREAM_KEY, False)

            try:
                request_data = self.validator.validate_and_convert_request(request)
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid parameters for {self.agent_type}: {e}",
                )

            if is_streaming:
                return await self._handle_stream_request(request_data)
            else:
                return await self._handle_invoke_request(request_data)

        @self.app.get("/health")
        async def health_check() -> dict[str, str]:
            return {"status": "healthy"}

    async def _handle_invoke_request(self, request: dict[str, Any]) -> dict[str, Any]:
        from mlflow.genai.agent_server import _invoke_function

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

    async def generate(
        self,
        func: Callable[..., Any],
        request: dict[str, Any],
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

    async def _handle_stream_request(self, request: dict[str, Any]) -> StreamingResponse:
        from mlflow.genai.agent_server import _stream_function

        if _stream_function is None:
            raise HTTPException(status_code=500, detail="No stream function registered")

        func = _stream_function

        return StreamingResponse(self.generate(func, request), media_type="text/event-stream")

    @staticmethod
    def parse_server_args():
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
        args = self.parse_server_args()
        uvicorn.run(
            app_import_string, host=host, port=args.port, workers=args.workers, reload=args.reload
        )
