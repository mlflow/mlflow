import inspect
import json
import logging
from typing import Any, AsyncGenerator, Callable, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

import mlflow
from mlflow.genai.agent_server.utils import set_request_headers
from mlflow.genai.agent_server.validator import (
    BaseAgentValidator,
    ChatAgentValidator,
    ChatCompletionValidator,
    ResponsesAgentValidator,
)
from mlflow.pyfunc import ResponsesAgent
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.types.llm import (
    ChatCompletionChunk,
)
from mlflow.utils.annotations import experimental

logger = logging.getLogger(__name__)
RETURN_TRACE_KEY = "return_trace"
STREAM_KEY = "stream"

AgentType = Literal["ResponsesAgent", "ChatCompletion", "ChatAgent"]


@experimental(version="3.6.0")
class AgentServer:
    """FastAPI-based server for hosting agents.

    Args:
        agent_type: An optional parameter to specify the type of agent to serve. If provided,
        input/output validation and streaming tracing aggregation will be done automatically.
        The agent type must be one of the following:
        - "ResponsesAgent"
        - "ChatCompletion"
        - "ChatAgent"

        If ``None``, no input/output validation and streaming tracing aggregation will be done.
        Default to ``None``.
    """

    def __init__(self, agent_type: AgentType | None = None):
        self.agent_type = agent_type
        if agent_type == "ResponsesAgent":
            self.validator = ResponsesAgentValidator()
        elif agent_type == "ChatCompletion":
            self.validator = ChatCompletionValidator()
        elif agent_type == "ChatAgent":
            self.validator = ChatAgentValidator()
        else:
            self.validator = BaseAgentValidator()

        self.app = FastAPI(title="Agent Server")
        self._setup_routes()

    @staticmethod
    def _get_in_memory_trace(trace_id: str) -> dict[str, Any]:
        with InMemoryTraceManager.get_instance().get_trace(trace_id) as trace:
            return {"trace": trace.to_mlflow_trace().to_dict()}

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

            is_streaming = data.get(STREAM_KEY, False)
            return_trace = data.get(RETURN_TRACE_KEY, False)

            request_data = {k: v for k, v in data.items() if k != STREAM_KEY}

            try:
                request_data = self.validator.validate_and_convert_request(request_data)
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid parameters for {self.agent_type}: {e}",
                )

            if is_streaming:
                return await self._handle_stream_request(request_data, return_trace)
            else:
                return await self._handle_invoke_request(request_data, return_trace)

        @self.app.get("/health")
        async def health_check() -> dict[str, str]:
            return {"status": "healthy"}

    async def _handle_invoke_request(
        self, request: dict[str, Any], return_trace: bool
    ) -> dict[str, Any]:
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
                    span.set_attribute("mlflow.message.format", "openai")
                span.set_outputs(result)

                if return_trace:
                    result |= self._get_in_memory_trace(span.trace_id)

            logger.debug(
                "Response sent",
                extra={
                    "endpoint": "invoke",
                    "response_size": len(json.dumps(result)),
                    "function_name": func_name,
                    "return_trace": return_trace,
                },
            )

            return result

        except Exception as e:
            logger.error(
                "Error response sent",
                extra={
                    "endpoint": "invoke",
                    "error": str(e),
                    "function_name": func_name,
                    "return_trace": return_trace,
                },
            )

            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    def _extract_content(chunk: ChatCompletionChunk | dict[str, Any]) -> str:
        if isinstance(chunk, dict):
            choices = chunk.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("delta", {}).get("content", "")
        if not chunk.choices:
            return ""
        return chunk.choices[0].delta.content or ""

    async def generate(
        self,
        func: Callable[..., Any],
        request: dict[str, Any],
        return_trace: bool,
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
                    span.set_attribute("mlflow.message.format", "openai")
                    span.set_outputs(ResponsesAgent.responses_agent_output_reducer(all_chunks))
                elif self.agent_type == "ChatCompletion":
                    content = "".join(map(self._extract_content, all_chunks))
                    span.set_outputs({"choices": [{"role": "assistant", "content": content}]})
                elif self.agent_type == "ChatAgent":
                    span.set_outputs({"messages": [chunk["delta"] for chunk in all_chunks]})
                else:
                    span.set_outputs(all_chunks)

                if return_trace:
                    trace = self._get_in_memory_trace(span.trace_id)
                    yield f"data: {json.dumps(trace)}\n\n"

                yield "data: [DONE]\n\n"

            logger.info(
                "Streaming response completed",
                extra={
                    "endpoint": "stream",
                    "total_chunks": len(all_chunks),
                    "function_name": func_name,
                    "return_trace": return_trace,
                },
            )

        except Exception as e:
            logger.error(
                "Streaming response error",
                extra={
                    "endpoint": "stream",
                    "error": str(e),
                    "function_name": func_name,
                    "chunks_sent": len(all_chunks),
                    "return_trace": return_trace,
                },
            )

            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    async def _handle_stream_request(
        self, request: dict[str, Any], return_trace: bool
    ) -> StreamingResponse:
        from mlflow.genai.agent_server import _stream_function

        if _stream_function is None:
            raise HTTPException(status_code=500, detail="No stream function registered")

        func = _stream_function

        return StreamingResponse(
            self.generate(func, request, return_trace), media_type="text/event-stream"
        )

    def run(
        self,
        app_import_string: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
        reload: bool = False,
    ) -> None:
        uvicorn.run(app_import_string, host=host, port=port, workers=workers, reload=reload)
