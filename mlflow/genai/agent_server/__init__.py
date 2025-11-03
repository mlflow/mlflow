"""
FastAPI-based server for hosting MLflow agents with multiple protocol support.

This module provides a production-ready agent server that supports multiple agent types:
- ResponsesAgent (agent/v1/responses): OpenAI-compatible responses format

Key Features:
- Decorator-based function registration (@invoke, @stream) for easy agent development
- Protocol-specific request/response validation using AgentValidator
- Context-aware request header management for Databricks Apps authentication
- Streaming and non-streaming response support with Server-Sent Events (SSE)
- MLflow tracing integration with automatic span creation and attribute setting
- Health check endpoint for monitoring

Architecture:
- AgentServer: Main FastAPI application with route setup and middleware
- AgentValidator: Protocol-specific validation for requests and responses
- Context isolation: Thread-safe request header management using contextvars
- Function registration: Global decorators for invoke/stream endpoint functions

Usage:
    from mlflow.genai.agent_server import AgentServer, invoke, stream

    @invoke()
    def my_agent_invoke(request):
        return {"response": "Hello"}

    @stream()
    async def my_agent_stream(request):
        yield {"delta": {"content": "Hello"}}

    server = AgentServer(agent_type="ResponsesAgent")
    server.run("my_app:server.app")
"""

import functools
from typing import Any, Callable, ParamSpec, TypeVar

from mlflow.genai.agent_server.server import AgentServer
from mlflow.genai.agent_server.utils import (
    get_request_headers,
    set_request_headers,
    setup_mlflow_git_based_version_tracking,
)
from mlflow.utils.annotations import experimental

__all__ = [
    "set_request_headers",
    "get_request_headers",
    "AgentServer",
    "invoke",
    "stream",
    "get_invoke_function",
    "get_stream_function",
    "setup_mlflow_git_based_version_tracking",
]


_P = ParamSpec("_P")
_R = TypeVar("_R")

_invoke_function: Callable[..., Any] | None = None
_stream_function: Callable[..., Any] | None = None


def get_invoke_function():
    return _invoke_function


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
