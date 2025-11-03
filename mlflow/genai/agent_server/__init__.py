"""
FastAPI-based server for hosting MLflow agents with multiple protocol support.

This module provides a production-ready agent server that supports multiple agent types:
- ResponsesAgent (agent/v1/responses): OpenAI-compatible responses format
- ChatCompletion (agent/v1/chat): OpenAI chat completion format
- ChatAgent (agent/v2/chat): MLflow's enhanced chat agent format

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

    server = AgentServer(agent_type="agent/v1/responses")
    server.run("my_app:server.app")
"""

import argparse
from typing import Any, Callable

from mlflow.genai.agent_server.server import AgentServer
from mlflow.genai.agent_server.utils import (
    get_request_headers,
    get_user_workspace_client,
    set_request_headers,
    setup_mlflow_git_based_version_tracking,
)
from mlflow.utils.annotations import experimental

__all__ = [
    "set_request_headers",
    "get_request_headers",
    "get_user_workspace_client",
    "AgentServer",
    "invoke",
    "stream",
    "parse_server_args",
    "setup_mlflow_git_based_version_tracking",
]


_invoke_function: Callable[..., Any] | None = None
_stream_function: Callable[..., Any] | None = None


@experimental(version="3.6.0")
def invoke() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a function as an invoke endpoint. Can only be used once."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        global _invoke_function
        if _invoke_function is not None:
            raise ValueError("invoke decorator can only be used once")
        _invoke_function = func
        return func

    return decorator


def get_invoke_function():
    return _invoke_function


@experimental(version="3.6.0")
def stream() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a function as a stream endpoint. Can only be used once."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        global _stream_function
        if _stream_function is not None:
            raise ValueError("stream decorator can only be used once")
        _stream_function = func
        return func

    return decorator


@experimental(version="3.6.0")
def parse_server_args():
    """Parse command line arguments for the agent server"""
    parser = argparse.ArgumentParser(description="Start the agent server")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on (default: 8000)"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers to run the server on (default: 1)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload the server on code changes (default: False)",
    )
    return parser.parse_args()
