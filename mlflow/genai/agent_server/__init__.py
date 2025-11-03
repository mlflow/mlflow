"""
FastAPI-based server for hosting agents that either use the Responses API schema or a custom schema.

Key Features:
- Decorator-based function registration (@invoke, @stream) for easy agent development
- Automatic request and response validation for Responses API schema agents
- Context-aware request header management for Databricks Apps authentication
- MLflow tracing integration
- Command-line argument parsing for server configuration (e.g. --port, --workers, --reload)

Usage:
    from mlflow.genai.agent_server import AgentServer, invoke, stream

    @invoke()
    def my_agent_invoke(request):
        return {
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello"}]
            }]
        }

    @stream()
    async def my_agent_stream(request):
        yield {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello"}]
            }
        }

    server = AgentServer(agent_type="ResponsesAgent")
    server.run("my_app:server.app")
"""

from mlflow.genai.agent_server.server import (
    AgentServer,
    get_invoke_function,
    get_stream_function,
    invoke,
    stream,
)
from mlflow.genai.agent_server.utils import (
    get_request_headers,
    set_request_headers,
    setup_mlflow_git_based_version_tracking,
)

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
