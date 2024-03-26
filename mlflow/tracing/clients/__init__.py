from mlflow.tracing.clients.base import TraceClient
from mlflow.tracing.clients.local import InMemoryTraceClient

__all__ = ["InMemoryTraceClient", "TraceClient", "get_trace_client"]


def get_trace_client() -> TraceClient:
    return InMemoryTraceClient.get_instance()
