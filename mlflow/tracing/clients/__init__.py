from mlflow.tracing.clients.base import TraceClient
from mlflow.tracing.clients.local import InMemoryTraceClient, InMemoryTraceClientWithTracking

__all__ = [
    "InMemoryTraceClient",
    "InMemoryTraceClientWithTracking",
    "TraceClient",
    "get_trace_client",
]


def get_trace_client() -> TraceClient:
    return InMemoryTraceClientWithTracking.get_instance()
