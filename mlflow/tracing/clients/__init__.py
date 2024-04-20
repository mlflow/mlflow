from mlflow.tracing.clients.base import TraceClient
from mlflow.tracing.clients.local import InMemoryTraceClient
from mlflow.tracing.clients.tracking import InMemoryTraceClientWithTracking

__all__ = [
    "InMemoryTraceClient",
    "InMemoryTraceClientWithTracking",
    "TraceClient",
    "get_trace_client",
]


def get_trace_client() -> TraceClient:
    return InMemoryTraceClient.get_instance()
