from mlflow.tracing.clients.base import TraceClient
from mlflow.tracing.clients.local import InMemoryTraceClient
from mlflow.tracing.clients.tracking import InMemoryTraceClientWithTracking
from mlflow.utils.uri import is_databricks_uri

__all__ = [
    "InMemoryTraceClient",
    "InMemoryTraceClientWithTracking",
    "TraceClient",
    "get_trace_client",
]


def get_trace_client() -> TraceClient:
    """Get the trace client instance based on the environment."""
    from mlflow.tracking._tracking_service.utils import get_tracking_uri

    if (uri := get_tracking_uri()) and is_databricks_uri(uri):
        return InMemoryTraceClientWithTracking.get_instance()
    else:
        return InMemoryTraceClient.get_instance()
