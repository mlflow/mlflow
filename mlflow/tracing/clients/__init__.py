from mlflow.tracing.clients.base import TraceClient
from mlflow.tracing.clients.local import InMemoryTraceClient
from mlflow.tracing.clients.tracking import InMemoryTraceClientWithTracking
from mlflow.utils.databricks_utils import is_in_databricks_runtime

__all__ = [
    "InMemoryTraceClient",
    "InMemoryTraceClientWithTracking",
    "TraceClient",
    "get_trace_client",
]


def get_trace_client() -> TraceClient:
    if is_in_databricks_runtime():
        return InMemoryTraceClientWithTracking.get_instance()
    else:
        return InMemoryTraceClient.get_instance()
