from mlflow.entities import Trace
from mlflow.tracing.clients.local import InMemoryTraceClient


class InMemoryTraceClientWithTracking(InMemoryTraceClient):
    """
    `InMemoryTraceClient` with tracking capabilities.
    """

    def __init__(self):
        from mlflow.tracking.client import MlflowClient  # avoid circular import

        super().__init__()
        self._client = MlflowClient()

    def log_trace(self, trace: Trace):
        super().log_trace(trace)
        created_info = self._client._create_trace_info(
            experiment_id=trace.info.experiment_id,
            timestamp_ms=trace.info.timestamp_ms,
            execution_time_ms=trace.info.execution_time_ms,
            status=trace.info.status,
            request_metadata=trace.info.request_metadata,
            tags=trace.info.tags,
        )
        self._client._upload_trace_data(created_info, trace.data)
