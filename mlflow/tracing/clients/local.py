import threading
from collections import deque
from typing import List, Optional

from mlflow.entities import Trace
from mlflow.environment_variables import MLFLOW_TRACING_CLIENT_BUFFER_SIZE
from mlflow.tracing.clients.base import TraceClient
from mlflow.tracing.display import get_display_handler


class InMemoryTraceClient(TraceClient):
    """
    Simple trace client that stores traces in a in-memory buffer with a fixed size.
    This is intended to be used for development purpose only, to allow quicker
    inner-loop by getting traces without interacting with the real tracing backend.
    """

    _instance: TraceClient = None

    @classmethod
    def get_instance(cls) -> TraceClient:
        # NB: Only implement the minimal singleton functionality but not thread-safety.
        #     as this is intended to be used in a demonstration and testing purpose.
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        from mlflow.tracking.client import MlflowClient

        queue_size = MLFLOW_TRACING_CLIENT_BUFFER_SIZE.get()
        self.queue = deque(maxlen=queue_size)
        self._lock = threading.Lock()  # Lock for accessing the queue
        self._client = MlflowClient()

    def log_trace(self, trace: Trace):
        with self._lock:
            self.queue.append(trace)
        get_display_handler().display_traces([trace])

        started_trace = self._client._tracking_client.start_trace(
            experiment_id=trace.info.experiment_id,
            timestamp_ms=trace.info.timestamp_ms,
            request_metadata=trace.info.request_metadata,
            tags=trace.info.tags,
        )
        self._client._tracking_client._upload_trace_data(started_trace, trace.data)
        self._client._tracking_client.end_trace(
            request_id=started_trace.request_id,
            timestamp_ms=trace.info.timestamp_ms + trace.info.execution_time_ms,
            status=trace.info.status,
            request_metadata={},
            tags={},
        )

    def get_traces(self, n: Optional[int] = 10) -> List[Trace]:
        """
        Get the last n traces from the buffer.

        Args:
            n: The number of traces to return. If None, return all traces.

        Returns:
            A list of Trace objects.
        """
        with self._lock:
            trace_list = list(self.queue)
        return trace_list if n is None else trace_list[-n:]

    def get_trace(self, request_id: str) -> Optional[Trace]:
        """
        Get the trace with the given request_id.

        Args:
            request_id: The request_id of the trace to return.

        Returns:
            A Trace object.
        """
        with self._lock:
            for trace in self.queue:
                if trace.info.request_id == request_id:
                    return trace

    def _flush(self):
        self.queue.clear()
