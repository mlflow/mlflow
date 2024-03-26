from collections import deque
from trace import Trace
from typing import List

from mlflow.environment_variables import MLFLOW_TRACING_CLIENT_BUFFER_SIZE
from mlflow.tracing.clients.base import TraceClient


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
            cls._instance = InMemoryTraceClient()
        return cls._instance

    def __init__(self):
        queue_size = MLFLOW_TRACING_CLIENT_BUFFER_SIZE.get()
        self.queue = deque(maxlen=queue_size)

    def log_trace(self, trace: Trace):
        self.queue.append(trace)

    def get_traces(self, n: int = 10) -> List[Trace]:
        """
        Get the last n traces from the buffer.

        Args:
            n: The number of traces to return.
        Returns:
            A list of Trace objects.
        """
        return list(self.queue) if n is None else list(self.queue)[-n:]

    def _flush(self):
        self.queue.clear()
