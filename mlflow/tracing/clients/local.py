import threading
from collections import deque
from trace import Trace
from typing import List, Optional

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
        self._lock = threading.Lock()  # Lock for accessing the queue

        # used for displaying traces in IPython notebooks
        self._prev_execution_count = -1

    def _display_trace(self, trace: Trace):
        """
        Display the trace in an IPython notebook. If multiple
        traces are generated in the same cell, only the last
        one will be displayed.

        This function is a no-op if not running in an IPython
        environment.
        """
        try:
            from IPython import get_ipython
            from IPython.display import display

            if get_ipython() is None:
                return

            # check the current exec count to know if the user is
            # running the command in a new cell or not. this is
            # useful because the user might generate multiple traces
            # in a single cell, and we only want to display the last one.
            current_exec_count = get_ipython().execution_count
            if self._prev_execution_count != current_exec_count:
                self._prev_execution_count = current_exec_count
                self.display_handle = display(trace, display_id=True)
            else:
                self.display_handle.update(trace)
        except Exception:
            pass

    def log_trace(self, trace: Trace):
        with self._lock:
            self.queue.append(trace)
        self._display_trace(trace)

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
                if trace.trace_info.request_id == request_id:
                    return trace

    def _flush(self):
        self.queue.clear()
