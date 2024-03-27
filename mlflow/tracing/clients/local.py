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
        self.queue.append(trace)
        self._display_trace(trace)

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
