from trace import Trace

from mlflow.tracing.clients.base import TraceClient
from mlflow.tracing.clients.local import InMemoryTraceClient


class IPythonTraceWrapper(InMemoryTraceClient):
    """
    Trace client that wraps the InMemoryTraceClient
    in IPython notebooks. The only purpose of this
    class is to add a hook on log_trace() to display
    the contents of the trace.
    """

    _instance: TraceClient = None

    @classmethod
    def get_instance(cls) -> InMemoryTraceClient:
        if cls._instance is None:
            cls._instance = IPythonTraceWrapper()
        return cls._instance

    def __init__(self):
        self.prev_execution_count = -1
        super().__init__()

    def log_trace(self, trace: Trace):
        from IPython import get_ipython
        from IPython.display import display

        # check the current exec count to know if the user is
        # running the command in a new cell or not. this is
        # useful because the user might generate multiple traces
        # in a single cell, and we only want to display the last one.
        current_exec_count = get_ipython().execution_count
        if self.prev_execution_count != current_exec_count:
            self.prev_execution_count = current_exec_count
            self.display_handle = display(trace, display_id=True)
        else:
            self.display_handle.update(trace)

        return super().log_trace(trace)
