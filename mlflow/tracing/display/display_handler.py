import json
import logging
from typing import List

from mlflow.entities import Trace

MAX_TRACES_TO_DISPLAY = 100


_logger = logging.getLogger(__name__)


def _serialize_trace_list(traces: List[Trace]):
    return json.dumps(
        # we can't just call trace.to_json() because this
        # will cause the trace to be serialized twice (once
        # by to_json and once by json.dumps)
        [json.loads(trace.to_json()) for trace in traces]
    )


class IPythonTraceDisplayHandler:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = IPythonTraceDisplayHandler()
        return cls._instance

    def __init__(self):
        self._prev_execution_count = -1
        self.display_handle = None
        self.traces_to_display = {}

    def get_mimebundle(self, traces: List[Trace]):
        if len(traces) == 1:
            return traces[0]._repr_mimebundle_()
        else:
            return {
                "application/databricks.mlflow.trace": _serialize_trace_list(traces),
                "text/plain": repr(traces),
            }

    def display_traces(self, traces: List[Trace]):
        # this should do nothing if not in an IPython environment
        try:
            from IPython import get_ipython
            from IPython.display import display

            if len(traces) == 0 or get_ipython() is None:
                return

            traces = traces[:MAX_TRACES_TO_DISPLAY]
            traces_dict = {trace.info.request_id: trace for trace in traces}

            # if the current ipython exec count has changed, then
            # we're in a different cell (or rerendering the current
            # cell), so we should create a new display handle.
            current_exec_count = get_ipython().execution_count
            if self._prev_execution_count != current_exec_count:
                self._prev_execution_count = current_exec_count
                self.traces_to_display = traces_dict
                self.display_handle = None
            else:
                self.traces_to_display.update(traces_dict)

            deduped_trace_list = list(self.traces_to_display.values())
            if self.display_handle:
                self.display_handle.update(
                    self.get_mimebundle(deduped_trace_list),
                    raw=True,
                )
            else:
                self.display_handle = display(
                    self.get_mimebundle(deduped_trace_list),
                    display_id=True,
                    raw=True,
                )
        except Exception:
            # swallow exceptions. this function is called as
            # a side-effect in a few other functions (e.g. log_trace,
            # get_traces, search_traces), and we don't want to block
            # the core functionality if the display fails.
            _logger.debug("Failed to display traces", exc_info=True)
