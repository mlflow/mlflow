import json
import logging
from typing import List

from mlflow.entities import Trace
from mlflow.environment_variables import MLFLOW_MAX_TRACES_TO_DISPLAY_IN_NOTEBOOK
from mlflow.utils.databricks_utils import is_in_databricks_runtime

_logger = logging.getLogger(__name__)


def _serialize_trace_list(traces: List[Trace]):
    return json.dumps(
        # we can't just call trace.to_json() because this
        # will cause the trace to be serialized twice (once
        # by to_json and once by json.dumps)
        [json.loads(trace._serialize_for_mimebundle()) for trace in traces],
        ensure_ascii=False,
    )


class IPythonTraceDisplayHandler:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = IPythonTraceDisplayHandler()
        return cls._instance

    def __init__(self):
        # This only works in Databricks notebooks
        if not is_in_databricks_runtime():
            return

        self.traces_to_display = {}
        try:
            from IPython import get_ipython

            if get_ipython() is None:
                return

            # Register a post-run cell display hook to display traces
            # after the cell has executed.
            get_ipython().events.register("post_run_cell", self._display_traces_post_run)
        except Exception:
            # swallow exceptions. this function is called as
            # a side-effect in a few other functions (e.g. log_trace,
            # get_traces, search_traces), and we don't want to block
            # the core functionality if the display fails.
            _logger.debug("Failed to register post-run cell display hook", exc_info=True)

    def _display_traces_post_run(self, result):
        # this should do nothing if not in an IPython environment
        try:
            from IPython import get_ipython
            from IPython.display import display

            if get_ipython() is None:
                return

            MAX_TRACES_TO_DISPLAY = MLFLOW_MAX_TRACES_TO_DISPLAY_IN_NOTEBOOK.get()
            traces_to_display = list(self.traces_to_display.values())[:MAX_TRACES_TO_DISPLAY]
            if len(traces_to_display) == 0:
                self.traces_to_display = {}
                return

            display(
                self.get_mimebundle(traces_to_display),
                display_id=True,
                raw=True,
            )

            # reset state
            self.traces_to_display = {}
        except Exception:
            # swallow exceptions. this function is called as
            # a side-effect in a few other functions (e.g. log_trace,
            # get_traces, search_traces), and we don't want to block
            # the core functionality if the display fails.
            _logger.debug("Failed to display traces", exc_info=True)

    def get_mimebundle(self, traces: List[Trace]):
        if len(traces) == 1:
            return traces[0]._repr_mimebundle_()
        else:
            return {
                "application/databricks.mlflow.trace": _serialize_trace_list(traces),
                "text/plain": repr(traces),
            }

    def display_traces(self, traces: List[Trace]):
        # This only works in Databricks notebooks
        if not is_in_databricks_runtime():
            return

        # this should do nothing if not in an IPython environment
        try:
            from IPython import get_ipython

            if len(traces) == 0 or get_ipython() is None:
                return

            traces_dict = {trace.info.request_id: trace for trace in traces}
            self.traces_to_display.update(traces_dict)
        except Exception:
            _logger.debug("Failed to update traces", exc_info=True)
