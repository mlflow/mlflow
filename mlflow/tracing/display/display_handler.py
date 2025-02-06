import html
import json
import logging
from typing import TYPE_CHECKING
from urllib.parse import urlencode, urljoin

import mlflow
from mlflow.environment_variables import MLFLOW_MAX_TRACES_TO_DISPLAY_IN_NOTEBOOK
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.uri import is_http_uri

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mlflow.entities import Trace


TRACE_RENDERER_ASSET_PATH = "/static-files/lib/notebook-trace-renderer/index.html"

IFRAME_HTML = """
<div>
  <style scoped>
  button {{
    border: none;
    border-radius: 4px;
    background-color: rgb(34, 114, 180);
    font-family: -apple-system, "system-ui", "Segoe UI", Roboto, "Helvetica Neue", Arial;
    font-size: 13px;
    color: white;
    margin-top: 8px;
    margin-bottom: 8px;
    padding: 8px 16px;
    cursor: pointer;
  }}
  button:hover {{
    background-color: rgb(66, 153, 224);
  }}
  </style>
  <button
    onclick="
        const display = this.nextElementSibling.style.display;
        const isCollapsed = display === 'none';
        this.nextElementSibling.style.display = isCollapsed ? null : 'none';

        const verb = isCollapsed ? 'Collapse' : 'Expand';
        this.innerText = `${{verb}} MLflow Trace`;
    "
  >Collapse MLflow Trace</button>
  <iframe
    id="trace-renderer"
    style="width: 100%; height: 500px; border: none; resize: vertical;"
    src="{src}"
  />
</div>
"""


def get_notebook_iframe_html(traces: list["Trace"]):
    # fetch assets from tracking server
    uri = urljoin(mlflow.get_tracking_uri(), TRACE_RENDERER_ASSET_PATH)
    query_string = _get_query_string_for_traces(traces)

    # include mlflow version to invalidate browser cache when mlflow updates
    src = html.escape(f"{uri}?{query_string}&version={mlflow.__version__}")
    return IFRAME_HTML.format(src=src)


def _serialize_trace_list(traces: list["Trace"]):
    return json.dumps(
        # we can't just call trace.to_json() because this
        # will cause the trace to be serialized twice (once
        # by to_json and once by json.dumps)
        [json.loads(trace._serialize_for_mimebundle()) for trace in traces],
        ensure_ascii=False,
    )


def _get_query_string_for_traces(traces: list["Trace"]):
    query_params = []

    for trace in traces:
        query_params.append(("trace_id", trace.info.request_id))
        query_params.append(("experiment_id", trace.info.experiment_id))

    return urlencode(query_params)


def _is_jupyter():
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def is_using_tracking_server():
    return is_http_uri(mlflow.get_tracking_uri())


def is_trace_ui_available():
    # the notebook display feature only works in
    # Databricks notebooks, or in Jupyter notebooks
    # with a tracking server
    return _is_jupyter() and (is_in_databricks_runtime() or is_using_tracking_server())


class IPythonTraceDisplayHandler:
    _instance = None
    disabled = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = IPythonTraceDisplayHandler()
        return cls._instance

    @classmethod
    def disable(cls):
        cls.disabled = True

    @classmethod
    def enable(cls):
        cls.disabled = False
        if cls._instance is None:
            cls._instance = IPythonTraceDisplayHandler()

    def __init__(self):
        self.traces_to_display = {}
        if not _is_jupyter():
            return

        try:
            from IPython import get_ipython

            # Register a post-run cell display hook to display traces
            # after the cell has executed. We don't validate that the
            # user is using a tracking server at this step, because
            # the user might set it later using mlflow.set_tracking_uri()
            get_ipython().events.register("post_run_cell", self._display_traces_post_run)
        except Exception:
            # swallow exceptions. this function is called as
            # a side-effect in a few other functions (e.g. log_trace,
            # get_traces, search_traces), and we don't want to block
            # the core functionality if the display fails.
            _logger.debug("Failed to register post-run cell display hook", exc_info=True)

    def _display_traces_post_run(self, result):
        if self.disabled or not is_trace_ui_available():
            self.traces_to_display = {}
            return

        try:
            from IPython.display import display

            MAX_TRACES_TO_DISPLAY = MLFLOW_MAX_TRACES_TO_DISPLAY_IN_NOTEBOOK.get()
            traces_to_display = list(self.traces_to_display.values())[:MAX_TRACES_TO_DISPLAY]
            if len(traces_to_display) == 0:
                self.traces_to_display = {}
                return

            display(self.get_mimebundle(traces_to_display), raw=True)

            # reset state
            self.traces_to_display = {}
        except Exception:
            # swallow exceptions. this function is called as
            # a side-effect in a few other functions (e.g. log_trace,
            # get_traces, search_traces), and we don't want to block
            # the core functionality if the display fails.
            _logger.error("Failed to display traces", exc_info=True)
            self.traces_to_display = {}

    def get_mimebundle(self, traces: list["Trace"]):
        if len(traces) == 1:
            return traces[0]._repr_mimebundle_()
        else:
            bundle = {"text/plain": repr(traces)}
            if is_in_databricks_runtime():
                bundle["application/databricks.mlflow.trace"] = _serialize_trace_list(traces)
            else:
                bundle["text/html"] = get_notebook_iframe_html(traces)
            return bundle

    def display_traces(self, traces: list["Trace"]):
        if self.disabled or not is_trace_ui_available():
            return

        try:
            if len(traces) == 0:
                return

            traces_dict = {trace.info.request_id: trace for trace in traces}
            self.traces_to_display.update(traces_dict)
        except Exception:
            _logger.debug("Failed to update traces", exc_info=True)
