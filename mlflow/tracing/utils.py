import inspect
import json
import logging
from typing import Any, Dict

_logger = logging.getLogger(__name__)


def capture_function_input_args(func, args, kwargs) -> Dict[str, Any]:
    try:
        # Avoid capturing `self`
        func_signature = inspect.signature(func)
        bound_arguments = func_signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        # Remove `self` from bound arguments if it exists
        if bound_arguments.arguments.get("self"):
            del bound_arguments.arguments["self"]

        return bound_arguments.arguments
    except Exception:
        _logger.warning(f"Failed to capture inputs for function {func.__name__}.")
        return {}


def _serialize_trace_list(traces):
    return json.dumps([json.loads(trace.to_json()) for trace in traces])


def display_traces(traces, display_handle=None):
    try:
        from IPython import get_ipython
        from IPython.display import display

        if len(traces) == 0 or get_ipython() is None:
            return

        if len(traces) == 1:
            mimebundle = traces[0]._repr_mimebundle_()
        else:
            mimebundle = {
                "application/databricks.mlflow.trace": _serialize_trace_list(traces),
                "text/plain": traces.__repr__(),
            }

        if display_handle:
            display_handle.update(
                mimebundle,
                raw=True,
            )
        else:
            display_handle = display(
                mimebundle,
                display_id=True,
                raw=True,
            )

        return display_handle
    except Exception:
        pass
