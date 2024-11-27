from mlflow.tracing.display import IPythonTraceDisplayHandler
from mlflow.tracing.provider import disable, enable

__all__ = ["disable", "enable", "disable_notebook_display", "enable_notebook_display"]


def disable_notebook_display():
    IPythonTraceDisplayHandler.disable()


def enable_notebook_display():
    IPythonTraceDisplayHandler.enable()
