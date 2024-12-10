from mlflow.tracing.display.display_handler import IPythonTraceDisplayHandler

__all__ = ["IPythonTraceDisplayHandler", "get_display_handler"]


def get_display_handler() -> IPythonTraceDisplayHandler:
    return IPythonTraceDisplayHandler.get_instance()


def disable_notebook_display():
    """
    Disables displaying the MLflow Trace UI in notebook output cells.
    Call :py:func:`mlflow.tracing.enable_notebook_display()` to re-enable display.
    """
    IPythonTraceDisplayHandler.disable()


def enable_notebook_display():
    """
    Enables the MLflow Trace UI in notebook output cells. The display is on
    by default, and the Trace UI will show up when any of the following operations
    are executed:

    * On trace completion (i.e. whenever a trace is exported)
    * When calling the :py:func:`mlflow.search_traces` fluent API
    * When calling the :py:meth:`mlflow.client.MlflowClient.get_trace`
      or :py:meth:`mlflow.client.MlflowClient.search_traces` client APIs

    To disable, please call :py:func:`mlflow.tracing.disable_notebook_display()`.
    """
    IPythonTraceDisplayHandler.enable()
