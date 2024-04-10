from mlflow.tracing.display.display_client import IPythonTraceDisplayHandler

__all__ = ["IPythonTraceDisplayHandler", "get_display_client"]


def get_display_client() -> IPythonTraceDisplayHandler:
    return IPythonTraceDisplayHandler.get_instance()
