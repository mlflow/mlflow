from mlflow.tracing.display.display_client import IPythonTraceDisplayClient

__all__ = ["IPythonTraceDisplayClient", "get_display_client"]


def get_display_client() -> IPythonTraceDisplayClient:
    return IPythonTraceDisplayClient.get_instance()
