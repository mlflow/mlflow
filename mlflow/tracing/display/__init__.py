from mlflow.tracing.display.display_client import IPythonTraceDisplayClient

__all__ = ["IPythonTraceDisplayClient"]


def get_display_client() -> IPythonTraceDisplayClient:
    return IPythonTraceDisplayClient.get_instance()
