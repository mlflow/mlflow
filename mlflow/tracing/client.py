from abc import ABC, abstractmethod

from mlflow.tracing.types.model import Trace


def get_trace_client():
    # TODO: There will be a real implementation of the trace client
    #  E.g. https://github.com/B-Step62/mlflow/blob/trace-api-poc/mlflow/traces/client.py
    return NoOpClient()


class TraceClient(ABC):
    @abstractmethod
    def log_trace(self, trace: Trace):
        pass


class NoOpClient(TraceClient):
    def log_trace(self, trace: Trace):
        pass
