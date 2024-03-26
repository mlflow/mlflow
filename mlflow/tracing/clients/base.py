from abc import ABC, abstractmethod

from mlflow.entities import Trace


class TraceClient(ABC):
    @abstractmethod
    def log_trace(self, trace: Trace):
        pass


class NoOpClient(TraceClient):
    def log_trace(self, trace: Trace):
        pass
