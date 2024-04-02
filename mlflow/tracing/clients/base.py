from abc import ABC, abstractmethod

from mlflow.entities import Trace


class TraceClient(ABC):
    @abstractmethod
    def get_trace(self, request_id: str):
        """Get the trace with the given request_id."""
        pass

    @abstractmethod
    def log_trace(self, trace: Trace):
        """Log the given trace."""
        pass
