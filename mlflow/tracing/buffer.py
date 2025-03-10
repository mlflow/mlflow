from threading import RLock
from typing import Any, Optional, Union

from cachetools import TTLCache

from mlflow.entities import Trace
from mlflow.environment_variables import (
    MLFLOW_TRACE_BUFFER_MAX_SIZE,
    MLFLOW_TRACE_BUFFER_TTL_SECONDS,
)


class _TraceBuffer:
    """
    A thread-safe buffer that stores traces in memory with a TTL.
    """

    def __init__(self, max_size: int, ttl_seconds: int):
        self._cache = TTLCache(
            maxsize=max_size,
            ttl=ttl_seconds,
        )
        self._buffer_lock = RLock()

    def insert(self, request_id: str, trace: Trace):
        """
        Insert the specified trace to the buffer and associate it with the specified request ID.
        This method is thread-safe.
        """
        with self._buffer_lock:
            self._cache[request_id] = trace

    def get(self, request_id: str, default=None) -> Union[Trace, Any]:
        """
        Get the trace with the specified request ID from the buffer. If the trace is not found,
        return the default value. This method is thread-safe.
        """
        with self._buffer_lock:
            return self._cache.get(request_id, default)

    def pop(self, request_id: str, default=None) -> Union[Trace, Any]:
        """
        Remove and return the trace with the specified request ID from the buffer. If the trace is
        not found, return the default value. This method is thread-safe.
        """
        with self._buffer_lock:
            return self._cache.pop(request_id, default)

    def latest(self) -> Optional[Trace]:
        """
        Get the latest trace from the buffer. This method is thread-safe.

        Returns:
            The latest trace in the buffer, or None if the buffer is empty.
        """
        with self._buffer_lock:
            if len(self._cache) > 0:
                last_active_request_id = list(self._cache.keys())[-1]
                return self._cache.get(last_active_request_id)

    def clear(self):
        """
        Clear all traces from the buffer. This method is thread-safe.
        """
        with self._buffer_lock:
            self._cache.clear()

    def __len__(self) -> int:
        """
        Get the number of traces in the buffer. This method is thread-safe.

        Returns:
            The number of traces in the buffer.
        """
        with self._buffer_lock:
            return len(self._cache)


# Traces are stored in memory after completion so they can be retrieved conveniently.
# For example, Databricks model serving fetches the trace data from the buffer after
# making the prediction request, and logging them into the Inference Table.
TRACE_BUFFER = _TraceBuffer(
    max_size=MLFLOW_TRACE_BUFFER_MAX_SIZE.get(),
    ttl_seconds=MLFLOW_TRACE_BUFFER_TTL_SECONDS.get(),
)

__all__ = ["TRACE_BUFFER"]
