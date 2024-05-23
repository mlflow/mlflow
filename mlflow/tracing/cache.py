import logging
from typing import Any, Callable

from cachetools import TTLCache

from mlflow.tracing.utils import size

_logger = logging.getLogger(__name__)


class SizedTTLCache(TTLCache):
    """
    Extended TTL Cache that keeps track of the size (byte) of the cache records and evicts the
    oldest records when the cache size exceeds the specified maximum size.

    In order to calculate the size of the entry, we need to serialize Based on the implementation
    of TTLCache, the size calculation is only done once when the entry is added to the cache,
    so the serialization cost is not a big concern for typical trace size (~MB).
    https://github.com/tkem/cachetools/blob/d8b5c608273051f2ebddaec8a4d033c66369dd7d/src/cachetools/__init__.py#L74

    Args:
        maxsize_bytes: The maximum size of the cache in bytes.
        ttl: The time-to-live value (sec) for the cache entries.
        serializer: The serializer function to calculate the size of the cache entries.
    """

    def __init__(self, maxsize_bytes: int, ttl: int, serializer: Callable[[Any], str]):
        super().__init__(maxsize=maxsize_bytes, ttl=ttl)
        self._serializer = serializer

    def getsizeof(self, value):
        # NB: For list, the size must be calculated for individual elements and summed up,
        # as sys.getsizeofdoes not handle the size of container well. This is not an accurate
        # size calculation because the list may contain reference to the same object, but it
        # should be good enough for our purpose.
        if isinstance(value, (list, tuple)):
            return sum(size(value, serializer=self._serializer) for value in value)
        return size(value, serializer=self._serializer)

    def __setitem__(self, key, value):
        try:
            super().__setitem__(key, value)
        except ValueError as e:
            # TTLCache by default raises ValueError when the size of the value exceeds the maxsize.
            # We catch the exception and discard the trace instead of blocking the application.
            if "value too large" in str(e):
                _logger.warning(
                    "The size of the trace exceeds the maximum size of the buffer. "
                    "The trace will be dropped. Please consider increasing the buffer "
                    "size by setting the environment variable MLFLOW_TRACE_BUFFER_MAX_SIZE_BYTES."
                )

    def update_size(self, key: str, delta: int):
        """
        Update the size of the cache by the given delta value.

        Args:
            key: The key of the cache entry.
            delta: The size change in bytes of the cache entry.
        """
        # Accessing private attributes is not ideal, but this is the only way to update size
        self._Cache__size[key] += delta
        self._Cache__currsize += delta
        while self.currsize > self.maxsize:
            self.popitem()
