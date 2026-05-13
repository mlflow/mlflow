import atexit
import logging
import threading
import time
from collections import OrderedDict

from cachetools import Cache, TTLCache

from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.environment_variables import (
    MLFLOW_TRACE_BUFFER_MAX_SIZE,
    MLFLOW_TRACE_BUFFER_TTL_SECONDS,
    MLFLOW_TRACE_TIMEOUT_CHECK_INTERVAL_SECONDS,
    MLFLOW_TRACE_TIMEOUT_SECONDS,
)
from mlflow.exceptions import MlflowTracingException

_logger = logging.getLogger(__name__)

_TRACE_EXPIRATION_MSG = (
    "Trace {request_id} is timed out after {ttl} seconds. The operation may be stuck or "
    "taking too long to complete. To increase the timeout, set the environment variable "
    "MLFLOW_TRACE_TIMEOUT_SECONDS to a larger value."
)


def get_trace_cache_with_timeout() -> Cache:
    """
    Return a cache object that stores traces in-memory while they are in-progress.

    If the timeout is specified, this returns a customized cache that logs the
    expired traces to the backend. Otherwise, this returns a regular cache.
    """

    if timeout := MLFLOW_TRACE_TIMEOUT_SECONDS.get():
        return MlflowTraceTimeoutCache(
            timeout=timeout,
            maxsize=MLFLOW_TRACE_BUFFER_MAX_SIZE.get(),
        )

    # NB: Ideally we should return the vanilla Cache object only with maxsize.
    # But we used TTLCache before introducing the timeout feature (that does not
    # monitor timeout periodically nor log the expired traces). To keep the
    # backward compatibility, we return TTLCache.
    return TTLCache(
        ttl=MLFLOW_TRACE_BUFFER_TTL_SECONDS.get(),
        maxsize=MLFLOW_TRACE_BUFFER_MAX_SIZE.get(),
    )


class _TimedCache(Cache):
    """
    This code is ported from cachetools library to avoid depending on the private class.
    https://github.com/tkem/cachetools/blob/d44c98407030d2e91cbe82c3997be042d9c2f0de/src/cachetools/__init__.py#L376
    """

    class _Timer:
        def __init__(self, timer):
            self.__timer = timer
            self.__nesting = 0

        def __call__(self):
            if self.__nesting == 0:
                return self.__timer()
            else:
                return self.__time

        def __enter__(self):
            if self.__nesting == 0:
                self.__time = time = self.__timer()
            else:
                time = self.__time
            self.__nesting += 1
            return time

        def __exit__(self, *exc):
            self.__nesting -= 1

        def __reduce__(self):
            return _TimedCache._Timer, (self.__timer,)

        def __getattr__(self, name):
            return getattr(self.__timer, name)

    def __init__(self, maxsize, timer=time.monotonic, getsizeof=None):
        Cache.__init__(self, maxsize, getsizeof)
        self.__timer = _TimedCache._Timer(timer)

    def __repr__(self, cache_repr=Cache.__repr__):
        with self.__timer as time:
            self.expire(time)
            return cache_repr(self)

    def __len__(self, cache_len=Cache.__len__):
        with self.__timer as time:
            self.expire(time)
            return cache_len(self)

    @property
    def currsize(self):
        with self.__timer as time:
            self.expire(time)
            return super().currsize

    @property
    def timer(self):
        """The timer function used by the cache."""
        return self.__timer

    def clear(self):
        with self.__timer as time:
            self.expire(time)
            Cache.clear(self)

    def get(self, *args, **kwargs):
        with self.__timer:
            return Cache.get(self, *args, **kwargs)

    def pop(self, *args, **kwargs):
        with self.__timer:
            return Cache.pop(self, *args, **kwargs)

    def setdefault(self, *args, **kwargs):
        with self.__timer:
            return Cache.setdefault(self, *args, **kwargs)


class MlflowTraceTimeoutCache(_TimedCache):
    """
    A different implementation of cachetools.TTLCache that logs the expired traces to the backend.

    NB: Do not use this class outside a singleton context. This class is not thread-safe.
    """

    def __init__(self, timeout: int, maxsize: int):
        super().__init__(maxsize=maxsize)
        self._timeout = timeout

        # Set up the linked list ordered by expiration time
        self._root = TTLCache._Link()
        self._root.prev = self._root
        self._root.next = self._root
        self._links = OrderedDict()

        self._start_expire_check_loop()

    @property
    def timeout(self) -> int:
        # Timeout should not be changed after the cache is created
        # because the linked list will not be updated accordingly.
        return self._timeout

    def __setitem__(self, key, value, cache_setitem=Cache.__setitem__):
        """Set the item in the cache, and also in the linked list if it is a new key"""
        with self.timer as time:
            cache_setitem(self, key, value)

        if key not in self._links:
            # Add the new item to the tail of the linked list
            # Inspired by https://github.com/tkem/cachetools/blob/d44c98407030d2e91cbe82c3997be042d9c2f0de/src/cachetools/__init__.py#L432
            tail = self._root.prev
            link = TTLCache._Link(key)
            link.expires = time + self._timeout
            link.next = self._root
            link.prev = tail
            tail.next = link
            self._root.prev = link
            self._links[key] = link

    def __delitem__(self, key, cache_delitem=Cache.__delitem__):
        """Delete the item from the cache and the linked list."""
        cache_delitem(self, key)
        link = self._links.pop(key)
        link.unlink()

    def _start_expire_check_loop(self):
        # Close the daemon thread when the main thread exits
        atexit.register(self.clear)

        self._expire_checker_thread = threading.Thread(
            target=self._expire_check_loop, daemon=True, name="TTLCacheExpireLoop"
        )
        self._expire_checker_stop_event = threading.Event()
        self._expire_checker_thread.start()

    def _expire_check_loop(self):
        while not self._expire_checker_stop_event.is_set():
            try:
                self.expire()
            except Exception as e:
                _logger.debug(f"Failed to expire traces: {e}")
                # If an error is raised from the expiration method, stop running the loop.
                # Otherwise, the expire task might get heavier and heavier due to the
                # increasing number of expired items.
                break

            time.sleep(MLFLOW_TRACE_TIMEOUT_CHECK_INTERVAL_SECONDS.get())

    def expire(self, time=None):
        """
        Trigger the expiration of traces that have exceeded the timeout.

        Args:
            time: Unused. Only for compatibility with the parent class.
        """
        expired = self._get_expired_traces()

        # End the expired traces and set the status to ERROR in background thread
        for request_id in expired:
            trace = self[request_id]
            if root_span := trace.get_root_span():
                try:
                    root_span.set_status(SpanStatusCode.ERROR)
                    msg = _TRACE_EXPIRATION_MSG.format(request_id=request_id, ttl=self._timeout)
                    exception_event = SpanEvent.from_exception(MlflowTracingException(msg))
                    root_span.add_event(exception_event)
                    root_span.end()  # Calling end() triggers span export
                    _logger.info(msg + " You can find the aborted trace in the MLflow UI.")
                except Exception as e:
                    _logger.debug(f"Failed to export an expired trace {request_id}: {e}")

                # NB: root_span.end() should pop the trace from the cache. But we need to
                # double-check it because it may not happens due to some errors.
                if request_id in self:
                    del self[request_id]

    def _get_expired_traces(self) -> list[str]:
        """
        Find all expired traces and return their request IDs.

        The linked list is ordered by expiration time, so we can traverse the list from the head
        and return early whenever we find a trace that has not expired yet.
        """
        time = self.timer()
        curr = self._root.next

        if curr.expires and time < curr.expires:
            return []

        expired = []
        while curr is not self._root and not (time < curr.expires):
            expired.append(curr.key)
            curr = curr.next
        return expired

    def clear(self):
        super().clear()
        self._expire_checker_stop_event.set()
        self._expire_checker_thread.join()
