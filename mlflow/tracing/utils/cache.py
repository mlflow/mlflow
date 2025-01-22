import atexit
import logging
import threading
import time
from collections import OrderedDict

from cachetools import Cache, TTLCache, _TimedCache

from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.environment_variables import (
    MLFLOW_TRACE_TIMEOUT_CHECK_INTERVAL_SECONDS,
    MLFLOW_TRACE_TIMEOUT_SECONDS,
)
from mlflow.exceptions import MlflowTracingException

_logger = logging.getLogger(__name__)

_TRACE_EXPIRATION_MSG = (
    "Trace {request_id} is automatically halted by MLflow due to the time-to-live (TTL) "
    "expiration. The operation may be stuck or taking too long to complete. To increase "
    "the TTL duration, set the environment variable MLFLOW_TRACE_TIMEOUT_SECONDS to a "
    "larger value. (Current: {ttl} seconds.)"
)


class MLflowTraceTimeoutCache(_TimedCache):
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
        # TTL may be updated by users after initial creation. Warn users because it does not
        # change the TTL of existing traces and may cause confusion.
        new_timeout = MLFLOW_TRACE_TIMEOUT_SECONDS.get()
        if new_timeout and self._timeout != new_timeout:
            if len(self._links) > 0:
                _logger.warning(
                    f"The timeout of the trace buffer has been updated to {new_timeout} seconds. "
                    "However the timeout won't be applied to existing traces and may cause "
                    "unexpected behavior. To ensure the new timeout is applied correctly, please "
                    "restart the application or update timeout when there is no active trace. "
                )
            self._timeout = new_timeout

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

                # Remove the expired trace from the linked list
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
