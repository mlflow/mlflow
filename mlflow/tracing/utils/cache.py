import logging
import threading
from typing import Optional

from cachetools import TTLCache

from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.environment_variables import MLFLOW_TRACE_BUFFER_TTL_SECONDS
from mlflow.exceptions import MlflowTracingException
from mlflow.tracing.trace_manager import _Trace

_logger = logging.getLogger(__name__)

_TRACE_EXPIRATION_MSG = (
    "This trace is automatically halted by MLflow due to the time-to-live (TTL) expiration. "
    "The operation may be stuck or taking too long to complete. To increase the TTL duration, "
    "set the environment variable MLFLOW_TRACE_BUFFER_TTL_SECONDS to a larger value. "
    "(Current: {ttl} seconds.)"
)


class TTLCacheWithLogging(TTLCache):
    """An extension of cachetools.TTLCache that logs the expired traces to the backend."""

    def __init__(self, maxsize: int, ttl: int):
        super().__init__(maxsize=maxsize, ttl=ttl)
        self._expire_traces_lock = threading.Lock()

    def expire(self, time: Optional[int] = None, block=False):
        """
        Trigger the TTL cache expiration (non-blocking).

        In addition to removing the expired traces from the cache, this method also mark it
        completed with an error status, and log it to the backend, so users can see the
        timeout traces on the UI. Since logging takes time, it is done in a background
        thread and this method returns immediately.

        Args:
            time: This parameter is not used but kept for compatibility with the base class.
            block: If True, wait for the expiration to complete before returning.
        """
        # TTL may be updated by users after initial creation
        if self.ttl != MLFLOW_TRACE_BUFFER_TTL_SECONDS.get():
            self._TTLCache__ttl = MLFLOW_TRACE_BUFFER_TTL_SECONDS.get()

        expired = self._get_expired_traces()
        if not expired:
            return

        # End the expired traces and set the status to ERROR in background thread
        def _expire_traces():
            for request_id, trace in expired:
                if root_span := trace.get_root_span():
                    try:
                        root_span.set_status(SpanStatusCode.ERROR)
                        exception_event = SpanEvent.from_exception(
                            MlflowTracingException(_TRACE_EXPIRATION_MSG.format(ttl=self.ttl))
                        )
                        root_span.add_event(exception_event)
                        root_span.end()  # Calling end() triggers span export

                        _logger.info(f"Trace `{request_id}` is aborted due to TTL expiration.")

                    except Exception:
                        _logger.debug(f"Failed to expire a trace {request_id}", exc_info=True)

                # NB: Remove the trace from the cache after logging it, because
                #   the span exporter requires the trace to be in the cache
                del self[request_id]

        thread = threading.Thread(target=_expire_traces, daemon=True)
        thread.start()
        if block:
            thread.join()

    def __getitem__(self, key):
        """
        The original TTLCache only trigger expiration when an item is set or deleted.
        This is not enough, because it means a trace is only expired when a new trace is created.

        To increase the cadence of expiring the traces, we trigger expiration whenever an item
        is accessed. Checking the expired trace is fairy lightweight if there is no expired span
        (most of the time). Even if there are, the expiration is done in a non-blocking way,
        so the performance impact should be minimal.

        NB: This is still a 'best-effort' approach, because expiration won't be triggered when
        there is only one span running and it hangs forever.
        """
        self.expire(block=False)
        return super().__getitem__(key)

    def get(self, key):
        # Call expire() with the same reason as __getitem__.
        self.expire(block=False)
        return super().get(key)

    def _get_expired_traces(self) -> list[tuple[str, _Trace]]:
        """
        Fine all TTL expired traces.
        Ref: https://github.com/tkem/cachetools/blob/d44c98407030d2e91cbe82c3997be042d9c2f0de/src/cachetools/__init__.py#L469-L489
        """
        time = self.timer()
        root = self._TTLCache__root
        curr = root.next

        if curr.expires and time < curr.expires:
            return []

        expired = []
        # Traversal linked list to find expired traces (linear time to number of expired traces)
        # Requires a lock to ensure only one thread is checking the linked list at a time
        with self._expire_traces_lock:
            while curr is not root and not (time < curr.expires):
                # Set the expiration time to a far future to avoid expiring it twice
                curr.expires = 1e9
                self._TTLCache__links.move_to_end(curr.key)
                expired.append((curr.key, self[curr.key]))
                curr = curr.next
        return expired
