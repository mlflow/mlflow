import atexit
from collections import OrderedDict
import logging
import threading
import time
from typing import Any, Optional
from cachetools import _TimedCache

from mlflow.environment_variables import MLFLOW_TRACE_TTL_CHECK_INTERVAL_SECONDS


_logger = logging.getLogger(__name__)

class TTLCache(_TimedCache):
    class _Link:
        __slots__ = ("key", "expires", "next", "prev")

        def __init__(self, key=None, expires=None):
            self.key = key
            self.expires = expires

        def __reduce__(self):
            return TTLCache._Link, (self.key, self.expires)

        def unlink(self):
            next = self.next
            prev = self.prev
            prev.next = next
            next.prev = prev

    def __init__(self, maxsize: int, ttl: int):
        super().__init__(maxsize)

        self.__root = root = TTLCache._Link()
        root.prev = root.next = root
        self.__links = OrderedDict()
        self.__ttl = ttl

        # Start the background thread only when the first item is added
        self._lock = threading.RLock()
        self._event_loop_started = False


    def _start_expire_loop(self):
        with self._lock:
            if self._event_loop_started:
                return

            # Close the thread when the main thread exits
            atexit.register(self._shutdown)

            self._expire_checker_thread = threading.Thread(target=self._run_expire_loop, daemon=True, name="TTLCacheExpireLoop")
            self._expire_checker_stop_event = threading.Event()
            self._expire_checker_thread.start()
            self._event_loop_started = True

    def __setitem__(self, key, value):
        with self.timer as time:
            super().__setitem__(key, value)

        try:
            link = self.__links[key]
            self.__links.move_to_end(key)
            link.unlink()
        except KeyError:
            link = TTLCache._Link(key)
            self.__links[key] = link

        link.expires = time + self.__ttl
        link.next = root = self.__root
        link.prev = prev = root.prev
        prev.next = root.prev = link

        if not self._event_loop_started:
            self._start_expire_loop()


    def __delitem__(self, key):
        super().__delitem__(key)
        link = self.__links.pop(key)
        link.unlink()


    def _run_expire_loop(self):
        while not self._expire_checker_stop_event.is_set():
            try:
                self.expire()
            except Exception as e:
                _logger.debug(f"Failed to expire traces: {e}")
                # If an error is raised from the expiration method, stop running the loop.
                # Otherwise, the expire task might get heavier and heavier due to the
                # increasing number of expired items.
                break

            time.sleep(MLFLOW_TRACE_TTL_CHECK_INTERVAL_SECONDS.get())


    def _get_all_expired(self) -> list[tuple[str, Any]]:
        # Find all expired traces. Not removing them here because
        # because span exporter may access the trace stored in the cache
        # Ref: https://github.com/tkem/cachetools/blob/d44c98407030d2e91cbe82c3997be042d9c2f0de/src/cachetools/__init__.py#L469-L489
        time = self.timer() # get current time
        curr = self.__root.next
        expired = []
        while curr is not self.__root and not (time < curr.expires):
            expired.append((curr.key, self[curr.key]))
            curr = curr.next

        return expired

    def expire(self, time=None):
        if time is None:
            time = self.timer()

        expired = self._get_all_expired()
        for key, _ in expired:
            del self[key]
        return expired

    def _shutdown(self):
        self.clear()
        if self._event_loop_started:
            self._expire_checker_stop_event.set()
