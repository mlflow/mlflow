"""
Daemon-thread-backed worker pool used by the async logging queues.

`concurrent.futures.ThreadPoolExecutor` registers its worker threads with
`concurrent.futures.thread._threads_queues` and installs `_python_exit` via
`threading._register_atexit`. During interpreter shutdown `_python_exit` joins
every tracked worker with no timeout, executing *before* any
`atexit.register` hook. When a worker is blocked on network I/O the process
hangs indefinitely and `_at_exit_callback` never runs (the symptom in #20575).

`DaemonThreadPool` is API-compatible with the small subset of
`ThreadPoolExecutor` used by the async logging queues -- `submit(fn, *args)`
returning a `concurrent.futures.Future`, and `shutdown(wait=, timeout=)`. Its
workers are plain daemon `threading.Thread`s, so they are not joined by
`_python_exit` and the interpreter can finalize even when a worker is stuck.
"""

import logging
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable

_logger = logging.getLogger(__name__)

_SHUTDOWN_SENTINEL = object()


class DaemonThreadPool:
    def __init__(self, max_workers: int, thread_name_prefix: str = "DaemonThreadPool") -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self._queue: queue.Queue = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        self._thread_name_prefix = thread_name_prefix
        for i in range(max_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"{thread_name_prefix}_{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SHUTDOWN_SENTINEL:
                return
            fn, args, kwargs, fut = item
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                fut.set_exception(exc)
            else:
                fut.set_result(result)

    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> Future:
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            fut: Future = Future()
            self._queue.put((fn, args, kwargs, fut))
            return fut

    def shutdown(self, wait: bool = True, timeout: float | None = None) -> None:
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True
            for _ in self._workers:
                self._queue.put(_SHUTDOWN_SENTINEL)
        if not wait:
            return
        deadline = None if timeout is None else time.monotonic() + timeout
        for t in self._workers:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            t.join(timeout=remaining)
